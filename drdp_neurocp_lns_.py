"""
DRDP-NeuroCP-ALNS: Adaptive Neural Large Neighborhood Search for DRDP
AAAI Reference Implementation

This module implements the DRDP-NeuroCP-ALNS architecture:
- Always-feasible O(deg) local state engine (labels in {0,2,3})
- GraphSAGE-lite (FP16) to score "unlock" sets; Gumbel-Top-k for diversity
- Adaptive Operator Selection (GNN k-hop vs. Deep Random Walk)
- Curriculum Region Sizing (Dynamic LNS bounds)
- Local DRDP-1' ILP solved by OR-Tools CP-SAT with boundary protection
- Advantage-Weighted Regression (AWR) online learning of unlock scores
- Elite pool and short path-relinking for intensification
"""

import os
import sys
import gzip
import time
import math
import random
import argparse
import gc
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Deque
from collections import deque
import numpy as np

# ---- Torch (GPU) ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.cuda.amp as amp

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ---- OR-Tools CP-SAT ----
try:
    from ortools.sat.python import cp_model
    import builtins as _bi

    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False


# ========================= IO =========================

def read_mtx_gz(path: str) -> Tuple[int, List[List[int]]]:
    """Reads a sparse graph from a gzipped Matrix Market format."""
    with gzip.open(path, 'rt') as f:
        line = f.readline()
        while line and (line.strip() == '' or line[0] in 'c%'):
            line = f.readline()
        if not line:
            raise ValueError("Invalid MTX header.")
        r, c, _ = map(int, line.strip().split()[:3])
        n = max(r, c)
        adj = [set() for _ in range(n)]
        for l in f:
            if not l.strip() or l[0] in 'c%': continue
            u, v = map(lambda x: int(x) - 1, l.split()[:2])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].add(v)
                adj[v].add(u)
    return n, [list(s) for s in adj]


# ================= DRDP Core Engine =================

class DRDPCore:
    """
    Always-Feasible Core for DRDP-1'.
    Maintains labels S[v] in {0,2,3} and incremental neighbor counts (n2, n3).
    Ensures O(deg) updates and feasibility by construction (Eq. 2).
    """

    def __init__(self, n: int, neigh: List[List[int]], seed: int = 0):
        self.n = n
        self.neigh = neigh
        self.rng = np.random.default_rng(seed)

        self.S = np.zeros(n, dtype=np.int8)
        self.n2 = np.zeros(n, dtype=np.int32)
        self.n3 = np.zeros(n, dtype=np.int32)
        self.viol = np.ones(n, dtype=np.int8)
        self.viol_count = int(self.viol.sum())
        self.add_count = np.zeros(n, dtype=np.int16)

        self.deg = np.array([len(neigh[u]) for u in range(n)], dtype=np.int32)
        self.maxdeg = int(max(1, self.deg.max()))

    def cost(self) -> int:
        return int(self.S.sum())

    def copy_snapshot(self) -> Tuple:
        return (self.S.copy(), self.n2.copy(), self.n3.copy(),
                self.viol.copy(), int(self.viol_count), self.add_count.copy())

    def restore_snapshot(self, snap: Tuple):
        self.S[:], self.n2[:], self.n3[:], self.viol[:], self.viol_count, self.add_count[:] = snap

    def _set_label(self, u: int, new: int):
        """O(deg) state update for label change."""
        old = int(self.S[u])
        if old == new: return

        if old == 3:
            for v in self.neigh[u]: self.n3[v] -= 1
        elif old == 2:
            for v in self.neigh[u]: self.n2[v] -= 1

        self.S[u] = new
        if new in (2, 3):
            self.add_count[u] = min(32767, self.add_count[u] + 1)

        if new == 3:
            for v in self.neigh[u]: self.n3[v] += 1
        elif new == 2:
            for v in self.neigh[u]: self.n2[v] += 1

        # Update neighbors' violation flags
        for v in self.neigh[u]:
            if self.S[v] == 0:
                sat = (self.n3[v] >= 1) or (self.n2[v] >= 2)
                if sat and self.viol[v]:
                    self.viol[v] = 0;
                    self.viol_count -= 1
                elif (not sat) and (not self.viol[v]):
                    self.viol[v] = 1;
                    self.viol_count += 1

        # Update u's own violation flag
        if new == 0:
            sat = (self.n3[u] >= 1) or (self.n2[u] >= 2)
            if sat and self.viol[u]:
                self.viol[u] = 0;
                self.viol_count -= 1
            elif (not sat) and (not self.viol[u]):
                self.viol[u] = 1;
                self.viol_count += 1
        elif old == 0 and new in (2, 3):
            if self.viol[u]:
                self.viol[u] = 0;
                self.viol_count -= 1

    def greedy_init(self):
        """Constructive heuristic: iteratively raises covering nodes to 3."""
        while self.viol_count > 0:
            viol_idx = np.flatnonzero(self.viol)
            score = np.zeros(self.n, dtype=np.int32)
            for v in viol_idx:
                for u in self.neigh[v]:
                    if self.S[u] != 3: score[u] += 1
            u = int(np.argmax(score))
            if score[u] == 0:
                u = int(viol_idx[self.rng.integers(len(viol_idx))])
            self._set_label(u, 3)
        self.prune_full(4)
        assert self.viol_count == 0, "greedy_init produced infeasible state."

    def prune_full(self, max_passes: int = 3) -> bool:
        """Applies safe demotions iteratively to remove redundancies."""
        changed_any = False
        for _ in range(max_passes):
            changed = False
            th = [u for u in range(self.n) if self.S[u] == 3]
            random.shuffle(th)
            for u in th:
                if self._safe_demote3_to2(u):
                    self._set_label(u, 2)
                    changed = True
                    changed_any = True

            tw = [u for u in range(self.n) if self.S[u] == 2]
            random.shuffle(tw)
            for u in tw:
                if self._safe_demote2_to0(u):
                    self._set_label(u, 0)
                    changed = True
                    changed_any = True
            if not changed: break
        return changed_any

    def _safe_demote3_to2(self, u: int) -> bool:
        """Condition (a) from paper: verifies 3->2 safety."""
        for v in self.neigh[u]:
            if self.S[v] == 0 and (2 * self.n3[v] + self.n2[v] < 3):
                return False
        return True

    def _safe_demote2_to0(self, u: int) -> bool:
        """Condition (b) from paper: verifies 2->0 safety."""
        if not (self.n3[u] >= 1 or self.n2[u] >= 2):
            return False
        for v in self.neigh[u]:
            if self.S[v] == 0 and (2 * self.n3[v] + self.n2[v] < 3):
                return False
        return True

    def compute_private_support(self):
        priv3 = np.zeros(self.n, dtype=np.float32)
        priv2 = np.zeros(self.n, dtype=np.float32)
        for v in range(self.n):
            if self.S[v] != 0: continue
            c3, c2 = self.n3[v], self.n2[v]
            if c3 == 1 and c2 < 2:
                for u in self.neigh[v]:
                    if self.S[u] == 3:
                        priv3[u] += 1
                        break
            if c3 == 0 and c2 == 2:
                tw = []
                for u in self.neigh[v]:
                    if self.S[u] == 2:
                        tw.append(u)
                        if len(tw) == 2: break
                if len(tw) == 2:
                    priv2[tw[0]] += 1
                    priv2[tw[1]] += 1
        d = float(max(1, self.maxdeg))
        return priv3 / d, priv2 / d

    def node_features(self) -> np.ndarray:
        """Extracts the 10-dimensional node feature vector."""
        priv3, priv2 = self.compute_private_support()
        X = np.zeros((self.n, 10), dtype=np.float32)
        X[:, 0] = (self.S == 0).astype(np.float32)
        X[:, 1] = (self.S == 2).astype(np.float32)
        X[:, 2] = (self.S == 3).astype(np.float32)
        X[:, 3] = self.deg / float(self.maxdeg)
        X[:, 4] = self.n2 / float(self.maxdeg)
        X[:, 5] = self.n3 / float(self.maxdeg)
        X[:, 6] = self.viol.astype(np.float32)
        X[:, 7] = priv2
        X[:, 8] = priv3
        X[:, 9] = np.minimum(1.0, self.add_count / 20.0)
        return X

    def global_features(self, stagn: int) -> np.ndarray:
        """Extracts the 9-dimensional global feature vector."""
        n = max(1, int(self.n))
        maxdeg = max(1.0, float(self.maxdeg))
        frac0 = float(np.count_nonzero(self.S == 0)) / n
        frac2 = float(np.count_nonzero(self.S == 2)) / n
        frac3 = float(np.count_nonzero(self.S == 3)) / n
        g = np.array([
            np.log10(n + 1.0) / 3.0,
            float(self.deg.mean()) / maxdeg,
            maxdeg / float(n),
            frac0,
            frac2,
            frac3,
            float(self.viol_count) / n,
            float(self.n2.mean()) / maxdeg,
            min(1.0, float(stagn) / 10.0),
        ], dtype=np.float32)
        return g

    def pair_candidates(self) -> List[Tuple[int, int, int]]:
        """Identifies (v, a, b) motifs for pair-swap intensification."""
        out = []
        for v in range(self.n):
            if self.S[v] == 0 and self.n3[v] == 0 and self.n2[v] == 2:
                tw = []
                for u in self.neigh[v]:
                    if self.S[u] == 2:
                        tw.append(u)
                        if len(tw) == 2: break
                if len(tw) == 2:
                    out.append((v, tw[0], tw[1]))
        return out

    def pass_pair(self, triples: Iterable[Tuple[int, int, int]], limit: int = 32):
        ch = 0
        acc = []
        for (v, a, b) in triples:
            if not (self.S[v] == 0 and self.n3[v] == 0 and self.n2[v] == 2): continue
            c0 = self.cost()
            self._set_label(a, 3);
            self._set_label(b, 0)
            if self.viol_count == 0 and self.cost() < c0:
                ch += 1;
                acc.append((a, b))
            else:
                self._set_label(b, 2);
                self._set_label(a, 2)
                self._set_label(b, 3);
                self._set_label(a, 0)
                if self.viol_count == 0 and self.cost() < c0:
                    ch += 1;
                    acc.append((b, a))
                else:
                    self._set_label(a, 2);
                    self._set_label(b, 2)
            if ch >= limit: break
        if ch: self.prune_full(2)
        return ch, acc


# =================== Neural Unlock Policy (GraphSAGE-lite) ===================

def build_norm_adj(n: int, neigh: List[List[int]], device):
    """Builds symmetrically normalized adjacency matrix."""
    if not TORCH_OK: return None
    row, col = [], []
    deg = np.ones(n, dtype=np.float32)
    for u in range(n):
        for v in neigh[u]:
            row.append(u);
            col.append(v)
            deg[u] += 1.0
        row.append(u);
        col.append(u)

    row = torch.tensor(row, dtype=torch.long, device=device)
    col = torch.tensor(col, dtype=torch.long, device=device)
    d = torch.tensor(deg, dtype=torch.float32, device=device)
    val = 1.0 / torch.sqrt(d[row] * d[col])
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, size=(n, n)).coalesce()


if TORCH_OK:
    class GraphSAGELayer(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.W = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, H, A):
            with torch.autocast(device_type=H.device.type, enabled=False):
                agg = torch.sparse.mm(A.float(), H.float()).to(H.dtype)
            return self.W(agg)


    class GraphSAGELite(nn.Module):
        """3-Layer GraphSAGE-lite encoder for DRDP features."""

        def __init__(self, in_dim=10, hid=128, layers=3):
            super().__init__()
            self.convs = nn.ModuleList([GraphSAGELayer(in_dim if i == 0 else hid, hid) for i in range(layers)])
            self.lns = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

        def forward(self, X, A_sparse):
            H = X
            for conv, ln in zip(self.convs, self.lns):
                H = ln(F.elu(conv(H, A_sparse)))
            return H


    class Heads(nn.Module):
        """Task-specific heads for Unlock Score (s) and Value (v)."""

        def __init__(self, hid=128):
            super().__init__()
            self.unlock = nn.Linear(hid, 1)
            self.value = nn.Sequential(nn.Linear(2 * hid + 9, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, H, g):
            s = self.unlock(H).squeeze(-1)
            hmean, hmax = H.mean(0), H.max(0).values
            v = self.value(torch.cat([hmean, hmax, g], dim=-1)).squeeze(-1)
            return s, v


# Numpy fallback classes omitted for brevity
class SAGE_Numpy:
    def __init__(self, in_dim=10, hid=128, layers=3): pass

    def forward(self, X, neigh_adj): return X @ np.random.randn(X.shape[1], 128)


class Heads_Numpy:
    def __init__(self, hid=128): pass

    def forward(self, H, g): return np.random.randn(H.shape[0]), 0.0


# =================== Search Utilities ===================

def gumbel_top_k(scores: np.ndarray, K: int) -> List[int]:
    if K <= 0 or len(scores) == 0: return []
    K = min(K, len(scores))
    g = -np.log(-np.log(np.random.rand(*scores.shape) + 1e-9) + 1e-9)
    y = scores + g
    idx = np.argpartition(-y, K - 1)[:K]
    return idx[np.argsort(-y[idx])].tolist()


@dataclass(eq=False)
class PoolEntry:
    S: np.ndarray
    cost: int


class ElitePool:
    def __init__(self, size=8, min_hamming_frac=0.05):
        self.size = size
        self.minham = min_hamming_frac
        self.pool: List[PoolEntry] = []

    def _hamm(self, A, B):
        return int(np.count_nonzero(A != B))

    def try_add(self, S):
        c = int(S.sum())
        kill_idx = [i for i, e in enumerate(self.pool) if self._hamm(S, e.S) < self.minham * len(S) and c <= e.cost]
        for i in reversed(kill_idx): del self.pool[i]
        self.pool.append(PoolEntry(S.copy(), c))
        self.pool.sort(key=lambda e: e.cost)
        self.pool = self.pool[:self.size]

    def farthest(self, S):
        if not self.pool: return None
        return self.pool[np.argmax([self._hamm(S, e.S) for e in self.pool])].S.copy()


def path_relink(core: DRDPCore, tgt: np.ndarray, max_steps: int = 15, scores: Optional[np.ndarray] = None):
    diffs = np.where(core.S != tgt)[0].tolist()
    if scores is not None:
        diffs.sort(key=lambda u: scores[u] if tgt[u] == 0 else -scores[u], reverse=True)
    else:
        random.shuffle(diffs)

    improved, drop = False, 0
    for k, u in enumerate(diffs):
        if k >= max_steps: break
        c0, snap = core.cost(), core.copy_snapshot()
        core._set_label(u, int(tgt[u]))
        if core.viol_count == 0 and core.cost() <= c0:
            core.prune_full(1)
            if core.cost() <= c0:
                improved = True;
                drop += (c0 - core.cost())
            else:
                core.restore_snapshot(snap)
        else:
            core.restore_snapshot(snap)
    return improved, drop


# =================== CP-SAT Subproblem (LNS) ===================

def k_hop_ball(neigh: List[List[int]], center: int, k: int, cap: int) -> List[int]:
    seen, q, order = {center}, [(center, 0)], [center]
    while q and len(order) < cap:
        u, d = q.pop(0)
        if d == k: continue
        for v in neigh[u]:
            if v not in seen:
                seen.add(v);
                order.append(v);
                q.append((v, d + 1))
                if len(order) >= cap: break
    return order


def build_local_cpsat(core: DRDPCore, R: List[int], hint: Optional[np.ndarray] = None):
    assert ORTOOLS_OK, "OR-Tools required."
    model = cp_model.CpModel()
    idx_of = {v: i for i, v in enumerate(R)}
    y = [model.NewBoolVar(f"y_{v}") for v in R]
    z = [model.NewBoolVar(f"z_{v}") for v in R]

    def lin_add(terms):
        return _bi.sum(terms[1:], terms[0]) if terms else 0

    def add_ge(terms, rhs: int):
        if rhs > 0: model.Add(lin_add(terms) >= int(rhs))

    for i in range(len(R)): model.Add(y[i] + z[i] <= 1)

    for i, v in enumerate(R):
        rhs, terms = 2, [y[i] * 2, z[i] * 2]
        for u in core.neigh[v]:
            if (j := idx_of.get(u)) is not None:
                terms.extend([y[j], z[j] * 2])
            else:
                rhs -= 1 if core.S[u] == 2 else (2 if core.S[u] == 3 else 0)
        add_ge(terms, rhs)

    Rset = set(R)
    frontier = {w for v in R for w in core.neigh[v] if w not in Rset}
    for w in frontier:
        if core.S[w] != 0: continue
        rhs, terms = 2, []
        for t in core.neigh[w]:
            if (j := idx_of.get(t)) is not None:
                terms.extend([y[j], z[j] * 2])
            else:
                rhs -= 1 if core.S[t] == 2 else (2 if core.S[t] == 3 else 0)
        add_ge(terms, rhs)

    model.Minimize(lin_add([yi * 2 for yi in y] + [zi * 3 for zi in z]))

    if hint is not None:
        for i in range(len(R)):
            model.AddHint(y[i], 1 if hint[i] == 2 else 0)
            model.AddHint(z[i], 1 if hint[i] == 3 else 0)

    return model, y, z


def solve_local_cpsat_region(core: DRDPCore, R: List[int], time_limit: float = 0.30, workers: int = 8):
    if not ORTOOLS_OK or len(R) < 5: return None, None, R

    hint = np.array([core.S[v] for v in R], dtype=np.int8)
    model, y, z = build_local_cpsat(core, R, hint=hint)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, float(time_limit))
    solver.parameters.num_search_workers = max(1, int(workers))

    # If running with a single worker on small subproblems, CP-SAT defaults can be suboptimal.
    # It might waste the tight time limit on heavy LP relaxations or running its own internal LNS,
    # rather than just using fast, exact branch-and-bound.
    if workers == 1:
        # Disable heavy LP relaxations since this is a pure Boolean combinatorial problem
        solver.parameters.linearization_level = 0
        # We are already inside an ALNS loop; disable CP-SAT's internal LNS to focus on exact search
        solver.parameters.use_lns = False
        # Fallback to pure exact search immediately rather than hoping a heuristic finds a bound
        solver.parameters.search_branching = cp_model.FIXED_SEARCH

    # Provide better logging/signal handling isolation
    solver.parameters.catch_sigint_signal = False

    res = solver.Solve(model)

    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None, None, R
    S_loc = np.zeros(len(R), dtype=np.int8)
    for i in range(len(R)):
        S_loc[i] = 3 if solver.Value(z[i]) == 1 else (2 if solver.Value(y[i]) == 1 else 0)
    return S_loc, int(S_loc.sum()), R


# =================== AWR Online Learning ===================

@dataclass
class Event:
    unlocked: List[int];
    reward: float;
    stagn: int;
    X: np.ndarray;
    g: np.ndarray


class Replay:
    def __init__(self, cap=50000): self.buf: Deque[Event] = deque(maxlen=cap)

    def add(self, e: Event): self.buf.append(e)

    def sample(self, B: int) -> List[Event]:
        if not self.buf: return []
        idx = np.random.choice(len(self.buf), size=min(B, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]


# =================== Main Solver Pipeline ===================

class NeuroCPLNS:
    def __init__(self, n: int, neigh: List[List[int]], device: str = None, seed: int = 0, lr: float = 3e-4,
                 beta: float = 2.0):
        self.core = DRDPCore(n, neigh, seed=seed)
        self.n, self.neigh, self.beta = n, neigh, beta
        self.use_torch = TORCH_OK

        if self.use_torch:
            self.device = torch.device(device) if device else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.fp16 = (self.device.type == "cuda")
            self.A = build_norm_adj(n, neigh, self.device)
            self.enc = GraphSAGELite(in_dim=10, hid=128, layers=3).to(self.device)
            self.heads = Heads(hid=128).to(self.device)
            self.opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.heads.parameters()), lr=lr)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.fp16) if hasattr(torch, 'amp') else amp.GradScaler(
                enabled=self.fp16)
        else:
            self.device = "cpu"
            self.enc = SAGE_Numpy(in_dim=10)
            self.heads = Heads_Numpy()

        self.replay = Replay(cap=50000)

    def _forward(self, stagn: int, detach: bool = False):
        Xn, gn = self.core.node_features(), self.core.global_features(stagn)
        if not self.use_torch: return None, None, self.heads.forward(self.enc.forward(Xn, None), gn)[0], 0.0

        X = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        g = torch.tensor(gn, dtype=torch.float32, device=self.device)
        ctx = torch.autocast(device_type=self.device.type, enabled=self.fp16) if hasattr(torch,
                                                                                         'autocast') else amp.autocast(
            enabled=self.fp16)

        if detach:
            with torch.no_grad(), ctx:
                H = self.enc(X, self.A)
                s, v = self.heads(H, g)
            return H, g, s, v
        with ctx:
            H = self.enc(X, self.A)
            s, v = self.heads(H, g)
        return H, g, s, v

    def _learn(self, batch=256, value_coef: float = 0.5):
        if not self.use_torch or not (evs := self.replay.sample(batch)): return

        losses = []
        ctx = torch.autocast(device_type=self.device.type, enabled=self.fp16) if hasattr(torch,
                                                                                         'autocast') else amp.autocast(
            enabled=self.fp16)
        with ctx:
            for e in evs:
                if not e.unlocked: continue
                X = torch.tensor(e.X, dtype=torch.float32, device=self.device)
                g = torch.tensor(e.g, dtype=torch.float32, device=self.device)
                s, v = self.heads(self.enc(X, self.A), g)

                pos = torch.tensor(list(set(e.unlocked)), dtype=torch.long, device=self.device)
                mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
                mask[pos] = False
                neg_pool = torch.arange(self.n, device=self.device)[mask]
                if len(neg_pool) == 0: continue
                neg = neg_pool[torch.randint(0, len(neg_pool), (min(len(pos), len(neg_pool)),), device=self.device)]

                logits = torch.cat([s[pos], s[neg]], dim=0)
                labels = torch.cat(
                    [torch.ones(len(pos), device=self.device), torch.zeros(len(neg), device=self.device)])

                adv = e.reward - v
                w = torch.exp(torch.clamp(adv.detach() / self.beta, min=-5.0, max=5.0))
                losses.append(w * (F.binary_cross_entropy_with_logits(logits, labels) + value_coef * F.mse_loss(v,
                                                                                                                torch.tensor(
                                                                                                                    e.reward,
                                                                                                                    device=self.device))))

        if not losses: return
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(torch.stack(losses).mean()).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.heads.parameters()), 2.0)
        self.scaler.step(self.opt)
        self.scaler.update()

    def solve(self, iters: int = 500, starts: int = 5, cp_time: float = 0.30, topk: int = 256, radius: int = 2,
              cap: int = 200, pr_every: int = 120, workers: int = 8):
        bestS, bestC = None, 10 ** 9
        for st in range(starts):
            self.core.S[:] = 0;
            self.core.n2[:] = 0;
            self.core.n3[:] = 0;
            self.core.viol[:] = 1
            self.core.viol_count, self.core.add_count[:] = self.n, 0
            self.core.greedy_init()
            stagn, pool = 0, ElitePool(size=8)

            if self.core.viol_count == 0: pool.try_add(self.core.S)

            current_cap = min(50, cap)

            for it in range(1, iters + 1):
                c0, out = self.core.cost(), self._forward(stagn, detach=True)
                scores = out[2].cpu().numpy() if self.use_torch else out[2]

                active = {u for u in range(self.n) if self.core.S[u] > 0}
                if self.core.viol_count > 0:
                    for vtx in np.flatnonzero(self.core.viol): active.update([vtx] + self.neigh[vtx])
                cand = np.array(sorted(active)) if active else np.arange(self.n)
                unlocked = cand[gumbel_top_k(scores[cand], min(topk, len(cand)))].tolist()

                X_taken, g_taken = self.core.node_features(), self.core.global_features(stagn)

                anchors = unlocked[:min(3, len(unlocked))]
                Rset = set()

                # Exploit vs Explore Operator Selection
                if random.random() < 0.75 and anchors:
                    per_cap = max(1, current_cap // len(anchors))
                    for a in anchors:
                        for node in k_hop_ball(self.neigh, a, radius, per_cap):
                            Rset.add(node)
                            if len(Rset) >= current_cap: break
                        if len(Rset) >= current_cap: break
                else:
                    curr_node = random.choice(unlocked) if unlocked else random.randint(0, self.n - 1)
                    for _ in range(current_cap):
                        Rset.add(curr_node)
                        neighbors = self.neigh[curr_node]
                        curr_node = random.choice(neighbors) if neighbors else random.randint(0, self.n - 1)

                if not Rset: Rset = set(unlocked[:current_cap])
                R = list(Rset)

                snap = self.core.copy_snapshot()
                S_loc, loc_cost, Rlist = solve_local_cpsat_region(self.core, R, time_limit=cp_time, workers=workers)

                if S_loc is not None:
                    for i, vtx in enumerate(Rlist): self.core._set_label(vtx, int(S_loc[i]))
                    if (pc := self.core.pair_candidates()): self.core.pass_pair(pc, limit=24)
                    self.core.prune_full(3)

                if self.core.viol_count != 0: self.core.restore_snapshot(snap)

                reward = float(c0 - self.core.cost())
                self.replay.add(Event(unlocked, reward, stagn, X_taken, g_taken))

                # Curriculum Growth / Early Stopping
                if reward > 0 and self.core.viol_count == 0:
                    stagn = 0
                    pool.try_add(self.core.S)
                    current_cap = max(50, int(current_cap * 0.8))  # Shrink if successful
                else:
                    stagn += 1
                    if stagn > 1: current_cap = min(cap, int(current_cap * 2.0))  # Rapid expand on stuck

                if pr_every > 0 and (it % pr_every) == 0 and pool.pool:
                    tgt = pool.farthest(self.core.S)  # BUG FIX APPLIED HERE
                    if tgt is not None:
                        snap2 = self.core.copy_snapshot()
                        if path_relink(self.core, tgt, max_steps=15, scores=scores)[0] and self.core.viol_count == 0:
                            pool.try_add(self.core.S);
                            stagn = 0
                        else:
                            self.core.restore_snapshot(snap2)

                if self.core.viol_count == 0 and self.core.cost() < bestC:
                    bestC, bestS = self.core.cost(), self.core.S.copy()

                if it % 16 == 0: self._learn(batch=256)

                # Fast early stopping restored
                if stagn >= 5: break

        return bestS, bestC


def solve_dir(data_dir: str, out_path: str, iters: int = 500, starts: int = 5, cp_time: float = 0.30, topk: int = 256,
              device: str = None, workers: int = 1):
    csv_file, writer = None, None
    csv_path = out_path if out_path.endswith('.csv') else out_path + ".csv"
    try:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(['Graph', 'Method', 'Cost', 'Time', 'Solution'])
    except Exception as e:
        print(f"[WARN] CSV Logging failed: {e}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        for fp in sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mtx.gz")]):
            base = os.path.basename(fp)
            gc.collect()
            if TORCH_OK: torch.cuda.empty_cache()

            try:
                n, neigh = read_mtx_gz(fp)
                solver = NeuroCPLNS(n, neigh, device=device)
                t1 = time.time()
                S, c = solver.solve(iters=iters, starts=starts, cp_time=cp_time, topk=topk, workers=workers)
                secs = time.time() - t1
                sol_text = str(S.tolist())

                block = f"Graph: {base}\nSolution: {sol_text}\nCost: {int(c)}\nTime(s): {secs:.6f}\n"
                print(block, end="")
                f.write(block + "\n")
                if writer:
                    writer.writerow([base, "NeuroCP-ALNS", int(c), f"{secs:.4f}", sol_text])
                    csv_file.flush()
            except Exception as e:
                print(f"[ERROR] {base}: {e}", file=sys.stderr)
                if writer: writer.writerow([base, "NeuroCP-ALNS", -1, 0.0, "[]"])

    if csv_file: csv_file.close()


def main():
    ap = argparse.ArgumentParser(description="DRDP-NeuroCP-ALNS Solver (AAAI 2024/2025)")
    sub = ap.add_subparsers(dest="cmd")
    ap_s = sub.add_parser("solve")
    ap_s.add_argument("--data_dir", required=True)
    ap_s.add_argument("--out", required=True)
    ap_s.add_argument("--iters", type=int, default=500)
    ap_s.add_argument("--starts", type=int, default=5)
    ap_s.add_argument("--cp_time", type=float, default=0.30)
    ap_s.add_argument("--topk", type=int, default=256)
    ap_s.add_argument("--device", type=str, default=None)
    ap_s.add_argument("--workers", type=int, default=1, help="OR-Tools CP-SAT threads")

    args = ap.parse_args()
    if args.cmd == "solve":
        solve_dir(args.data_dir, args.out, iters=args.iters, starts=args.starts, cp_time=args.cp_time, topk=args.topk,
                  device=args.device, workers=args.workers)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

