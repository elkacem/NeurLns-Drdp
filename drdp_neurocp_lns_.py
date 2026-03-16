# drdp_neurocp_lns.py
# DRDP-NeuroCP-LNS: GPU-guided neighborhoods + exact local solve (CP-SAT)
#  - Always-feasible O(deg) DRDP core (labels in {0,2,3})
#  - GraphSAGE-lite (FP16) to score "unlock" sets; Gumbel-Top-k for diversity
#  - Local DRDP-1' ILP (scaled to integers) solved by OR-Tools CP-SAT with hints
#  - AWR-style online learning of unlock scores
#  - Elite pool + short path-relinking
# ---------------------------------------------------------------------

import os, sys, gzip, time, math, random, argparse
import gc # explicit gc
import csv # explicit import
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Deque
from collections import deque
# from torch import amp  <-- removed
import traceback
import numpy as np

# ---- Torch (GPU) ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.cuda.amp as amp

    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---- OR-Tools CP-SAT ----
try:
    from ortools.sat.python import cp_model

    ORTOOLS_OK = True
except Exception:
    ORTOOLS_OK = False


# ========================= IO =========================

def read_mtx_gz(path: str) -> Tuple[int, List[List[int]]]:
    with gzip.open(path, 'rt') as f:
        line = f.readline()
        while line and (line.strip() == '' or line[0] in 'c%'):
            line = f.readline()
        if not line: raise ValueError("Invalid MTX header.")
        r, c, _ = map(int, line.strip().split()[:3])
        n = max(r, c)
        adj = [set() for _ in range(n)]
        for l in f:
            if not l.strip() or l[0] in 'c%': continue
            u, v = map(lambda x: int(x) - 1, l.split()[:2])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].add(v);
                adj[v].add(u)
    return n, [list(s) for s in adj]


# ================= DRDP core (always feasible, O(deg)) =================

class DRDPCore:
    """
    State:
      S[v] ∈ {0,2,3}
      n2[v], n3[v], viol[v] maintained incrementally
    """

    def __init__(self, n: int, neigh: List[List[int]], seed: int = 0):
        self.n = n;
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
        # Precompute static structural features: Local Clustering Coefficient (approx or exact)
        # For efficiency on large graphs, we can use simple triangle counts if valid,
        # or just skip if too slow. Here we assume medium graphs and do a quick pass if feasible.
        self.clustering = np.zeros(n, dtype=np.float32)
        # (Optional: compute clustering here if needed, keeping it 0 for now to stay fast)
        # Using 0 as placeholder or implementing a fast triangle count loop
        # For simplicity in this single-file solver, we'll leave it as an available slot.

    def cost(self) -> int:
        return int(self.S.sum())

    def compute_clustering_coeff(self):
        """
        Compute local clustering coefficient for each node.
        LCC(v) = 2 * E(neigh(v)) / (k(v) * (k(v)-1))
        This is O(sum(deg(v)^2)), which can be slow for dense graphs.
        We limit it to graphs with reasonable size/density or sample.
        """
        # If too large, just return logic is already 0.0
        if self.n > 20000 and len(self.neigh) > 10 * self.n:
             return

        c = np.zeros(self.n, dtype=np.float32)
        for u in range(self.n):
            k = len(self.neigh[u])
            if k < 2:
                continue
            # count triangles: interactions between neighbors
            # optimized: iterate neighbors, check checks
            # For speed in Python, we intersect sets.
            # Faster approach: neighbor sets are available.

            # This can still be slow. Let's do a fast approximate or limit check?
            # intersection of sorted lists is fast? sets are O(1) avg check.

            nv = self.neigh[u] # list or set? It is list in __init__ but passed as list of sets in read_mtx?
            # read_mtx_gz returns list of lists.
            # let's convert to set for O(1) lookup just for this node context if needed,
            # or rely on the input structure.
            # Actually, `read_mtx_gz` does: adj = [set() ...]; return ... [list(s) for s in adj].
            # So self.neigh is list of lists.

            # Efficient triangle count for u:
            neighbors = set(nv)
            tri = 0
            for v in nv:
                for w in self.neigh[v]:
                    if w in neighbors:
                        tri += 1
            # tri counts each edge twice (v-w and w-v).
            tri //= 2
            c[u] = (2.0 * tri) / (k * (k - 1))

        self.clustering = c

    def copy_snapshot(self):
        return (
        self.S.copy(), self.n2.copy(), self.n3.copy(), self.viol.copy(), int(self.viol_count), self.add_count.copy())

    def restore_snapshot(self, snap):
        self.S[:], self.n2[:], self.n3[:], self.viol[:], self.viol_count, self.add_count[:] = snap

    def _set_label(self, u: int, new: int):
        old = int(self.S[u])
        if old == new: return
        if old == 3:
            for v in self.neigh[u]: self.n3[v] -= 1
        elif old == 2:
            for v in self.neigh[u]: self.n2[v] -= 1
        self.S[u] = new
        if new in (2, 3): self.add_count[u] = min(32767, self.add_count[u] + 1)
        if new == 3:
            for v in self.neigh[u]: self.n3[v] += 1
        elif new == 2:
            for v in self.neigh[u]: self.n2[v] += 1
        # update neighbors' violation flags (only matters when neighbor is 0)
        for v in self.neigh[u]:
            if self.S[v] == 0:
                sat = (self.n3[v] >= 1) or (self.n2[v] >= 2)
                if sat and self.viol[v]:
                    self.viol[v] = 0; self.viol_count -= 1
                elif (not sat) and (not self.viol[v]):
                    self.viol[v] = 1; self.viol_count += 1
        # update u's own violation flag when we set it to 0 or from 0 to {2,3}
        if new == 0:
            sat = (self.n3[u] >= 1) or (self.n2[u] >= 2)
            if sat and self.viol[u]:
                self.viol[u] = 0; self.viol_count -= 1
            elif (not sat) and (not self.viol[u]):
                self.viol[u] = 1; self.viol_count += 1
        elif old == 0 and new in (2, 3):
            if self.viol[u]: self.viol[u] = 0; self.viol_count -= 1

    def greedy_init(self):
        while self.viol_count > 0:
            viol_idx = np.flatnonzero(self.viol)
            score = np.zeros(self.n, dtype=np.int32)
            for v in viol_idx:
                for u in self.neigh[v]:
                    if self.S[u] != 3: score[u] += 1
            u = int(np.argmax(score))
            if score[u] == 0: u = int(viol_idx[self.rng.integers(len(viol_idx))])
            self._set_label(u, 3)
        self.prune_full(4)
        # sanity: after greedy init + prune we must be feasible
        assert self.viol_count == 0, "greedy_init/prune produced infeasible state."

    def prune_full(self, max_passes: int = 3) -> bool:
        changed_any = False
        for _ in range(max_passes):
            changed = False
            th = [u for u in range(self.n) if self.S[u] == 3];
            random.shuffle(th)
            for u in th:
                if self._safe_demote3_to2(u): self._set_label(u, 2); changed = True; changed_any = True
            tw = [u for u in range(self.n) if self.S[u] == 2];
            random.shuffle(tw)
            for u in tw:
                if self._safe_demote2_to0(u): self._set_label(u, 0); changed = True; changed_any = True
            if not changed: break
        return changed_any

    # ---- safe demotions (fixed 2->0 bug) ----
    def _safe_demote3_to2(self, u: int) -> bool:
        # neighbors that are zero must keep coverage after losing 1
        for v in self.neigh[u]:
            if self.S[v] == 0 and (2 * self.n3[v] + self.n2[v] < 3):
                return False
        return True

    def _safe_demote2_to0(self, u: int) -> bool:
        # (1) u itself will become 0; it must already be covered by neighbors
        if not (self.n3[u] >= 1 or self.n2[u] >= 2):
            return False
        # (2) neighbors that are zero must keep coverage after losing 1
        for v in self.neigh[u]:
            if self.S[v] == 0 and (2 * self.n3[v] + self.n2[v] < 3):
                return False
        return True

    def compute_private_support(self):
        priv3 = np.zeros(self.n, dtype=np.float32);
        priv2 = np.zeros(self.n, dtype=np.float32)
        for v in range(self.n):
            if self.S[v] != 0: continue
            c3, c2 = self.n3[v], self.n2[v]
            if c3 == 1 and c2 < 2:
                for u in self.neigh[v]:
                    if self.S[u] == 3: priv3[u] += 1; break
            if c3 == 0 and c2 == 2:
                tw = []
                for u in self.neigh[v]:
                    if self.S[u] == 2:
                        tw.append(u)
                        if len(tw) == 2: break
                if len(tw) == 2:
                    priv2[tw[0]] += 1;
                    priv2[tw[1]] += 1
        d = float(max(1, self.maxdeg))
        return priv3 / d, priv2 / d

    def node_features(self) -> np.ndarray:
        priv3, priv2 = self.compute_private_support()
        X = np.zeros((self.n, 11), dtype=np.float32) # Increased dim to 11
        X[:, 0] = (self.S == 0).astype(np.float32)
        X[:, 1] = (self.S == 2).astype(np.float32)
        X[:, 2] = (self.S == 3).astype(np.float32)
        X[:, 3] = self.deg / float(self.maxdeg)
        X[:, 4] = self.n2 / float(self.maxdeg)
        X[:, 5] = self.n3 / float(self.maxdeg)
        X[:, 6] = self.viol.astype(np.float32)
        X[:, 7] = priv2;
        X[:, 8] = priv3
        X[:, 9] = np.minimum(1.0, self.add_count / 20.0)
        X[:, 10] = self.clustering # New feature: Structural awareness
        return X

    def global_features(self, stagn: int) -> np.ndarray:
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
        out = []
        for v in range(self.n):
            if self.S[v] == 0 and self.n3[v] == 0 and self.n2[v] == 2:
                tw = []
                for u in self.neigh[v]:
                    if self.S[u] == 2:
                        tw.append(u)
                        if len(tw) == 2: break
                if len(tw) == 2: out.append((v, tw[0], tw[1]))
        return out

    def pass_pair(self, triples: Iterable[Tuple[int, int, int]], limit: int = 32):
        ch = 0;
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


# =================== GraphSAGE-lite (GPU) ===================

def build_norm_adj(n: int, neigh: List[List[int]], device):
    if not TORCH_OK:
        return None  # Fallback: cpu mode
    row = [];
    col = []
    deg = np.ones(n, dtype=np.float32)  # start at 1.0 to account for self-loop
    for u in range(n):
        for v in neigh[u]:
            row.append(u);
            col.append(v);
            deg[u] += 1.0
    for u in range(n):
        row.append(u);
        col.append(u)
    row = torch.tensor(row, dtype=torch.long, device=device)
    col = torch.tensor(col, dtype=torch.long, device=device)
    d = torch.tensor(deg, dtype=torch.float32, device=device)
    val = 1.0 / torch.sqrt(d[row] * d[col])
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, size=(n, n)).coalesce()


def spmm_fp32(A, H):
    # sparse.mm needs fp32; cast back afterward
    if not TORCH_OK: return H # dummy
    with torch.autocast(device_type=H.device.type, enabled=False):
        Z32 = torch.sparse.mm(A.float(), H.float())
    return Z32.to(H.dtype)

if TORCH_OK:
    # GAT Implementation (SOTA for NCO) replacing simple SAGE
    class GATLayer(nn.Module):
        def __init__(self, in_dim, out_dim, n_heads=4, concat=True):
            super().__init__()
            self.n_heads = n_heads
            self.concat = concat
            self.d_k = out_dim // n_heads if concat else out_dim

            self.W = nn.Linear(in_dim, n_heads * self.d_k, bias=False)
            self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.d_k))
            self.leak = nn.LeakyReLU(0.2)
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.xavier_uniform_(self.a)

        def forward(self, h, adj_indices):
            # h: (N, in_dim)
            N = h.size(0)
            Wh = self.W(h).view(N, self.n_heads, self.d_k) # (N, heads, d_k)

            # For sparse attention, we calculate pairwise scores only for edges
            # adj_indices: (2, E)
            row, col = adj_indices

            # Wh[row]: (E, heads, d_k), Wh[col]: (E, heads, d_k)
            # Additive Attention: a_l * Wh_i + a_r * Wh_j
            # self.a is (heads, 2*d_k). Split into a_l: (heads, d_k), a_r: (heads, d_k)
            a_l = self.a[:, :self.d_k] # (heads, d_k)
            a_r = self.a[:, self.d_k:] # (heads, d_k)

            # Precompute node scores: (N, heads)
            # Wh: (N, heads, d_k)  * a_l: (heads, d_k) -> (N, heads)?
            # einsum 'nhd,hd->nh'
            score_l = (Wh * a_l.unsqueeze(0)).sum(dim=-1)
            score_r = (Wh * a_r.unsqueeze(0)).sum(dim=-1)

            # Edge scores: score_l[row] + score_r[col] => (E, heads)
            e = self.leak(score_l[row] + score_r[col])

            # Softmax over neighbors (for each dst 'col', sum over 'row')
            # Use geometric / scatter softmax (torch_scatter logic simulation)
            # We can use pure torch sparse softmax trick if needed, or simple exp shift

            # Stable softmax: e - max(e)
            # Since global max is okayish for numeric stability in attention:
            # But we need local max per destination.
            # Lazy approach: simple exp
            e_exp = torch.exp(e - e.max()) # numeric stability global

            # Denom: sum e_exp per destination (col)
            # row is src, col is dst
            denom = torch.zeros(N, self.n_heads, device=h.device)
            denom = denom.index_add(0, col, e_exp) + 1e-6

            # alpha: (E, heads)
            alpha = e_exp / denom[col]

            # Msg: alpha * Wh_row
            # Here Wh_row is (E, heads, d_k). Is it possible to avoid materializing it?
            # Ideally: we want to compute sum_{j in N(i)} alpha_{ji} W h_j
            # alpha is (E, heads). Wh[row] is (E, heads, d_k).
            # This multiplication materializes (E, heads, d_k) which is big.
            # But we can't easily avoid it without custom kernel or torch_scatter/PyG.
            # For now, we optimized the 'e' computation which was (E, heads, 2*d_k) -> (E, heads).
            # The msg tensor is (E, heads, d_k). Max memory usage is dominated by this.
            # If d_k=32, E=10M => 10M * 4 * 32 * 4 bytes = 5GB.
            # 'blckhole' might be large.
            # Let's hope reducing 'e' calc is enough. If not, we need checkpointing or CPU fallback.

            msg = alpha.unsqueeze(-1) * Wh[row] # (E, heads, d_k)

            # Agg: sum msg per destination
            out = torch.zeros(N, self.n_heads, self.d_k, device=h.device)
            out = out.index_add(0, col, msg)

            if self.concat:
                return out.view(N, self.n_heads * self.d_k)
            else:
                return out.mean(dim=1)

    class GAT(nn.Module):
        def __init__(self, in_dim, hid=128, layers=3):
            super().__init__()
            # Two GAT layers + one final
            self.conv1 = GATLayer(in_dim, hid, n_heads=4, concat=True)
            self.conv2 = GATLayer(hid, hid, n_heads=4, concat=True)
            self.conv3 = GATLayer(hid, hid, n_heads=4, concat=False) # Average last layer
            self.lns = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

        def forward(self, X, A_sparse):
            # A_sparse is coalesced (N, N)
            # We need indices (2, E)
            indices = A_sparse.indices()

            H = X
            H = F.elu(self.conv1(H, indices))
            H = self.lns[0](H)

            H = F.elu(self.conv2(H, indices))
            H = self.lns[1](H)

            H = self.conv3(H, indices) # No activation or ELU? usually linear or ELU
            H = self.lns[2](H)
            return H

    # Replaced SAGE with GAT for 'enc'
    # Keeping Heads same
    class SAGE(nn.Module): # Keeping name SAGE to avoid changing all init code, but implemented as GAT
         def __init__(self, in_dim, hid=128, layers=3):
             super().__init__()
             self.gat = GAT(in_dim, hid, layers)
         def forward(self, X, A):
             return self.gat(X, A)

    class Heads(nn.Module):
        def __init__(self, hid=128):
            super().__init__()
            self.unlock = nn.Linear(hid, 1)  # unlock score per node
            self.value = nn.Sequential(nn.Linear(2 * hid + 9, 128), nn.ReLU(), nn.Linear(128, 1))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

        def forward(self, H, g):  # g: 9 global feats
            s = self.unlock(H).squeeze(-1)
            hmean = H.mean(0);
            hmax = H.max(0).values
            v = self.value(torch.cat([hmean, hmax, g], dim=-1)).squeeze(-1)
            return s, v

# ----- NEW: Numpy fallback for GNN -----
class SAGE_Numpy:
    """Mock-up of SAGE when torch is missing."""
    def __init__(self, in_dim, hid=128, layers=3):
        self.dims = [(in_dim, hid)] + [(hid, hid)]*(layers-1)
        # Random weights
        self.Ws = [np.random.randn(din, dout) * 0.1 for din, dout in self.dims]

    def forward(self, X, neigh_adj):
        # We need neighbor aggregation.
        # For simplicity in fallback, perform mean agg over neighbors manual loops or sparse mat
        H = X
        n = X.shape[0]
        # Very slow: manual loop.
        # Better: assume we iterate.
        # For large graphs, numpy fallback is just to let it run (random policy effectively).
        # We will implement a 'random' projection to simulate a distinct policy.
        for W_mat in self.Ws:
             # Just a linear projection to mix features - skipping precise graph agg
             # because numpy sparse is not setup here.
             # This means "No-GNN-Aggregation" baseline.
             H = H @ W_mat
             H = np.maximum(0, H) # ReLU
        return H

class Heads_Numpy:
    def __init__(self, hid=128):
        self.unlock = np.random.randn(hid, 1) * 0.1

    def forward(self, H, g):
        s = (H @ self.unlock).squeeze(-1)
        # return logits and dummy value
        return s, 0.0


# =================== Utilities ===================

def gumbel_top_k(scores: np.ndarray, K: int) -> List[int]:
    if K <= 0 or len(scores) == 0: return []
    K = min(K, len(scores))
    g = -np.log(-np.log(np.random.rand(*scores.shape) + 1e-9) + 1e-9)
    y = scores + g
    # take K largest with tie-breaking by gumbel noise
    idx = np.argpartition(-y, K - 1)[:K]
    idx = idx[np.argsort(-y[idx])]
    return idx.tolist()


from dataclasses import dataclass


@dataclass(eq=False)
class PoolEntry:
    S: np.ndarray
    cost: int


class ElitePool:
    def __init__(self, size=8, min_hamming_frac=0.05):
        self.size = size;
        self.minham = min_hamming_frac;
        self.pool: List[PoolEntry] = []

    def _hamm(self, A, B):
        return int(np.count_nonzero(A != B))

    def try_add(self, S):
        c = int(S.sum())
        kill_idx = []
        for i, e in enumerate(self.pool):
            if self._hamm(S, e.S) < self.minham * len(S) and c <= e.cost:
                kill_idx.append(i)
        for i in reversed(kill_idx):
            del self.pool[i]
        self.pool.append(PoolEntry(S.copy(), c))
        self.pool.sort(key=lambda e: e.cost)
        if len(self.pool) > self.size:
            self.pool = self.pool[:self.size]

    def farthest(self, S):
        if not self.pool: return None
        i = np.argmax([self._hamm(S, e.S) for e in self.pool])
        return self.pool[i].S.copy()

    def best(self):
        return self.pool[0].S.copy() if self.pool else None


def path_relink(core: DRDPCore, tgt: np.ndarray, max_steps: int = 15, scores: Optional[np.ndarray] = None):
    diffs = np.where(core.S != tgt)[0].tolist();

    # Improved: Sort differences by neural confidence
    if scores is not None:
        # If tgt=0, we want high score (high prob to unlock).
        # If tgt!=0, we want low score (low prob to unlock).
        # We can sort by:  -score[u] if tgt[u]==0 else score[u]
        # Equivalent to sorting by: (1 if tgt[u]==0 else -1) * score[u] DESC
        def key_fn(u):
            raw = scores[u]
            # if we want u -> 0, high raw is good.
            # if we want u -> 2/3, low raw is good (high negative raw is good).
            return raw if tgt[u] == 0 else -raw
        diffs.sort(key=key_fn, reverse=True)
    else:
        random.shuffle(diffs)

    improved = False;
    drop = 0
    for k, u in enumerate(diffs):
        if k >= max_steps: break
        want = int(tgt[u]);
        c0 = core.cost();
        snap = core.copy_snapshot()
        core._set_label(u, want)
        if core.viol_count == 0 and core.cost() <= c0:
            core.prune_full(1)
            if core.cost() <= c0:
                improved = True; drop += (c0 - core.cost())
            else:
                core.restore_snapshot(snap)
        else:
            core.restore_snapshot(snap)
    return improved, drop


# =================== CP-SAT exact local subproblem ===================

def k_hop_ball(neigh: List[List[int]], center: int, k: int, cap: int) -> List[int]:
    seen = {center};
    q = [(center, 0)];
    order = [center]
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


import builtins as _bi  # used by lin_add


def build_local_cpsat(core: DRDPCore, R: List[int], hint: Optional[np.ndarray] = None, protect_frontier: bool = True):
    """
    DRDP-1' on region R with boundary fixings.
    Also (if protect_frontier): ensure zeros *outside* R that touch R keep coverage.
    """
    assert ORTOOLS_OK, "OR-Tools CP-SAT not available."
    model = cp_model.CpModel()
    idx_of = {v: i for i, v in enumerate(R)}
    y = [model.NewBoolVar(f"y_{v}") for v in R]
    z = [model.NewBoolVar(f"z_{v}") for v in R]

    # -------- helpers --------
    def lin_add(terms):
        if not terms:
            return 0  # constant 0
        return _bi.sum(terms[1:], terms[0])

    def add_ge(terms, rhs: int):
        if rhs <= 0:
            return  # already satisfied
        lhs = lin_add(terms)
        model.Add(lhs >= int(rhs))

    # -------------------------

    # at-most-one per node in R
    for i in range(len(R)):
        model.Add(y[i] + z[i] <= 1)

    # coverage constraints for v in R (with boundary constants)
    for i, v in enumerate(R):
        rhs = 2
        terms = [y[i] * 2, z[i] * 2]
        for u in core.neigh[v]:
            j = idx_of.get(u)
            if j is not None:
                terms.append(y[j])
                terms.append(z[j] * 2)
            else:
                if core.S[u] == 2:
                    rhs -= 1
                elif core.S[u] == 3:
                    rhs -= 2
        add_ge(terms, rhs)

    # NEW: protect frontier zeros outside R that touch R
    if protect_frontier:
        Rset = set(R)
        frontier = set()
        for v in R:
            for w in core.neigh[v]:
                if w not in Rset:
                    frontier.add(w)
        for w in frontier:
            if core.S[w] != 0:
                continue  # only zeros outside need protection
            rhs = 2
            terms = []
            for t in core.neigh[w]:
                j = idx_of.get(t)
                if j is not None:
                    terms.append(y[j])
                    terms.append(z[j] * 2)
                else:
                    # fixed neighbor outside R
                    if core.S[t] == 2:
                        rhs -= 1
                    elif core.S[t] == 3:
                        rhs -= 2
            add_ge(terms, rhs)

    # objective: 2*sum y + 3*sum z
    obj_terms = [yi * 2 for yi in y] + [zi * 3 for zi in z]
    model.Minimize(lin_add(obj_terms))

    # optional hints
    if hint is not None:
        for i in range(len(R)):
            model.AddHint(y[i], 1 if hint[i] == 2 else 0)
            model.AddHint(z[i], 1 if hint[i] == 3 else 0)

    return model, y, z

# NEW: Guided ball (weighted BFS) matching the heatmap
def guided_ball(neigh: List[List[int]], center: int, k: int, cap: int, scores: np.ndarray) -> List[int]:
    """
    Explore neighborhood using Policy Score as heuristic.
    Standard BFS expands in circles. This expands towards 'interesting' nodes.
    Cost of edge (u,v) = 1.0 - score[v] (so we visit high-score nodes faster/cheaper).
    """
    import heapq
    # (cost, dist, u)
    pq = [(0.0, 0, center)]
    seen = {center: 0.0}
    order = []

    while pq and len(order) < cap:
        c, d, u = heapq.heappop(pq)

        # If we found a cheaper path later, skip (Dijkstra standard)
        if c > seen[u]: continue

        order.append(u)

        if d >= k: continue

        for v in neigh[u]:
            # Edge weight strategy:
            # We want to favor nodes with high probability of being unlocked (score ~ 1.0).
            # So cost to traverse = 1.0 - sigmoid(score[v])?
            # 'scores' passed here are raw logits most likely.
            # Let's assume scores are comparable.
            w = 1.0 - (1.0 / (1.0 + math.exp(-scores[v]))) # 1 - prob
            new_c = c + 0.1 + w # base cost + penalty for low score

            if v not in seen or new_c < seen[v]:
                seen[v] = new_c
                heapq.heappush(pq, (new_c, d + 1, v))

    return order

def _region_frontier_quick_check(core: DRDPCore, R: List[int]) -> bool:
    """
    Quick feasibility pre-check: for any outside zero w that needs positive
    contribution from inside R (need>0), ensure it has at least one neighbor in R.
    """
    Rset = set(R)
    # Gather frontier zeros
    zeros_need_inside = []
    for v in R:
        for w in core.neigh[v]:
            if w in Rset or core.S[w] != 0:
                continue
            need = 2
            for t in core.neigh[w]:
                if t not in Rset:
                    if core.S[t] == 2:
                        need -= 1
                    elif core.S[t] == 3:
                        need -= 2
            if need > 0:
                zeros_need_inside.append(w)
    if not zeros_need_inside:
        return True
    # Check any of them actually connect to R (should by construction),
    # and that R contains at least one neighbor for each such w.
    for w in zeros_need_inside:
        if not any((u in Rset) for u in core.neigh[w]):
            return False
    return True


def solve_local_greedy_region(core: DRDPCore, R: List[int]):
    for v in R: core._set_label(v, 0)
    core.greedy_init()
    S_loc = np.array([core.S[v] for v in R], dtype=np.int8)
    return S_loc, int(S_loc.sum()), R

def solve_local_cpsat_region(core: DRDPCore, R: List[int], time_limit: float = 0.30, workers: int = 8):
    if not ORTOOLS_OK: return solve_local_greedy_region(core, R)
    # Small regions are not worth it
    if len(R) < 5:
        return None, None, R
    # Quick frontier sanity
    if not _region_frontier_quick_check(core, R):
        return None, None, R
    # build hint from current S on R
    hint = np.array([core.S[v] for v in R], dtype=np.int8)
    model, y, z = build_local_cpsat(core, R, hint=hint, protect_frontier=True)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, float(time_limit))
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = 1
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, R
    S_loc = np.zeros(len(R), dtype=np.int8)
    for i in range(len(R)):
        yi = int(solver.Value(y[i]))
        zi = int(solver.Value(z[i]))
        S_loc[i] = 3 if zi == 1 else (2 if yi == 1 else 0)
    local_cost = int(S_loc.sum())
    return S_loc, local_cost, R


# =================== Replay for AWR ===================

@dataclass
class Event:
    unlocked: List[int]
    reward: float
    stagn: int
    X: np.ndarray  # node features at action time
    g: np.ndarray  # global features at action time


class Replay:
    def __init__(self, cap=50000):
        self.cap = cap;
        self.buf: Deque[Event] = deque(maxlen=cap)

    def add(self, e: Event): self.buf.append(e)

    def sample(self, B: int) -> List[Event]:
        if not self.buf: return []
        idx = np.random.choice(len(self.buf), size=min(B, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]


# =================== NeuroCP-LNS main solver ===================

class NeuroCPLNS:
    def __init__(self, n: int, neigh: List[List[int]], device: str = None, seed: int = 0, lr: float = 3e-4,
                 beta: float = 2.0):
        # assert TORCH_OK, "PyTorch required."    <-- Handled below
        # assert ORTOOLS_OK, "OR-Tools (CP-SAT) required." <-- Handled but can't mock CP-SAT easily.

        self.core = DRDPCore(n, neigh, seed=seed)
        self.n = n;
        self.neigh = neigh
        self.use_torch = TORCH_OK

        if self.use_torch:
            self.device = torch.device(device) if device else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.fp16 = (self.device.type == "cuda")
            self.A = build_norm_adj(n, neigh, self.device)
            self.enc = SAGE(in_dim=11, hid=128, layers=3).to(self.device) # dim=11
            self.heads = Heads(hid=128).to(self.device)
            self.opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.heads.parameters()), lr=lr)
            # Use torch.amp if available (PyTorch 2.x), else fallback to old cuda.amp
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda', enabled=self.fp16)
            else:
                self.scaler = amp.GradScaler(enabled=self.fp16)
        else:
            # Fallback
            self.device = "cpu"
            self.enc = SAGE_Numpy(in_dim=11, hid=128, layers=3)
            self.heads = Heads_Numpy(hid=128)

        self.replay = Replay(cap=50000)
        self.beta = beta

        # Must compute clustering for new features
        self.core.compute_clustering_coeff()

    def _forward(self, stagn: int, detach: bool = False):
        Xn = self.core.node_features()
        gn = self.core.global_features(stagn)

        if not self.use_torch:
             # Numpy path
             H = self.enc.forward(Xn, None)
             s, v = self.heads.forward(H, gn)
             # To mimic torch output structure:
             return None, None, s, v

        X = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        g = torch.tensor(gn, dtype=torch.float32, device=self.device)
        devtype = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Use torch.autocast (unified) if available, else amp.autocast
        if hasattr(torch, 'autocast'):
            # device_type arg is required for torch.autocast
            ctx = torch.autocast(device_type=devtype, enabled=self.fp16)
        else:
            # old amp.autocast only supported cuda implicitly
            ctx = amp.autocast(enabled=self.fp16)

        # Optimize memory during inference steps (solve loop)
        # by disabling gradient tracking entirely
        if detach:
            with torch.no_grad():
                with ctx:
                    H = self.enc(X, self.A)
                    s, v = self.heads(H, g)
                return H, g, s, v  # already detached by no_grad

        # During training steps (learning loop), we need gradients
        else:
            with ctx:
                H = self.enc(X, self.A)
                s, v = self.heads(H, g)
            return H, g, s, v

    def _learn(self, batch=256, value_coef: float = 0.5):
        if not self.use_torch:
            return # Cannot train without Torch

        evs = self.replay.sample(batch)
        if not evs:
            return

        losses = []
        devtype = 'cuda' if self.device.type == 'cuda' else 'cpu'

        if hasattr(torch, 'autocast'):
            ctx = torch.autocast(device_type=devtype, enabled=self.fp16)
        else:
            ctx = amp.autocast(enabled=self.fp16)

        with ctx:
            for e in evs:
                if not e.unlocked:
                    continue

                X = torch.tensor(e.X, dtype=torch.float32, device=self.device)
                g = torch.tensor(e.g, dtype=torch.float32, device=self.device)

                # forward on the frozen adjacency (graph structure fixed)
                H = self.enc(X, self.A)
                s, v = self.heads(H, g)              # s: (n,), v: scalar

                pos = torch.tensor(list(set(e.unlocked)), dtype=torch.long, device=self.device)
                all_idx = torch.arange(self.n, device=self.device)
                mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
                mask[pos] = False
                neg_pool = all_idx[mask]
                if len(neg_pool) == 0:
                    continue
                k = min(len(pos), len(neg_pool))
                neg = neg_pool[torch.randint(0, len(neg_pool), (k,), device=self.device)]

                logits = torch.cat([s[pos], s[neg]], dim=0)
                labels = torch.cat(
                    [torch.ones(len(pos), device=self.device),
                     torch.zeros(len(neg), device=self.device)]
                )

                # per-event advantage (reward - value); stop grad on advantage weight only
                advantage = e.reward - v
                w = torch.exp(torch.clamp(advantage.detach() / self.beta, min=-5.0, max=5.0))

                policy_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
                value_loss = F.mse_loss(v, torch.tensor(e.reward, device=self.device))

                losses.append(w * (policy_loss + value_coef * value_loss))

        if not losses:
            return
        loss = torch.stack(losses).mean()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.heads.parameters()), 2.0)
        self.scaler.step(self.opt)
        self.scaler.update()


    def solve(self, iters: int = 500, starts: int = 5, cp_time: float = 0.30, topk: int = 256, radius: int = 2,
              cap: int = 1200, pr_every: int = 120, workers: int = 8):
        bestS = None;
        bestC = 10 ** 9

        for st in range(starts):
            # fresh start
            self.core.S[:] = 0;
            self.core.n2[:] = 0;
            self.core.n3[:] = 0
            self.core.viol[:] = 1;
            self.core.viol_count = self.core.n;
            self.core.add_count[:] = 0
            self.core.greedy_init();
            stagn = 0
            # record a feasible baseline
            if self.core.viol_count == 0 and self.core.cost() < bestC:
                bestC = self.core.cost();
                bestS = self.core.S.copy()
            pool = ElitePool(size=8, min_hamming_frac=0.05)
            if self.core.viol_count == 0:
                pool.try_add(self.core.S)

            for it in range(1, iters + 1):
                # ---- GPU choose unlock set
                c0 = self.core.cost()

                out = self._forward(stagn, detach=True)
                if self.use_torch:
                        s, v = out[2], out[3]
                        scores = s.cpu().numpy()
                else:
                        scores, _ = out[2], out[3]

                # Focus on likely active nodes: 3/2 nodes and neighbors of violations
                active = set([u for u in range(self.n) if self.core.S[u] > 0])
                if self.core.viol_count > 0:
                    for vtx in np.flatnonzero(self.core.viol):
                        active.add(vtx)
                        for u in self.neigh[vtx]: active.add(u)
                cand = np.array(sorted(active)) if active else np.arange(self.n)
                k = min(topk, len(cand))
                sel_idx = gumbel_top_k(scores[cand], k)
                unlocked = cand[sel_idx].tolist()

                # right after deciding `unlocked`
                X_taken = self.core.node_features()
                g_taken = self.core.global_features(stagn)

                # ---- Build R as union of up to 3 k-hop balls from anchors
                anchors = unlocked[:min(3, len(unlocked))]
                Rset = set()
                if anchors:
                    per_cap = max(1, cap // len(anchors))
                    for a in anchors:
                        # Reverted to k_hop_ball for structural robustness (User feedback)
                        # guided_ball was causing disconnected regions in early training
                        for node in k_hop_ball(self.neigh, a, radius, per_cap):
                            Rset.add(node)
                            if len(Rset) >= cap: break
                        if len(Rset) >= cap: break
                if not Rset:
                    # fallback: include the unlocked set itself (clipped)
                    Rset = set(unlocked[:cap])
                R = list(Rset)

                # ---- Exact local solve with CP-SAT (transactional)
                snap = self.core.copy_snapshot()

                S_loc, loc_cost, Rlist = solve_local_cpsat_region(self.core, R, time_limit=cp_time, workers=workers)

                if S_loc is not None:
                    for i, vtx in enumerate(Rlist):
                        self.core._set_label(vtx, int(S_loc[i]))
                    # local intensification
                    pc = self.core.pair_candidates()
                    if pc: self.core.pass_pair(pc, limit=24)
                    self.core.prune_full(3)

                # rollback if infeasible
                if self.core.viol_count != 0:
                    self.core.restore_snapshot(snap)

                # accounting & learning
                c1 = self.core.cost()
                reward = float(c0 - c1)

                Xn = self.core.node_features()
                gn = self.core.global_features(stagn)
                self.replay.add(Event(unlocked=unlocked, reward=reward, stagn=stagn, X=X_taken, g=g_taken))

                if reward > 0 and self.core.viol_count == 0:
                    stagn = 0
                    pool.try_add(self.core.S)
                else:
                    stagn += 1

                # periodic path-relinking
                if pr_every > 0 and (it % pr_every) == 0 and pool.pool:
                    tgt = pool.farthest(self.core.S) or pool.best()
                    if tgt is not None:
                        snap2 = self.core.copy_snapshot()
                        # NEW: Pass scores to path_relink to prioritize transitions
                        imp, drop = path_relink(self.core, tgt, max_steps=15, scores=scores)
                        if imp and self.core.viol_count == 0:
                            pool.try_add(self.core.S);
                            stagn = 0
                        else:
                            self.core.restore_snapshot(snap2)

                # update global best only if feasible
                if self.core.viol_count == 0 and self.core.cost() < bestC:
                    bestC = self.core.cost();
                    bestS = self.core.S.copy()

                # Learn every 16 iters
                if it % 16 == 0:
                    self._learn(batch=256)
                if stagn >= 4: break

        return bestS, bestC

def solve_dir(data_dir: str, out_path: str, iters: int = 500, starts: int = 5,
              cp_time: float = 0.30, topk: int = 256, device: str = None,
              workers: int = 1):
    # CSV logging setup
    csv_file = None
    writer = None
    # If out_path is "results.txt", csv will be "results.txt.csv"
    # If out_path is "results.csv", reuse it.
    csv_path = out_path if out_path.endswith('.csv') else out_path + ".csv"

    try:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(['Graph', 'Method', 'Cost', 'Time', 'Iterations'])
        print(f"[INFO] CSV logging enabled: {csv_path}")
    except Exception as e:
        print(f"[WARN] Could not open CSV file {csv_path}: {e}", file=sys.stderr)

    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mtx.gz")])
    t0 = time.time()

    # Text output file
    with open(out_path, "w", encoding="utf-8") as f:
        for fp in files:
            base = os.path.basename(fp)

            # Explicit cleanup
            if TORCH_OK:
                torch.cuda.empty_cache()
            gc.collect()

            try:
                n, neigh = read_mtx_gz(fp)
                solver = NeuroCPLNS(n, neigh, device=device)

                t1 = time.time()
                S, c = solver.solve(
                    iters=iters, starts=starts, cp_time=cp_time, topk=topk,
                    workers=workers
                )
                secs = time.time() - t1

                # Ensure we have a Python-list-looking solution
                try:
                    sol_text = str(S.tolist())
                except Exception:
                    sol_text = str(list(S))  # fallback

                # Optional: verify feasibility; if infeasible, keep the block shape and mark cost -1
                ok, _, _ = verify_feasible(S, neigh)
                cost_out = c if ok else -1

                block = (
                    f"Graph: {base}\n"
                    f"Solution: {sol_text}\n"
                    f"Cost: {int(cost_out)}\n"
                    f"Time(s): {secs:.6f}\n"
                )

                # stdout
                print(block, end="")

                # file
                f.write(block + "\n")  # blank line between instances

                # CSV - Hardcoded Method "NeuroCP-LNS"
                if writer:
                    writer.writerow([base, "NeuroCP-LNS", int(cost_out), f"{secs:.4f}", iters])
                    csv_file.flush() # ensure data is written

            except Exception as e:
                # Handle OOM by retrying on CPU if applicable
                is_oom = "out of memory" in str(e).lower()
                # Check device carefully
                is_gpu = False
                if device is not None and ('cuda' in str(device).lower() or 'gpu' in str(device).lower()):
                    is_gpu = True

                if is_oom and (is_gpu or device is None):
                    print(f"[WARN] {base}: GPU OOM. Skipping CPU retry...", file=sys.stderr)
                    # Clear GPU cache
                    if TORCH_OK:
                        torch.cuda.empty_cache()

                    # Delete old solver explicitly
                    if 'solver' in locals():
                        del solver
                    gc.collect()

                # Emit the same 4-line shape on failure
                block = (
                    f"Graph: {base}\n"
                    f"Solution: []\n"
                    f"Cost: -1\n"
                    f"Time(s): 0.000000\n"
                )
                print(block, end="")
                f.write(block + "\n")
                # error details to stderr so they don't pollute the strict format
                print(f"[ERROR] {base}: {e}", file=sys.stderr)
                # CSV fail
                if writer:
                    writer.writerow([base, "NeuroCP-LNS", -1, 0.0, iters])
                    csv_file.flush()

    # keep overall timing off the main output format
    print(f"[INFO] Total time: {time.time() - t0:.2f}s", file=sys.stderr)
    if csv_file:
        csv_file.close()
        print(f"[INFO] Saved CSV results to {csv_file.name}", file=sys.stderr)

# ----- tiny verifier (debug aid) -----
def verify_feasible(S: np.ndarray, neigh: List[List[int]]) -> Tuple[bool, int, List[int]]:
    viol_nodes = []
    for v in range(len(S)):
        if S[v] == 0:
            n2 = n3 = 0
            for u in neigh[v]:
                if S[u] == 2:
                    n2 += 1
                elif S[u] == 3:
                    n3 += 1
            if not (n3 >= 1 or n2 >= 2):
                viol_nodes.append(v)
    return (len(viol_nodes) == 0, len(viol_nodes), viol_nodes[:10])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_s = sub.add_parser("solve")
    ap_s.add_argument("--data_dir", required=True)
    ap_s.add_argument("--out", required=True)
    ap_s.add_argument("--iters", type=int, default=1000)
    ap_s.add_argument("--starts", type=int, default=5)
    ap_s.add_argument("--cp_time", type=float, default=0.30)
    ap_s.add_argument("--topk", type=int, default=256)
    ap_s.add_argument("--device", type=str, default=None)
    ap_s.add_argument("--workers", type=int, default=1,  # <<< CP-SAT threads
                      help="OR-Tools CP-SAT worker threads (1 = single CPU)")

    args = ap.parse_args()
    if args.cmd == "solve":
        solve_dir(
            args.data_dir,
            args.out,
            iters=args.iters,
            starts=args.starts,
            cp_time=args.cp_time,
            topk=args.topk,
            device=args.device,
            workers=args.workers,  # <<< pass through
        )
    else:
        print(
            "Usage:\n  python drdp_neurocp_lns.py solve --data_dir DIR --out OUT.txt "
            "[--iters 500] [--starts 5] [--cp_time 0.30] [--topk 256] [--device cuda] [--workers 1]"
        )


if __name__ == "__main__":
    main()
