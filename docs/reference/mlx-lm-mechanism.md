# mlx-lm mechanism reference (for Track B / M1)

Upstream: `~/Documents/Guest/mlx-lm` @ **0.31.3** (HEAD `2ed22318`, 2026-06-24). This is the
UPSTREAM ORIGINAL of the batched decode we port to Swift. Read the clone for full context; the
load-bearing pieces are reproduced here so the port stays faithful even without clone access.

**Architecture note:** mlx-lm does NOT build the batch cache from an explicit left-padding vector.
Each sequence is prefilled into its own single-row `KVCache`, and N of those are MERGED into one
`BatchKVCache` (`merge` derives `left_padding` from the length gaps). The explicit-padding
constructor is dead code. → **Prefill per-row serial, then merge; batch only the decode.**

---

## 1. Ragged causal mask — `create_causal_mask` (`mlx_lm/models/base.py:24-42`)

```python
def create_causal_mask(N, offset=0, window_size=None, right_padding=None, left_padding=None):
    rinds = mx.arange(offset + N)                          # key positions
    linds = mx.arange(offset, offset + N) if offset else rinds   # query positions
    linds = linds[:, None]                                 # (N, 1)
    rinds = rinds[None]                                    # (1, offset+N)
    mask = linds >= rinds                                  # boolean causal triangle
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)   # ← per-row pad mask
    return mask
```

- **Boolean** mask, shape `(N, offset+N)` → `(B,1,N,offset+N)` after the `(1,2,3)` batch broadcast.
- `left_padding` is a **per-row vector** (length B); `left_padding <= rinds` masks each row's first
  `left_padding[row]` key columns. This is how ragged rows share one rectangular buffer.
- `offset` = the cache's shared write index `_idx`.

**`-inf` conversion happens at the attention call** (`base.py:87-96`):
```python
if mask.dtype == mx.bool_:
    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
```

**Dispatch:** `BatchKVCache.make_mask` (`cache.py:1011-1014`) ALWAYS calls `create_causal_mask`
with `offset=self._idx, left_padding=self.left_padding`. The `N==1 → None` fast path is in
`create_attention_mask` (`base.py:45-55`), the single-sequence router — **NOT** the batch path.

> ⚠ PORT RULE: the `N==1`→no-mask fast path is valid ONLY when `leftPadding.max()==0`. A ragged
> batched decode step (1 new token/row) STILL needs the mask, or rows attend to their own left-pad
> columns and silently diverge. Gate the fast path on `leftPadding.max()==0`.

---

## 2. `BatchKVCache` (`mlx_lm/models/cache.py:912-1130`)

State: `keys/values` `(B, n_kv_heads, T, head_dim)`, `left_padding` (per-row vec),
`offset = -left_padding` (per-row, starts negative), `_idx` (shared scalar fill level).

**merge — the real construction path** (`cache.py:1088-1118`):
```python
@classmethod
def merge(cls, caches):
    lengths = [c.size() for c in caches]; max_length = max(lengths)
    padding = [max_length - l for l in lengths]           # left_padding DERIVED
    keys   = mx.zeros((B, H, max_length, Dk), dtype=dt)
    values = mx.zeros((B, H, max_length, Dv), dtype=dt)
    for i, (p, c) in enumerate(zip(padding, caches)):
        if c.keys is None: continue
        keys  [i:i+1, :, p:p+c.offset] = c.keys  [..., :c.offset, :]   # right-justify each row
        values[i:i+1, :, p:p+c.offset] = c.values[..., :c.offset, :]
    cache = cls(padding); cache.keys = keys; cache.values = values
    cache.offset += keys.shape[2]; cache._idx = keys.shape[2]
```

**extract — peel one finished row back to a contiguous normal cache** (`cache.py:1080-1086`):
```python
def extract(self, idx):
    cache = KVCache(); padding = self.left_padding[idx].item()
    cache.keys   = mx.contiguous(self.keys[idx:idx+1, :, padding:self._idx])
    cache.values = mx.contiguous(self.values[idx:idx+1, :, padding:self._idx])
    cache.offset = cache.keys.shape[2]
    return cache
```

**filter (continuous batching only, M1.5)** (`cache.py:1016-1033`): fancy-index kept rows, then
re-minimize padding: `min_left_pad = left_padding.min()`; if `>0`, slice `[..., min_left_pad:, :]`
off the front and decrement `_idx` + `left_padding`.

**Defer:** `prepare`/`finalize`/`dynamic_roll` (`cache.py:967-988`, `:903`) — the right-pad-prefill
+ modular-roll path. Unneeded because we prefill per-row serially (true length) then `merge`.

---

## 3. Decode step — `GenerationBatch._step` (`mlx_lm/generate.py:1320-1378`)

```python
logits = self.model(inputs[:, None], cache=self.prompt_cache)   # (B,1) forward
logits = logits[:, -1, :]
logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
sampled = self.fallback_sampler(logprobs)          # or per-row samplers[e]
self._next_tokens = sampled
mx.async_eval(self._next_tokens, ...)              # kick next step BEFORE returning current
mx.eval(inputs, self._current_logprobs)
```
Replicate the `async_eval(next) … eval(current)` **double-buffer** — it overlaps GPU work for
step k+1 with host bookkeeping of step k. Greedy iff sampler temp == 0.

Row management (M1.5): `filter`/`extend`/`insert`/`remove` (`generate.py:1383-1464`, `:1293-1318`,
`:1585-1767`) + `SequenceStateMachine.match` for multi-token stop sequences.

---

## 4. Prompt-cache serialization (for Track A) — `mlx_lm/models/cache.py:15-85,127-175`

`state`/`meta_state` contract: `KVCache.state` → `(keys, values)` (trimmed to `[:offset]`);
`meta_state` → tuple of **stringified scalars** (safetensors metadata is string-only).

```python
def save_prompt_cache(file_name, cache, metadata={}):
    cache_data     = dict(tree_flatten([c.state for c in cache]))
    cache_classes  = [type(c).__name__ for c in cache]
    cache_metadata = dict(tree_flatten([[c.meta_state for c in cache], metadata, cache_classes]))
    mx.save_safetensors(file_name, cache_data, cache_metadata)

def load_prompt_cache(file_name, return_metadata=False):
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)   # ← G2: metadata loader
    ... cache = [globals()[c].from_state(state, meta_state) for c, state, meta_state in ...]
```

PORT: follow the `state`/`meta_state` split; store `offset` explicitly in `meta_state`; persist the
class name for load dispatch. Diverge: use explicit keyed tensor names (`layer.{i}.keys`) instead of
`tree_flatten` dotted auto-keys. **H7:** encode bools explicitly (`"1"/"0"`) — `Bool("False")` is
truthy (upstream bug mlx-lm #1251).

---

Full omlx serving-architecture specs (scheduler, block cache, hot/cold SSD, per-model handlers) are
in `~/Documents/Guest/omlx/omlx/` — see `PLAN.md` Appendices A/B for the file:line porting maps.
