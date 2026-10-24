"""Frozen benchmark: open-loop generation scored for looks (gFVD), action following (AF), game logic and memory, plus
the reward and termination heads, on fixed clips of held-out real play (built by make_eval.py).

Every clip starts with 8 real frames and the action into each; the model generates one frame per action after them.
  play    2048 windows of the data driver's play, 64 actions each; 512 hold a diamond, an iron tool, a plant eaten
          or a death in their first 16 future frames (a death clip ends at the death; later frames are not scored).
          Also rolled out 16 frames under random actions with the same sampling keys.
  logic   8 scripted actions from a real state next to something to use, in 7 families: chop, mine (with the right
          and the wrong pickaxe), drink, eat, place, craft (with and without table, furnace or ingredients, in any
          order), sleep. In each family the real game changed something in half of the cases, nothing in the rest.
  memory  512 planned walks out and back (48 actions) and 256 long ones (144 actions): map cells seen in the context
          that left the view and come back unchanged, on a screen tile that showed another block in the last context
          frame.

Probes read generated frames with the game's own textures (stored in the data file): a HUD slot is the nearest of its
10 templates (icon and digit), a map cell the nearest block or mob texture as rendered at the clip's light, sleep the
gray map tint.
  gFVD     Frechet distance between the real and generated play frames 1-64 in the feature space of a fixed random
           CNN (seed 1234, never trained) over stacks of 4 consecutive frames, times 10^4. Lower is better.
  AF       the share of play clips where the rollout under the recorded actions is closer (salient-pixel MSE over
           frames 1-16) to the real future than the rollout under random actions; ties score half. 0.5 = blind.
  logic    a case passes if the HUD items (and food or drink where they are the point), the target cell and the sleep
           tint read the same as in the real game on all 8 generated frames. Per family: pass rate where something
           changed + pass rate where nothing changed - 1 (Youden's J; 0 for "never changes" and "always changes").
           logic = the mean over families.
  memory   recall: the share of returning cells whose generated block reads as the real one, per gap bucket (frames
           since the cell was last in view): 4-7, 8-15, 16-31, 32-63, 64+. Its gain over chance per bucket is
           (recall - chance) / (1 - chance), chance = the probability that the block drawn equals a random block of
           the last context frame; painting the common block, shuffling the context's tiles or copying the last
           frame gains nothing. memory = the mean gain over the buckets.
  reward_ap / done_ap  the heads teacher-forced on the real play frames, each future frame read with 8 frames of
           context before it: average precision of |predicted reward| for the reward events and of the done
           probability for the deaths. reward_base / done_base are the shares of positives.
  score    the north star, the mean of four parts, each clipped to [0, 1], 0 at a trivial model and 1 at the real
           game: gFVD log(GFVD_ZERO / gFVD) / log(GFVD_ZERO / GFVD_ONE), AF (AF - 0.5) / (AF_ONE - 0.5), logic,
           memory. GFVD_ZERO is the gFVD of copying the last context frame; GFVD_ONE and AF_ONE are the gFVD and AF
           of the real game itself, replayed from the same states with fresh randomness (mobs, saplings, night).

The model is any module (default train.py) with this interface; how it generates a frame is its own business, and
so is how much of the past it keeps:
  load_model(path), load_meta(path) -> {"cfg": {"seed": ...}, "train_s": ...}
  start(model, frames (B, n0, H, W, 3) uint8, actions (B, n0)) -> state       actions[t] is the action INTO frame t
  step(model, state, action (B,), key) -> (frame (B, H, W, 3) uint8, state)  one frame, never sees future actions
  heads(model, frames (B, L, H, W, 3), actions (B, L), key) -> (reward (B, L), done probability (B, L)), L = 24

eval_s is timed from importing the model module to the last metric, so model loading and compilation count and
reading the data does not; it must stay within 300 s on the maintainer's A100.

    python eval.py --model runs/base/model.npz --tag base
"""
import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

H = W = 63
A_N = 17
CTX, FUT, AF_HOR, HEAD_HOR = 8, 64, 16, 16
EVAL_SEED, BATCH, FEAT_BATCH, PROBE_BATCH = 0, 256, 16, 2048
CLIP, FEAT_SEED, THR = 4, 1234, 16 / 255
GFVD_SCALE, TRAIN_BUDGET_S, EVAL_BUDGET_S = 1e4, 600, 300
GFVD_ZERO, GFVD_ONE, AF_ONE = 170.0, 14.0, 0.93     # measured on data/eval.npz (copy, replayed game)
GAPS = (4, 8, 16, 32, 64, 10 ** 4)
GROUPS = ("play", "logic", "memory", "long")


class FeatCNN(nnx.Module):
    """Fixed random conv stack over CLIP stacked frames; the final 4x4 map is kept flat so layout counts."""
    def __init__(self, rngs):
        self.c1 = nnx.Conv(3 * CLIP, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(64, 128, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(128, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)

    def __call__(self, x):
        h = nnx.gelu(self.c1(x))
        h = nnx.gelu(self.c2(h))
        h = nnx.gelu(self.c3(h))
        return self.c4(h).reshape(x.shape[0], -1)


def frame_features():
    """A function from frames (B, T, H, W, 3) uint8 to features (B, T, F): for every frame, the stack of CLIP frames
    ending at it (the first frame repeated before the start) through the fixed CNN, on the device."""
    cnn = FeatCNN(nnx.Rngs(FEAT_SEED))

    @nnx.jit
    def run(cnn, video):
        B, T = video.shape[:2]
        v = video.astype(jnp.float32) / 255
        v = jnp.concatenate([jnp.repeat(v[:, :1], CLIP - 1, 1), v], 1)
        stacks = jnp.concatenate([v[:, t:t + T] for t in range(CLIP)], -1)
        return cnn(stacks.reshape(B * T, H, W, 3 * CLIP)).reshape(B, T, -1)

    def features(video):
        with jax.default_matmul_precision("tensorfloat32"):   # fixed, whatever the module sets
            return np.concatenate([np.asarray(run(cnn, video[i:i + FEAT_BATCH]))
                                   for i in range(0, len(video), FEAT_BATCH)])
    return features


def frechet(fr, fg):
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False) + 1e-6 * np.eye(fr.shape[1])
    cg = np.cov(fg, rowvar=False) + 1e-6 * np.eye(fg.shape[1])
    eig = np.linalg.eigvals(cr @ cg)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cr) + np.trace(cg) - 2 * np.sum(np.sqrt(np.clip(eig.real, 0, None))))


def salient_mse(pred, real, last_ctx, valid):
    """Per clip, the squared error over the pixels of valid frames where the real future differs from the last
    context frame."""
    real, pred = real.astype(np.float32) / 255, pred.astype(np.float32) / 255
    sal = (np.abs(real - last_ctx[:, None]).max(-1, keepdims=True) > THR) & valid[:, :, None, None, None]
    return ((pred - real) ** 2 * sal).sum((1, 2, 3, 4)) / (sal.sum((1, 2, 3, 4)) + 1e-6)


def average_precision(scores, labels):
    """Area under the precision-recall curve: the mean precision at the rank of each positive; None without one."""
    hits = np.asarray(labels).ravel() > 0.5
    if not hits.any():
        return None
    hits = hits[np.argsort(-np.asarray(scores, np.float64).ravel(), kind="stable")]
    return float(np.mean(np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)))


def nearest(x, tpl):
    """x (N, K, 7, 7, 3), tpl (N or 1, K, C, 7, 7, 3) -> (N, K) index of the nearest template (mean squared error)."""
    return np.asarray(jnp.argmin(jnp.mean((jnp.asarray(x)[:, :, None] - jnp.asarray(tpl)) ** 2, (-1, -2, -3)), -1))


def read_hud(frames, tpl):
    """frames (N, H, W, 3) uint8 -> (N, 16) counts 0-9 of the HUD slots (health food drink energy sapling wood stone
    coal iron, then diamond and the six tools) under the map."""
    slots = lambda f: f[:, 49:].reshape(-1, 2, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 18, 7, 7, 3)[:, :16]
    return np.concatenate([nearest(slots(frames[i:i + PROBE_BATCH]).astype(np.float32), tpl[None])
                           for i in range(0, len(frames), PROBE_BATCH)])


def read_cells(frames, rc, light, tex, night):
    """frames (N, H, W, 3) uint8, rc (N, 2) map cell (row 0-6, column 0-8), light (N,) -> (N,) the nearest tile class:
    block id - 1 (0-15), then cow, zombie, skeleton, each as rendered at that light (no static above 0.5)."""
    out = []
    for i in range(0, len(frames), PROBE_BATCH):
        f, r, c, d = frames[i:i + PROBE_BATCH], rc[i:i + PROBE_BATCH, 0], rc[i:i + PROBE_BATCH, 1], light[i:i + PROBE_BATCH]
        x = f[np.arange(len(f))[:, None, None], r[:, None, None] * 7 + np.arange(7)[:, None], c[:, None, None] * 7 + np.arange(7)]
        tpl = d[:, None, None, None, None] * tex + (1 - d[:, None, None, None, None]) * night
        out.append(nearest(x[:, None].astype(np.float32), tpl[:, None])[:, 0])
    return np.concatenate(out) if out else np.zeros(0, int)


def asleep(frames):
    """frames (N, H, W, 3) uint8 -> (N,) the map shows the sleep tint (gray, blue = red + 8)."""
    m = frames[:, :49].astype(np.int16)
    return (np.abs(m[..., 0] - m[..., 1]) + np.abs(m[..., 2] - m[..., 0] - 8)).mean((1, 2)) < 6


def make_gif(real, predicted, path, k=8, scale=4):
    import imageio.v2 as imageio

    k = min(k, len(real))
    separator = np.full((H, 2, 3), (90, 90, 110), np.uint8)
    images = []
    for t in range(real.shape[1]):
        grid = np.concatenate([np.concatenate([real[i, t], separator, predicted[i, t]], 1) for i in range(k)], 0)
        images.append(np.repeat(np.repeat(grid, scale, 0), scale, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 6, loop=0)


def rollout(wm, model, context, context_actions, actions, keys):
    """Frames (B, T, H, W, 3) generated one action of actions (B, T) at a time after the real context, one key each."""
    state = wm.start(model, context, context_actions)
    out = []
    for action, key in zip(actions.T, keys):
        frame, state = wm.step(model, state, jnp.asarray(action), key)
        out.append(frame)
    return np.asarray(jnp.stack(out, 1))


def batches(d, g, *extra):
    """Per batch of BATCH clips of a group (the last one padded with copies of its last clip, so every call has the
    same shape): the number of real clips, their frames, actions and extra arrays, and the batch's sampling key."""
    n = len(d[f"{g}_frames"])
    for b0 in range(0, n, BATCH):
        key = jax.random.PRNGKey(int(np.random.SeedSequence([EVAL_SEED, GROUPS.index(g), b0]).generate_state(1)[0]))
        idx = np.minimum(np.arange(b0, b0 + BATCH), n - 1)
        yield min(BATCH, n - b0), d[f"{g}_frames"][idx], d[f"{g}_actions"][idx].astype(np.int32), [x[idx] for x in extra], key


def read_heads(wm, model, frames, actions, key):
    """Reward and done probability (B, FUT) of the transition into each future frame, from windows of CTX + HEAD_HOR
    frames every HEAD_HOR frames, each scoring its last HEAD_HOR."""
    reward, done = np.zeros((len(frames), FUT), np.float32), np.zeros((len(frames), FUT), np.float32)
    for w, s in enumerate(range(0, FUT, HEAD_HOR)):
        window = slice(s, s + CTX + HEAD_HOR)
        r, d = wm.heads(model, frames[:, window], actions[:, window], jax.random.fold_in(key, w))
        reward[:, s:s + HEAD_HOR], done[:, s:s + HEAD_HOR] = np.asarray(r)[:, CTX:], np.asarray(d)[:, CTX:]
    return reward, done


def score_play(wm, model, d):
    features = frame_features()
    out = {k: [] for k in ("fr", "fg", "err_true", "err_rand", "r_tf", "d_tf")}
    valid = CTX + np.arange(FUT)[None] < d["play_valid_len"][:, None]
    random_actions = np.random.default_rng(EVAL_SEED + 7).integers(0, A_N, (len(valid), AF_HOR)).astype(np.int32)
    preview = None
    for n, frames, actions, (rand, ok), key in batches(d, "play", random_actions, valid):
        kg, kt = jax.random.split(key)
        keys = jax.random.split(kg, FUT)
        context, context_actions = frames[:, :CTX], actions[:, :CTX]
        generated = rollout(wm, model, context, context_actions, actions[:, CTX:], keys)[:n]
        randomized = rollout(wm, model, context, context_actions, rand, keys[:AF_HOR])[:n]
        reward, done = read_heads(wm, model, frames, actions, kt)
        real, last, ok = frames[:n, CTX:], frames[:n, CTX - 1].astype(np.float32) / 255, ok[:n, :AF_HOR]
        out["err_true"].append(salient_mse(generated[:, :AF_HOR], real[:, :AF_HOR], last, ok))
        out["err_rand"].append(salient_mse(randomized, real[:, :AF_HOR], last, ok))
        out["fr"].append(features(real))
        out["fg"].append(features(generated))
        out["r_tf"].append(reward[:n])
        out["d_tf"].append(done[:n])
        if preview is None:
            preview = real[:8].copy(), generated[:8].copy()
    o = {k: np.concatenate(v) for k, v in out.items()}
    reward, death = d["play_rewards"][:, CTX:], (d["play_dones"][:, CTX:] > 0) & valid
    fd = lambda hi: GFVD_SCALE * frechet(o["fr"][:, :hi][valid[:, :hi]], o["fg"][:, :hi][valid[:, :hi]])
    t, r = o["err_true"], o["err_rand"]
    event = np.abs(reward[valid]) > 1e-6
    return {"gFVD": fd(FUT), "gFVD_16": fd(AF_HOR), "AF": float(np.mean((r > t + 1e-8) + 0.5 * (np.abs(r - t) <= 1e-8))),
            "reward_ap": average_precision(np.abs(o["r_tf"][valid]), event), "reward_base": float(event.mean()),
            "done_ap": average_precision(o["d_tf"][valid], death[valid]), "done_base": float(death[valid].mean())}, preview


def generate(wm, model, d, g):
    out = []
    for n, frames, actions, _, key in batches(d, g):
        keys = jax.random.split(key, actions.shape[1] - CTX)
        out.append(rollout(wm, model, frames[:, :CTX], actions[:, :CTX], actions[:, CTX:], keys)[:n])
    return np.concatenate(out)


def score_logic(wm, model, d):
    gen = generate(wm, model, d, "logic")
    n, L = gen.shape[:2]
    flat = gen.reshape(n * L, H, W, 3)
    cell = d["logic_cell"]
    ok = ((read_hud(flat, d["hud_tpl"]).reshape(n, L, 16) == d["logic_hud"]) | ~d["logic_slots"][:, None]).all((1, 2))
    cells = read_cells(flat, np.repeat(cell.clip(0), L, 0), d["logic_light"].ravel(), d["tile_tex"], d["tile_night"])
    ok &= (cell[:, 0] < 0) | (cells.reshape(n, L) == d["logic_tile"]).all(1)
    ok &= (asleep(flat).reshape(n, L) == d["logic_sleep"]).all(1)
    fam, changed = d["logic_family"], d["logic_changed"]
    rates = {str(f): (ok[(fam == i) & changed].mean(), ok[(fam == i) & ~changed].mean())
             for i, f in enumerate(d["logic_families"]) if ((fam == i) & changed).any() and ((fam == i) & ~changed).any()}
    j = {f: float(c + u - 1) for f, (c, u) in rates.items()}
    return {"logic": float(np.mean(list(j.values()))), "logic_pass": float(ok.mean()), "logic_families": j,
            "logic_pass_changed": {f: float(c) for f, (c, u) in rates.items()},
            "logic_pass_unchanged": {f: float(u) for f, (c, u) in rates.items()}}


def score_memory(wm, model, d):
    """Recall of the returning cells per gap bucket, and its gain over chance: the probability that the block drawn
    equals a random block of the clip's last context frame."""
    hit, gap, chance = [], [], []
    for g in ("memory", "long"):
        gen, ch = generate(wm, model, d, g), d[f"{g}_checks"].astype(int)
        drawn = read_cells(gen[ch[:, 0], ch[:, 1]], ch[:, 2:4], d[f"{g}_light"][ch[:, 0], ch[:, 1]], d["tile_tex"],
                           d["tile_night"])
        hit.append(drawn == ch[:, 4])
        gap.append(ch[:, 5])
        chance.append(d[f"{g}_prior"][ch[:, 0], drawn])
    hit, gap, chance = map(np.concatenate, (hit, gap, chance))
    bucket = np.digitize(gap, GAPS) - 1
    names = [f"{lo}-{hi - 1}" if hi < GAPS[-1] else f"{lo}+" for lo, hi in zip(GAPS, GAPS[1:])]
    rows = [(n, bucket == i) for i, n in enumerate(names) if (bucket == i).any()]
    gain = {n: float((hit[r].mean() - chance[r].mean()) / (1 - chance[r].mean())) for n, r in rows}
    return {"memory": float(np.mean(list(gain.values()))), "memory_gain": gain,
            "memory_recall": {n: float(hit[r].mean()) for n, r in rows}, "memory_checks": [int(r.sum()) for _, r in rows]}


def north_star(m):
    parts = {"gfvd": np.log(GFVD_ZERO / max(m["gFVD"], 1e-9)) / np.log(GFVD_ZERO / GFVD_ONE),
             "af": (m["AF"] - 0.5) / (AF_ONE - 0.5), "logic": m["logic"], "memory": m["memory"]}
    parts = {k: float(np.clip(v, 0, 1)) for k, v in parts.items()}
    return float(np.mean(list(parts.values()))), parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--module", default="train", help="the module with the model interface")
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"))
    p.add_argument("--out", type=Path, default=Path("ledger.jsonl"))
    p.add_argument("--tag")
    p.add_argument("--note", default="")
    p.add_argument("--every", type=int, default=1, help="score every k-th clip of each group, a quick check")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    data = np.load(a.data)
    d = {k: data[k] for k in data.files if k != "meta"}
    for g in GROUPS if a.every > 1 else ():
        n, ch = len(d[f"{g}_frames"]), d.get(f"{g}_checks")
        d |= {k: v[::a.every] for k, v in d.items() if k.startswith(g + "_") and len(v) == n and k != f"{g}_checks"}
        if ch is not None:
            d[f"{g}_checks"] = ch[ch[:, 0] % a.every == 0] // [a.every, 1, 1, 1, 1, 1]
    start = time.perf_counter()
    wm = importlib.import_module(a.module)
    model, meta = wm.load_model(a.model), wm.load_meta(a.model)
    metrics, preview = score_play(wm, model, d)
    metrics |= score_logic(wm, model, d) | score_memory(wm, model, d)
    score, parts = north_star(metrics)
    eval_s = round(time.perf_counter() - start, 1)
    clips = {g: len(d[f"{g}_frames"]) for g in GROUPS}
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": a.tag or a.model.parent.name,
           "note": a.note, "score": score, "parts": parts, **metrics, "train_seed": meta["cfg"]["seed"], "clips": clips,
           "batch": BATCH, "train_s": meta["train_s"], "eval_s": eval_s,
           "in_budget": a.every == 1 and meta["train_s"] <= TRAIN_BUDGET_S and eval_s <= EVAL_BUDGET_S,
           "module": a.module, "checkpoint": str(a.model), "data": str(a.data)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if a.gif:
        make_gif(*preview, a.out.parent / "gifs" / f"{row['tag']}.gif")


if __name__ == "__main__":
    main()
