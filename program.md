# program.md

How to beat tinywmcraft, for people and for coding agents.

The goal: a higher score than the best so far (iteration 3, 0.602 ± 0.005) from one 10-minute A100 run, without making the reward head worse.
The score is the mean of four parts, each 0 for a trivial model and 1 for the real game: gFVD, action following (AF), game logic and memory ([README](README.md#evaluation)).
Every iteration is on the [benchmark](https://maharishiva.github.io/tinywmcraft/benchmark.html).

## Rules

- **Data.** Train only on [`train.npz`](https://huggingface.co/datasets/maharishiva/tinywmcraft) (in `data/`), at most 500k transitions. No other data, no pretrained weights.
- **Training.** At most 600 s per run on one A100 40GB: data loading, tokenizer, compilation, everything. No persistent compilation cache. `train.py` stores its own wall clock as `train_s` in the checkpoint.
- **Evaluation.** `eval.py` stays byte-identical to `main`. Its `eval_s`, timed from importing your module to the last metric (model loading and compilation count, reading `eval.npz` does not), is at most 300 s on the same GPU; iteration 3 takes about 170 s. Anything else about generation is yours: the model family, the sampler, the number of steps, how much of the past the model keeps.
- **One frame per action.** The model generates frame by frame and never sees future actions or frames. The interface below hands it one action at a time; reading the eval data, caching frames across calls or other tricks around it are against the rules and checked in review.
- **Seeds.** Training seeds 0, 1 and 2. Report all three rows.
- **Gates.** Measured in the maintainer's rerun, on the same machine and seeds, on a private test set: clips built by `make_eval.py` in the same way from worlds that are not in `eval.npz`. Ranked on the score alone:
  - score: the new mean minus the best mean must exceed `1.65 * sqrt((sd_best^2 + sd_new^2) / 3)`, where `sd` is the sample SD over the three seeds. Stable recipes need a smaller margin.
  - reward AP: the new three-seed mean stays at or above a fixed floor, 95% of iteration 1's mean (`train_tiny.py`). The floor does not move when the best changes.
  - done AP: reported, not gated. Deaths are rare (about 180 of the 2048 play clips), so it is too noisy to gate on.

| reward_ap |
|---:|
| 0.877 |

One run is noisy: compare three-seed means, never single runs.

## The model interface

`eval.py` imports `train` (or `--module name`) and calls only these:

```python
load_model(path) -> model
load_meta(path)  -> {"cfg": {"seed": int, ...}, "train_s": float, ...}
start(model, frames, actions) -> state            # frames (B, 8, 63, 63, 3) uint8, actions (B, 8): the action INTO each frame
step(model, state, action, key) -> frame, state   # action (B,), frame (B, 63, 63, 3) uint8: exactly one new frame
heads(model, frames, actions, key) -> reward, done  # teacher-forced on 24 real frames, (B, 24) each; done is a probability, done[:, 0] = 0
```

For every clip, eval calls `start` with 8 real frames, then `step` once per action: 64 times for a play clip (and 16 times under random actions, for AF), 8 times for a logic case, 48 or 144 times for a memory walk. `heads` reads the play clips' real frames, 24 at a time.
`play_wm.py` uses the same interface, so any entry can be played.

## The loop

1. Read `README.md`, this file, and `train.py`.
2. Pick one idea. Change `train.py` and `kernels.py` (new files they import are fine). Never change `eval.py` or the data.
3. Train: `python train.py --seed 0 --out runs/<tag>/model.npz`. Check that `train_s` is at most 600.
4. Score: `python eval.py --model runs/<tag>/model.npz --tag <tag> --note "<the idea in a few words>"`. Every run appends a row to `ledger.jsonl`, kept or not.
5. Keep the change if the score rises and the reward head holds; otherwise revert it. The parts in the ledger row show where it came from.
6. When an idea survives one seed, run seeds 1 and 2 and check the gates above.

Exploratory runs can use any GPU and any seed; timings from other GPUs are only indicative. Iterations are decided on a Colab A100-SXM4-40GB.

## Submitting

1. Open a pull request with your changes. The template asks for the three `ledger.jsonl` rows and links to the three `model.npz` files on Hugging Face, so anyone can play and score them.
2. The maintainer retrains the three seeds from the pull request on a Colab A100 next to the best so far and scores them on the private test set.
3. If the rerun passes the gates, the maintainer adds it to `docs/records.jsonl` as the next iteration and merges. The benchmark on the site updates with the merge and credits you with a link to the pull request.
4. The crown goes next to your GitHub login, and your model becomes the one the site plays. If it does not export to the browser with `docs/export_web.py`, the maintainer helps.

## Open problems

- **Game logic.** Crafting is the hardest logic family: repeat crafts, iron tools without a furnace, the wrong pickaxe. A model that gets the rules right scores better, and makes the [challenge](https://maharishiva.github.io/tinywmcraft/#challenge) in the browser fairer.
- **Memory.** Every model so far draws a cell that comes back after 16 or more frames no better than chance. The context length is yours; the long walks reward a model that remembers.
- **Cheaper sampling.** Eight Euler steps per frame. Fewer steps, better solvers, distillation, or a model that is not flow matching at all could free eval time and let a bigger model train.
- **Long horizons.** The benchmark scores up to 144 frames; a run in the browser lasts up to 1000 moves. Anything that keeps the world coherent longer shows in the player first.
