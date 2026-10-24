# collect: how the training set is made

The 500k training set and the held-out pools come from two files:
`collect.py` (the Craftax parts: policies, the rules for forced attempts, the trip planner, the event list) and
`core.py` (the game-free parts: behaviour schedule, return trips, cutting and packing).

```
python collect/collect.py pool --out data/pool0 --seed 10                                   # 512 worlds x 4000 steps
python collect/collect.py pool --out data/eval --seed 7101 --worlds 256 --steps 2000 --states  # a held-out pool
python collect/collect.py pack --pools data/pool0 data/pool1 --out data                     # train.npz
```

`--policies` is a folder with the four checkpoints named in `POLICIES`.

## Who plays

Four RL policies run on every world at every step, so each one keeps its own memory; one of them acts.

| episodes | policy |
|---:|---|
| 60% | pixel PPO fine-tuned for diamonds (about 21 achievements per episode, a diamond in 75% of them), plus a small goal head; with no goal it plays as the plain policy |
| 15% | the same pixel PPO policy before the diamond fine-tune |
| 10% | a pixel policy distilled from the symbolic one |
| 15% | symbolic PPO |

Each episode draws its driver. About once per 800 steps another policy takes over for 50 to 400 steps.

## Behaviour schedule

Good agents have habits (the sword before the pickaxe, iron crafts right after the furnace is placed), and a world
model learns habits as if they were game rules. The schedule breaks them with generic tools. Each episode is calm
(intensity 0.2: long progress and diamonds) or chaotic (intensity 1: odd events and deaths); the intensity scales
every hazard below.

- **Hot windows.** About every 200 steps: 5 to 30 steps at sampling temperature 2 to 8, or in about a third of them
  3 to 24 uniformly random actions; then 20 to 80 steps at temperature 1 to 1.5. Paused in danger; never sleep.
- **Tool order.** Per tier and episode, a fair coin swaps "make pickaxe" and "make sword" until both tools of the tier
  are held. In half the episodes a tier's crafts wait until the materials for both tools are held. Now and then the
  stone and iron pickaxes and the swords are banned for 100 to 800 steps.
- **Setups.** After a table or a furnace is placed: a detour (hot steps, then no new setup for 60 to 250 steps), or
  the same action again 1 to 40 steps later. In 75% of the episodes the iron crafts wait 32 to 96 steps after a new
  furnace and the setup action itself 96 to 256 steps, so crafts also happen at an old furnace or table.
- **Needs.** Rarely, one need (eating, drinking, sleeping, fighting back) is ignored for 300 to 1500 steps. This gives
  deaths by starvation, thirst, fatigue and mobs.
- **Forced attempts.** Rules on the state propose actions that fail or work by the game logic: mining with a too weak
  tool, crafts at a table, iron crafts with no furnace or at an old one, the tier's second tool, second tables and
  furnaces, furnace presses that fail, plants and saplings, sleep by day, walks toward and into lava. At most one per
  4 to 16 steps.

## Return trips

Memory needs places that leave the view and come back. When the player is safe (daylight, awake, healthy, fed and
rested, no mob near), about once per 30 steps a trip draws a walking distance (2-3, 4-7, 8-15, 16-31, 32-63 or
64-127 tiles, mostly 4-15) and a tour. A is the start, R a place seen earlier (distinctive views are preferred), F a
point at that distance, G one at half of it. Tours: R (back to an old place), FA (out and back), GR, FGAG (A-B-C-A-C)
and RFA.

A breadth-first planner over grass, sand and path (no mobs, not next to lava) walks the legs, or, in half the trips,
the learned goal walker (the 60% policy steered by its goal head). One planner step in ten is the policy's own
action. Waypoints may get a pause, and the end gets a few free steps. Danger, no path, being stuck or running out of
steps ends a trip. A new table or furnace starts an out-and-back of 8+ tiles in 30% of the cases, so the agent comes
back to it.

## Packer

A pool is one long recording (512 worlds x 4000 steps, about 2M transitions). The packer:

1. marks every transition **busy** (an event from `events()`, a craft, place or sleep press, the do action on anything
   but ground, a trip step, a forced or repeated action) or **quiet** (no reward, no inventory or vitals change, no
   trip);
2. drops stretches with nothing busy for over 64 steps (16 steps kept at each end) and cuts the rest into pieces of
   about 250 to 1000 transitions that end at an episode end or on a quiet transition, never on an event;
3. picks pieces to balance events, not frames: first the pieces with diamonds (most first, up to 450 diamonds or half
   the set), then each time the piece with the best gain per (300 + its length). Each event type is worth up to its
   target (its supply, at most 400; deaths have their own targets per cause); counterexamples to known world-model
   flaws and a few rare events weigh 3x, deaths 6x;
4. packs the pieces into 250 streams of 2001 frames (longest first, first fit) and fills the free tails with the starts
   of spare pieces (the most eventful first).

## Data format

`train.npz`, the benchmark set:

| key | shape | meaning |
|---|---|---|
| frames | uint8 [2001, 250, 63, 63, 3] | 250 streams of 2001 frames |
| actions | int16 [2000, 250] | the 17 Craftax-Classic actions |
| rewards | float32 [2000, 250] | |
| dones | bool [2000, 250] | deaths only |
| truncated | bool [2000, 250] | a piece ends here (cut on a quiet transition, or the time limit) |
| meta | json string | recipe, policies, event supply and picks |

`actions[t, e]` and `rewards[t, e]` take `frames[t, e]` to `frames[t + 1, e]`. After a done or truncated transition,
`frames[t + 1, e]` is the first frame of the next piece.

`train_labels.npz` has side labels in the same layout, for analysis: per frame the true position, direction, the 7 x 9
view as tile ids, the HUD numbers, achievements, light, the facing block and table, furnace and mobs near; per
transition the death cause, driver, behaviour mode, intensity and trip records; `src` = (pool, world, frame) of every
frame.

A pool directory holds `frames.npy`, `labels.npz` and `records.npz`. With `--states` (held-out pools) it also holds
`states.npz`: the full game state before every action, actions, rewards, dones and the terminal states, so eval clips
can start from any state.
