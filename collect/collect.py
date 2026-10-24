"""Collect the Craftax-Classic training set and the held-out pools: play by pixel and symbolic PPO policies made more
varied by the behaviour schedule and the return trips of core.py, cut into pieces and packed so that events, not
frames, are balanced (500k transitions in 250 streams of 2000).

The Craftax part is this file:
  policies  a learned goal walker (the pixel PPO policy fine-tuned on diamonds plus a goal head; with no goal it plays
            as that policy) drives most episodes and walks half of the trips; the pixel policy before the fine-tune, a
            pixel policy distilled from the symbolic one and the symbolic policy drive the rest and take over for short
            windows (all four run on every world each step, so a switch keeps a full memory).
  breakers  make_X_pickaxe <-> make_X_sword swapped until both tools of a tier are held (a coin per tier and episode),
            tier crafts held until both tools are affordable; the stone and iron pickaxes and the swords banned for a
            while; place_table and place_furnace are the setups (place_furnace holds back the iron crafts); the needs
            are eating, drinking, sleeping and fighting back.
  forced    attempts that fail or work by the rules: mining with a too weak tool, crafts at a table, iron crafts with no
            furnace or at an old one, the tier's second tool, second tables and furnaces, failing furnace presses, more
            plants and saplings, sleeping by day, walks into lava.
  trips     the player position; a breadth-first distance field over grass, sand and path (no mob, not next to lava).
            Trips start in daylight when the player is awake, healthy and alone, and yield to danger and low needs.
  packing   busy: an event, a craft / place / sleep press, the do action on anything but ground, a trip step, a forced
            action; quiet (a cut is allowed): no reward, no change in the inventory or the vitals, no trip.

    python collect/collect.py pool --out data/pool0 --seed 10
    python collect/collect.py pool --out data/eval --seed SEED --worlds 256 --steps 2000 --states
    python collect/collect.py pack --pools data/pool0 data/pool1 --out data

A pool holds frames.npy (T + 1, E, 63, 63, 3), labels.npz (per frame: position, direction, tiles, HUD, achievements,
light, ...), records.npz (per transition: action, reward, deaths and causes, behaviour and trip records) and, with
--states, states.npz: the state before every action (leaf i as s.<i>, the map as uint8), actions, rewards, dones and
the terminal states (term.<i> at term_idx), recomputed from state, action and step key. pack writes train.npz (the
benchmark layout) and train_labels.npz (every packed frame's labels, records and (pool, world, frame) source).
"""
import argparse, json, pickle, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT, Achievement, BlockType as BT
from craftax.craftax_classic.game_logic import is_near_block
from craftax.craftax_classic.renderer import render_craftax_pixels, render_craftax_symbolic
from craftax.craftax_env import make_craftax_env_from_name
import core

A, DO, SLEEP, P_STONE, P_TABLE, P_FURNACE, P_PLANT = 17, 5, 6, 7, 8, 9, 10     # 11-13 make pickaxes, 14-16 swords
HUD = ("health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe",
       "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
ACH = tuple(a.name.lower() for a in Achievement)
CAUSES = ("none", "lava", "zombie", "skeleton", "starvation", "thirst", "exhaustion", "other", "time limit")
POLICIES = (("goal walker on the pixel PPO diamond fine-tune", "walker.pkl", 0.6), ("pixel PPO", "h2_p1.pkl", 0.15),
            ("pixel, distilled from symbolic", "opd_1.pkl", 0.1), ("symbolic PPO", "sym_survive.pkl", 0.15))
PRIORITY = dict(iron_pickaxe_first=3, iron_sword_after_pickaxe=3, stone_sword_after_pickaxe=3, iron_at_old_furnace=3,
                iron_press_spell=3, furnace_then_no_iron_press=3, iron_press_no_furnace=3, second_furnace=3,
                place_plant=3, more_sapling=3, unlock_eat_plant=3, return_scale2=3, death_lava_walk=6,
                **{f"death_{k}": 6 for k in CAUSES[1:7]})                                  # piece-choice weights
DEATHS = dict(death_lava=60, death_lava_walk=60, death_zombie=250, death_skeleton=110, death_thirst=150,
              death_starvation=120, death_exhaustion=80)                                  # their targets (others 400)
D4 = jnp.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]])                     # moves 1..4 and facing directions 1..4
WALK = jnp.asarray([BT.GRASS.value, BT.SAND.value, BT.PATH.value])
PLACED = jnp.asarray([BT.CRAFTING_TABLE.value, BT.FURNACE.value, BT.PLANT.value, BT.RIPE_PLANT.value])
INF, DAY = 16384, 0.6
env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
EP = env.default_params


# --- policies: plain JAX over the saved parameters -------------------------------------------------------------------
def _plain(t):
    return {k: _plain(v) for k, v in t.items()} if hasattr(t, "items") else jnp.asarray(getattr(t, "value", t))
def _dense(x, p, dt=jnp.float32): return x.astype(dt) @ p["kernel"].astype(dt) + p["bias"].astype(dt)


def goal_head(q, f, pos, goal, active):                                  # the walker's logit change
    relu, d = jax.nn.relu, (goal - pos).astype(jnp.float32)
    n = jnp.abs(d).sum(-1, keepdims=True)
    g = jnp.concatenate([jnp.ones_like(n), jnp.tanh(d / 4), jnp.tanh(d / 16), jnp.tanh(n / 16),
                         (n == 0).astype(jnp.float32), pos.astype(jnp.float32) / 64.0 - 0.5], -1) * active[:, None]
    ln = (f - f.mean(-1, keepdims=True)) * jax.lax.rsqrt(f.var(-1, keepdims=True) + 1e-6) * q["ln_p"]["scale"]
    x = jnp.concatenate([ln + q["ln_p"]["bias"], relu(_dense(g, q["ge_p"]))], -1)
    return _dense(relu(_dense(relu(_dense(x, q["p1"])), q["p2"])), q["po"])


def pixel(p, h, s, obs, pa, lead, target):  # Impala CNN (bf16) -> GRU 256 with the previous action -> MLP 512 -> logits
    bf, relu = jnp.bfloat16, jax.nn.relu
    conv = lambda x, q: jax.lax.conv_general_dilated(x, q["kernel"].astype(bf), (1, 1), "SAME", dimension_numbers=(
        "NHWC", "HWIO", "NHWC")) + q["bias"].astype(bf)
    x = (obs / 255.0).astype(bf)
    for b in ("b1", "b2", "b3"):
        x = jax.lax.reduce_window(conv(x, p[b]["conv"]), -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME")
        for r in ("r1", "r2"):
            x = x + conv(relu(conv(relu(x), p[b][r + "a"])), p[b][r + "b"])
    e = relu(_dense(relu(x).reshape(x.shape[0], -1), p["fc"], bf)).astype(jnp.float32)
    x, g = jnp.concatenate([e, jax.nn.one_hot(pa, A)], -1), p["gru"]
    z, r = (jax.nn.sigmoid(_dense(jnp.concatenate([x, h], -1), g[k])) for k in ("wz", "wr"))
    h = (1 - z) * h + z * jnp.tanh(_dense(jnp.concatenate([x, r * h], -1), g["wh"]))
    f = relu(_dense(jnp.concatenate([e, h], -1), p["l1"]))
    if "goal" not in p:
        return h, _dense(f, p["actor"])
    return h, jax.nn.log_softmax(_dense(f, p["actor"])) + lead[:, None] * goal_head(p["goal"], f, s.player_position,
                                                                                     target, lead)


def symbolic(p, h, s, obs, pa, lead, target):  # Dense 512 -> GRU 512 -> MLP 512 -> logits, on the symbolic observation
    g = p["ScannedRNN_0"]["GRUCell_1"]
    hd = lambda x, k: x @ g[k]["kernel"] + g[k].get("bias", 0.0)
    x = jax.nn.relu(_dense(jax.vmap(render_craftax_symbolic)(s), p["Dense_0"]))
    r, z = (jax.nn.sigmoid(hd(x, "i" + k) + hd(h, "h" + k)) for k in "rz")
    h = (1 - z) * jnp.tanh(hd(x, "in") + r * hd(h, "hn")) + z * h
    return h, _dense(jax.nn.relu(_dense(jax.nn.relu(_dense(h, p["Dense_1"])), p["Dense_2"])), p["Dense_3"])


def load_policies(folder):
    "-> [(hidden size, step)], [params] of POLICIES."
    out = []
    for _, f, _ in POLICIES:
        d = pickle.load(open(Path(folder) / f, "rb"))
        st = d.get("state", {})
        out.append((512, symbolic, _plain(d["params"]["params"])) if "params" in d else (
            256, pixel, dict(_plain(st["ac"]), **({"goal": _plain(st["goal"])} if "goal" in st else {}))))
    return [p[:2] for p in out], [p[2] for p in out]


# --- the domain ------------------------------------------------------------------------------------------------------
def _nbr(x, fill, op):
    p = jnp.pad(x, 1, constant_values=fill)
    return op(op(p[:-2, 1:-1], p[2:, 1:-1]), op(p[1:-1, :-2], p[1:-1, 2:]))


def _near(mobs, p, r): return (mobs.mask & (jnp.abs(mobs.position - p).sum(-1) <= r)).any()


def facing(s):
    f = jnp.clip(s.player_position + D4[jnp.clip(s.player_direction - 1, 0, 3)], 0, 63)
    mob = lambda m: (m.mask & (m.position == f).all(-1)).any()
    return s.map[f[0], f[1]], mob(s.zombies) | mob(s.skeletons), mob(s.cows)


def plan(s, target):
    "Distance field to target (int16, INF where unreachable) over walkable tiles; the player's own tile counts."
    lava = s.map == BT.LAVA.value
    w = jnp.isin(s.map, WALK) & ~s.mob_map.astype(bool) & ~(lava | _nbr(lava, False, jnp.logical_or))
    w = w.at[s.player_position[0], s.player_position[1]].set(True)
    d0 = jnp.full(w.shape, INF, jnp.int32).at[target[0], target[1]].set(0)

    def body(c):
        nd = jnp.where(w, jnp.minimum(c[0], _nbr(c[0], INF, jnp.minimum) + 1), c[0])
        return nd, (nd != c[0]).any(), c[2] + 1
    d = jax.lax.while_loop(lambda c: c[1] & (c[2] < 256), body, (d0, jnp.array(True), 0))[0]
    return dict(field=d.astype(jnp.int16), target=target.astype(jnp.int32))


def plan_step(s, pl, key):
    f, p = pl["field"].astype(jnp.int32), s.player_position
    nb = jnp.clip(p[None] + D4, 0, 63)
    dn = jnp.where(((p[None] + D4 >= 0) & (p[None] + D4 < 64)).all(-1), f[nb[:, 0], nb[:, 1]], INF)
    k1, k2 = jax.random.split(key)
    face = jnp.clip(s.player_direction - 1, 0, 3)
    keep = (dn[face] == dn.min()) & jax.random.bernoulli(k1, 0.7)          # keep the facing direction on a tie
    a = jnp.where(keep, face, jnp.argmax(jnp.where(dn == dn.min(), jax.random.gumbel(k2, (4,)), -jnp.inf))) + 1
    at = (p == pl["target"]).all()
    return a.astype(jnp.int32), jnp.where(at, 0, jnp.where(dn.min() < INF, dn.min() + 1, INF)), at | (dn.min() < INF)


def plan_sample(pl, lo, hi, key):
    f = pl["field"].astype(jnp.int32)
    m = (f >= lo) & (f <= hi)
    i = jnp.argmax(jnp.where(m, jax.random.gumbel(key, f.shape), -jnp.inf).ravel())
    return jnp.stack([i // 64, i % 64]).astype(jnp.int32), m.any()


def safe(s):
    p, need = s.player_position, jnp.minimum(jnp.minimum(s.player_food, s.player_drink), s.player_energy)
    calm = lambda z, k, a: ~_near(s.zombies, p, z) & ~_near(s.skeletons, p, k) & ~_near(s.arrows, p, a)
    keep = ~s.is_sleeping & calm(2, 4, 2) & (need >= 2) & (s.player_health >= 4)
    return keep & calm(3, 4, 3) & (need >= 3) & (s.player_health >= 6) & (s.light_level >= DAY), keep


def salience(s):
    v = jax.lax.dynamic_slice(s.map, (s.player_position[0] - 3, s.player_position[1] - 4), (7, 9))
    kinds = (v[..., None] == jnp.arange(len(BT))).any((0, 1)).at[BT.GRASS.value].set(False)
    return (1.0 + kinds.sum() + jnp.isin(v, PLACED).sum()).astype(jnp.float32)


def tiers(s):
    "Per tier (wood, stone, iron): pickaxe and sword held, the materials for one / two tools, at the station."
    i = s.inventory
    tools = jnp.stack([i.wood_pickaxe, i.stone_pickaxe, i.iron_pickaxe, i.wood_sword, i.stone_sword, i.iron_sword]) > 0
    need = lambda k: jnp.stack([i.wood >= k, (i.wood >= k) & (i.stone >= k),
                                (i.wood >= k) & (i.stone >= k) & (i.coal >= k) & (i.iron >= k)])
    table, furnace = is_near_block(s, BT.CRAFTING_TABLE.value), is_near_block(s, BT.FURNACE.value)
    return tools[:3], tools[3:], need(1), need(2), jnp.stack([table, table, table & furnace]), table, furnace


def swap_active(s):
    pick, sword = tiers(s)[:2]
    return ~(pick & sword)


def defer(s):
    "(A,) the tier crafts held back (in p_defer of the episodes) until the materials for both tools are held."
    pick, sword, _, two = tiers(s)[:4]
    return jnp.zeros(A, bool).at[11:17].set(jnp.tile(~pick & ~sword & ~two, 2))


def serve(s):
    "(4, A): the actions that would eat, drink, sleep, fight back now."
    blk, mob, cow = facing(s)
    do = lambda c: jnp.zeros(A, bool).at[DO].set(c)
    return jnp.stack([do(cow | (blk == BT.RIPE_PLANT.value)), do(blk == BT.WATER.value),
                      jnp.zeros(A, bool).at[SLEEP].set(True), do(mob)])


def force_p(s, since):
    """(A,) the probability of forcing each action now. since: steps since the last setup press (an iron craft at a
    furnace is forced only when the furnace is old)."""
    i, blk = s.inventory, facing(s)[0]
    pick, sword, one, _, station, table, furnace = tiers(s)
    weak = ((blk == BT.STONE.value) | (blk == BT.COAL.value)) & (i.wood_pickaxe == 0) | (blk == BT.IRON.value) & (
        i.stone_pickaxe == 0) | (blk == BT.DIAMOND.value) & (i.iron_pickaxe == 0)
    iron, grass, solid = one[2], blk == BT.GRASS.value, ~jnp.isin(blk, WALK)
    p = jnp.zeros(A).at[DO].set(0.01 * weak + 0.02 * weak * (blk >= BT.IRON.value) + 0.05 * grass)  # weak, sapling
    p = p.at[11:17].set(0.01 * table)                                                    # crafts that work or fail
    p = p.at[jnp.asarray([13, 16])].add(table * iron * (0.05 * ~furnace + 0.1 * furnace * (since >= 32)))  # iron
    tools = jnp.concatenate([pick, sword])
    p = p.at[11:17].add(0.5 * jnp.tile(station & one, 2) * ~tools * jnp.roll(tools, 3))  # the tier's second tool
    p = p.at[P_PLANT].set(0.05 * grass * (i.sapling > 0)).at[P_STONE].set(0.005 * (i.stone > 0))
    p = p.at[P_TABLE].set(0.02 * table * (i.wood >= 2) + 0.003 * (i.wood < 2))           # a second table / no wood
    p = p.at[P_FURNACE].set(0.02 * furnace * (i.stone >= 1) + 0.003 * ~table +           # a second one, no table,
                            0.05 * table * iron * ~furnace * solid +                      # a press that fails,
                            0.02 * table * ~furnace * (i.stone >= 3) * ~solid)            # a first one, stone left
    p = p.at[SLEEP].set(0.001 * (s.light_level > 0.6) * (s.player_energy < 9) * ~s.is_sleeping)  # sleep by day
    o, v = jnp.arange(-6, 7), jax.lax.dynamic_slice(jnp.pad(s.map, 6), s.player_position, (13, 13)) == BT.LAVA.value
    to = jnp.where(v, jnp.abs(o[None, :, None] - D4[:, :1, None]) + jnp.abs(o[None, None] - D4[:, 1:, None]), 99)
    to, here = to.min((1, 2)), jnp.where(v, jnp.abs(o[:, None]) + jnp.abs(o), 99).min()
    return p.at[1:5].set(0.02 * (to < here) * (here <= 5) + 0.015 * (to == 0))          # walk to lava, step in


def cause(s, s2):
    "Why an episode ended on s -> s2 (index into CAUSES)."
    p, q = s2.player_position, s.player_position
    return jnp.select([s2.map[p[0], p[1]] == BT.LAVA.value, _near(s.zombies, q, 1),
                       _near(s.skeletons, q, 4) | _near(s.arrows, q, 2), s.player_food == 0, s.player_drink == 0,
                       s.player_energy == 0, s2.player_health <= 0], jnp.arange(1, 8), 8).astype(jnp.int8)


def labels(s):
    r, c = s.player_position                    # tiles: the 7 x 9 view as block type - 2, cow, zombie, skeleton, player
    v = jax.lax.dynamic_slice(jnp.pad(s.map, 4, constant_values=BT.STONE.value), (r + 1, c), (7, 9)) - 2
    for k, m in enumerate((s.cows, s.zombies, s.skeletons)):
        q = m.position - s.player_position + jnp.asarray([3, 4])
        ok = m.mask & (q >= 0).all(-1) & (q < jnp.asarray([7, 9])).all(-1)
        v = v.at[jnp.where(ok, q[:, 0], 7), q[:, 1]].set(15 + k, mode="drop")
    blk, mob, cow = facing(s)
    vit = (s.player_health, s.player_food, s.player_drink, s.player_energy)
    return dict(pos=s.player_position.astype(jnp.int8), dir=s.player_direction.astype(jnp.int8),
                tiles=v.at[3, 4].set(jnp.where(s.is_sleeping, 22, 17 + s.player_direction)).astype(jnp.int8),
                hud=jnp.stack(vit + tuple(getattr(s.inventory, k) for k in HUD[4:])).astype(jnp.int8),
                ach=(s.achievements.astype(jnp.uint32) << jnp.arange(22, dtype=jnp.uint32)).sum(),
                light=s.light_level.astype(jnp.float16), sleeping=s.is_sleeping, facing=blk.astype(jnp.int8),
                facing_mob=mob, facing_cow=cow, table=is_near_block(s, BT.CRAFTING_TABLE.value),
                furnace=is_near_block(s, BT.FURNACE.value), zombie=_near(s.zombies, s.player_position, 3),
                skeleton=_near(s.skeletons, s.player_position, 4), step=s.timestep.astype(jnp.int16))


def frame(s):
    r = render_craftax_pixels(s, BLOCK_PIXEL_SIZE_AGENT)
    return jnp.clip(r, 0, 255).astype(jnp.uint8), jnp.clip(jnp.rint(r), 0, 255).astype(jnp.uint8)   # data, policy input


BEHAVE = core.Behave(n_actions=A, w_policy=tuple(p[2] for p in POLICIES), setup=(P_TABLE, P_FURNACE),
                     swaps=((11, 14), (12, 15), (13, 16)), forbid=(12, 13, 14, 15, 16), follow=((), (13, 16)),
                     hot_ban=(SLEEP,), n_needs=4)
TRIPS = core.Trips(point=lambda s: s.player_position.astype(jnp.int32), safe=safe, salience=salience, plan=plan,
                   step=plan_step, sample=plan_sample,
                   dist=lambda pl, q: pl["field"][q[:, 0], q[:, 1]].astype(jnp.int32))


# --- the collector ---------------------------------------------------------------------------------------------------
def collect(out, seed, ne, nt, policies, params, states=False, block=50):
    "A pool of ne worlds x nt steps from root seed `seed` (keys: reset, behaviour, unused, step) -> out/."
    out.mkdir(parents=True, exist_ok=True)
    kr, kb, _, ke = jax.random.split(jax.random.PRNGKey(seed), 4)
    ids = jnp.arange(ne)
    reset = lambda e, n: env.reset(jax.random.fold_in(jax.random.fold_in(kr, e), n), EP)[1]
    step_keys = lambda e, t: jax.random.fold_in(jax.random.fold_in(ke, e), t)
    es = jax.vmap(reset)(ids, ids * 0)
    fr, obs = jax.vmap(frame)(es)
    c = dict(es=es, obs=obs, h=[jnp.zeros((ne, d)) for d, _ in policies], pa=ids * 0, fresh=ids >= 0, ep=ids * 0,
             beh=BEHAVE.init(ne, kb), trip=TRIPS.init(es))
    c = jax.tree.map(lambda x: jnp.array(x, x.dtype), c)                    # no weak types: one compilation

    @jax.jit
    def run(c, t0, params):
        def one(c, t):
            k, es = jax.random.split(jax.random.fold_in(kb, t), 6), c["es"]
            lead, target = TRIPS.leading(c["trip"])
            hs, lg = zip(*[step(p, jnp.where(c["fresh"][:, None], 0.0, h), es, c["obs"], jnp.where(c["fresh"], 0, c[
                "pa"]), lead, target) for (_, step), p, h in zip(policies, params, c["h"])])
            lg = jnp.stack(lg)
            a, brec = BEHAVE.act(c["beh"], lg, k[0], jax.vmap(swap_active)(es), jax.vmap(serve)(es),
                                 jax.vmap(force_p)(es, c["beh"]["since"]), jax.vmap(safe)(es)[1], jax.vmap(defer)(es))
            drv = jnp.take_along_axis(lg, c["beh"]["drv"][None, :, None], 0)[0]
            a, planner, rrec = TRIPS.act(c["trip"], es, a, drv, k[1], jax.random.categorical(k[5], lg[0]))
            _, es2, r, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.vmap(step_keys, (0, None))(ids, t),
                                                                                es, a, EP)
            why = jnp.where(done, jax.vmap(cause)(es, es2), 0).astype(jnp.int8)
            setup = ((a == P_TABLE) & (es2.inventory.wood < es.inventory.wood)) | (
                (a == P_FURNACE) & (es2.inventory.stone < es.inventory.stone))
            beh = BEHAVE.update(c["beh"], a, setup & ~done, brec["bmode"] == core.FORCED, k[2])
            beh = BEHAVE.new_episode(core._sel(c["trip"]["mode"] > 0, c["beh"], beh), k[3], done)  # frozen in a trip
            ep = c["ep"] + done
            es2 = core._sel(done, jax.vmap(reset)(ids, ep), es2)
            allow = ((beh["mode"] == core.NATIVE) | (beh["mode"] == core.RECOVER)) & (beh["rep_a"] < 0)
            trip = TRIPS.update(c["trip"], es2, a, planner, done, allow, setup & ~done, k[4])     # setup: walk away
            fr, obs = jax.vmap(frame)(es2)
            neglect = ((c["beh"]["need"] > 0) * (1 << jnp.arange(4))).sum(-1).astype(jnp.int8)
            rec = dict(frame=fr, act=a.astype(jnp.int8), rew=r.astype(jnp.float16), done=done, cause=why,
                       rend=trip["end"].astype(jnp.int8), neglect=neglect, **brec, **rrec, **jax.vmap(labels)(es2))
            if states:
                rec["state"] = es.replace(map=es.map.astype(jnp.uint8))
            return dict(es=es2, obs=obs, h=list(hs), pa=a, fresh=done, ep=ep, beh=beh, trip=trip), rec
        return jax.lax.scan(one, c, t0 + jnp.arange(block))

    frames = np.lib.format.open_memmap(out / "frames.npy", "w+", np.uint8, (nt + 1, ne, 63, 63, 3))
    frames[0] = np.asarray(fr)
    L, R, S, t00 = {k: [np.asarray(v)[None]] for k, v in jax.vmap(labels)(es).items()}, {}, [], time.time()
    for t0 in range(0, nt, block):
        c, rec = run(c, t0, params)
        rec = jax.device_get(rec)
        frames[t0 + 1:t0 + 1 + block] = rec.pop("frame")
        S += [jax.tree.leaves(rec.pop("state"))] if states else []
        for k, v in rec.items():
            (L if k in L else R).setdefault(k, []).append(v)
        print(f"step {t0 + block}/{nt} {time.time() - t00:.0f}s episodes {int(np.asarray(c['ep']).sum())}", flush=True)
    frames.flush()
    np.savez(out / "labels.npz", **{k: np.concatenate(v) for k, v in L.items()})
    R = {k: np.concatenate(v) for k, v in R.items()}
    R["done"], R["timeout"] = R["done"] & (R["cause"] != 8), R["done"] & (R["cause"] == 8)
    np.savez(out / "records.npz", **R)
    if states:
        S, like, idx = [np.concatenate(v) for v in zip(*S)], jax.tree.leaves(es), np.argwhere(R["done"] | R["timeout"])
        s = jax.tree.unflatten(jax.tree.structure(es), [jnp.asarray(x[tuple(idx.T)], y.dtype) for x, y in zip(S, like)])
        acts = jnp.asarray(R["act"][tuple(idx.T)], jnp.int32)
        _, term, _, ok, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.vmap(step_keys)(idx[:, 1], idx[:, 0]), s,
                                                                         acts, EP)
        assert bool(ok.all()), "terminal states do not reproduce"
        meta = dict(seed=seed, keys="root PRNGKey(seed) split in 4: reset (fold_in world, episode), behaviour, unused, "
                    "step (fold_in world, step)", frames="astype truncation of the clipped render",
                    policies=[p[1] for p in POLICIES])
        np.savez(out / "states.npz", **{f"s.{i}": x for i, x in enumerate(S)}, term_idx=idx.astype(np.int32),
                 **{f"term.{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(term.replace(map=term.map.astype(
                     jnp.uint8))))}, actions=R["act"].astype(np.uint8), rewards=R["rew"].astype(np.float32),
                 dones=R["done"] | R["timeout"], mode=R["bmode"].astype(np.uint8), trip=R["rmode"].astype(np.uint8),
                 meta=json.dumps(meta))
    (out / "meta.json").write_text(json.dumps(dict(seed=seed, worlds=ne, steps=nt, policies=POLICIES, behave=vars(
        BEHAVE), trips={k: v for k, v in vars(TRIPS).items() if not callable(v)}), indent=1, default=str))


# --- events, busy and quiet transitions, packing ---------------------------------------------------------------------
def since(x, bnd, reverse=False):
    "Steps since (or, reversed, until) x was last true inside the episode, per transition (T, E); 0 on x, else >= 1."
    out, s, big = np.empty(x.shape, np.int32), np.full(x.shape[1], 1 << 20), 1 << 20
    for t in (range(len(x) - 1, -1, -1) if reverse else range(len(x))):
        s = np.where(x[t], 0, (np.where(bnd[t], big, s) if reverse else s) + 1)
        out[t] = s
        s = s if reverse else np.where(bnd[t], big, s)
    return out


def events(L, R):
    "{name: (T, E) bool}: the event types the packer balances."
    a, bnd = R["act"].astype(int), R["done"] | R["timeout"]
    hud, ach, lt = L["hud"].astype(int), L["ach"].astype(np.int64), L["light"].astype(np.float32)
    was, h = hud[:-1], {k: i for i, k in enumerate(HUD)}
    up, dn = (hud[1:] > was) & ~bnd[..., None], (hud[1:] < was) & ~bnd[..., None]
    table, furnace, face = L["table"][:-1], L["furnace"][:-1], L["facing"][:-1]
    ev = {f"unlock_{k}": ((ach[1:] & ~ach[:-1]) >> i & 1).astype(bool) & ~bnd for i, k in enumerate(ACH)}
    ev |= {f"more_{k}": up[..., h[k]] for k in HUD[4:]}
    for act, k, name in ((P_STONE, "stone", "stone"), (P_TABLE, "wood", "table"), (P_FURNACE, "stone", "furnace"),
                         (P_PLANT, "sapling", "plant")):
        ok = (a == act) & dn[..., h[k]]
        ev[f"place_{name}"], ev[f"place_{name}_failed"] = ok, (a == act) & ~ok & ~bnd
    ev["second_table"], ev["second_furnace"] = ev["place_table"] & table, ev["place_furnace"] & furnace
    ev["craft_failed"] = (a >= 11) & ~up[..., h["wood_pickaxe"]:].any(-1) & table & ~bnd
    has_iron = (was[..., [h["wood"], h["stone"], h["coal"], h["iron"]]] >= 1).all(-1)
    ev["iron_press_no_furnace"] = np.isin(a, (13, 16)) & table & ~furnace & has_iron & ~bnd
    ev["iron_press_spell"] = ev["iron_press_no_furnace"] & (since(a == P_FURNACE, bnd) <= 10)
    ev["iron_at_old_furnace"] = (up[..., h["iron_pickaxe"]] | up[..., h["iron_sword"]]) & (
        since(a == P_FURNACE, bnd) > 31)
    ev["furnace_then_no_iron_press"] = ev["place_furnace"] & (since(np.isin(a, (13, 16)), bnd, True) > 31)
    for tier in ("wood", "stone", "iron"):
        p, s = h[tier + "_pickaxe"], h[tier + "_sword"]
        ev[f"{tier}_sword_first"], ev[f"{tier}_pickaxe_first"] = up[..., s] & (was[..., p] == 0), up[..., p] & (
            was[..., s] == 0)
        ev[f"{tier}_sword_after_pickaxe"] = up[..., s] & (was[..., p] > 0) & (was[..., s] == 0)
    weak = np.select([np.isin(face, (BT.STONE.value, BT.COAL.value)), face == BT.IRON.value, face == BT.DIAMOND.value],
                     [was[..., h[k + "_pickaxe"]] == 0 for k in ("wood", "stone", "iron")], False)
    ev["mine_weak_tool"] = (a == DO) & weak & ~bnd
    ev["eat"], ev["drink"] = up[..., h["food"]], up[..., h["drink"]]
    ev["damage"] = dn[..., h["health"]] & (L["zombie"][:-1] | L["skeleton"][:-1])
    ev["sleep_by_day"] = (a == SLEEP) & (lt[:-1] > 0.6) & ~L["sleeping"][:-1] & ~bnd
    ev["dusk"], ev["dawn"] = (lt[:-1] >= 0.5) & (lt[1:] < 0.5) & ~bnd, (lt[:-1] < 0.5) & (lt[1:] >= 0.5) & ~bnd
    ev |= {f"death_{k}": R["done"] & (R["cause"] == i) for i, k in enumerate(CAUSES[1:8], 1)}
    ev["death_lava_walk"] = ev["death_lava"] & (R["bmode"] != core.FORCED)          # not a single forced step
    ev |= {f"return_scale{k}": (R["rend"] == core.REACHED) & (R["rscale"] == k) for k in range(len(TRIPS.scales))}
    return ev


def pack_pools(pools, t_out=2000, e_out=250, cap=400, diamonds=450, dia_share=0.5):
    """Cut the pools into pieces; take the pieces with diamonds first (most first, until they hold `diamonds` or
    dia_share of the set), then balance the events (targets min(supply, cap or DEATHS), weights PRIORITY); pack.
    -> src (t_out + 1, e_out, 3): (pool, world, frame) of every frame; done, truncated; event supply, picked."""
    segs, counts = [], []
    for pi, d in enumerate(pools):
        L, R = dict(np.load(d / "labels.npz")), dict(np.load(d / "records.npz"))
        ev, act, bnd, hud = events(L, R), R["act"].astype(int), R["done"] | R["timeout"], L["hud"]
        names = sorted(ev)
        quiet = (R["rew"] == 0) & ~(hud[1:] != hud[:-1]).any(-1) & (R["rmode"] == 0) & ~bnd
        use = (act == DO) & (~np.isin(L["facing"][:-1], np.asarray(WALK)) | L["facing_mob"][:-1] | L["facing_cow"][:-1])
        busy = np.logical_or.reduce(list(ev.values())) | (act >= SLEEP) | use | (R["rmode"] > 0) | (
            R["bmode"] >= core.REPEAT)
        s = core.pieces(bnd, R["done"], busy, quiet)
        cs = np.concatenate([np.zeros((1,) + bnd.shape[1:] + (len(names),), np.int32), np.cumsum(np.stack(
            [ev[k] for k in names], -1), 0, dtype=np.int32)])
        counts += [cs[z + 1, e] - cs[a, e] for e, a, z, _ in s]
        segs += [(pi * 10 ** 6 + e, a, z, dd) for e, a, z, dd in s]
    counts, n = np.asarray(counts), np.asarray([z - a + 1 for _, a, z, _ in segs])
    dia = counts[:, names.index("more_diamond")]
    order, first = [i for i in np.lexsort((-n, -dia)) if dia[i] > 0], np.zeros(len(segs), bool)
    first[order[:min(np.searchsorted(np.cumsum(n[order]), dia_share * t_out * e_out),
                     np.searchsorted(np.cumsum(dia[order]), diamonds) + 1)]] = True
    target = np.minimum(counts.sum(0), [DEATHS.get(k, cap) for k in names])
    picked, got = core.choose(segs, counts, 0.98 * (t_out + 1) * e_out, target, [PRIORITY.get(k, 1.0) for k in names],
                              first)
    src, done, trunc = core.pack(segs, picked, list(np.argsort(-(counts > 0).sum(1) / n)), t_out, e_out)
    src = np.stack([src[..., 0] // 10 ** 6, src[..., 0] % 10 ** 6, src[..., 1]], -1)
    return src, done, trunc, dict(zip(names, counts.sum(0).tolist())), dict(zip(names, got.tolist()))


def write_set(pools, out):
    "train.npz in the benchmark layout and train_labels.npz (side labels, not published) from the packed pools."
    src, done, trunc, supply, got = pack_pools(pools)
    frames, lab, rec = np.zeros(src.shape[:2] + (63, 63, 3), np.uint8), {}, {}
    for pi, d in enumerate(pools):
        m = src[..., 0] == pi
        frames[m] = np.load(d / "frames.npy", mmap_mode="r")[src[..., 2][m], src[..., 1][m]]
        for k, v in np.load(d / "labels.npz").items():
            lab.setdefault(k, np.zeros(src.shape[:2] + v.shape[2:], v.dtype))[m] = v[src[..., 2][m], src[..., 1][m]]
        for k, v in np.load(d / "records.npz").items():
            mt, st = m[:-1], src[:-1]
            rec.setdefault(k, np.zeros(done.shape + v.shape[2:], v.dtype))[mt] = v[st[..., 2][mt], st[..., 1][mt]]
    meta = dict(env="Craftax-Classic-Pixels-v1", split="train", transitions=int(done.size), envs=done.shape[1],
                steps=done.shape[0], num_actions=A, alignment="actions[t, e] and rewards[t, e] describe frames[t, e] "
                "-> frames[t + 1, e]; a done or truncated transition stores the next piece's first frame.",
                boundaries="dones marks deaths only; truncated marks pieces cut at quiet transitions (no reward, no "
                "inventory or vitals change, not in a return trip) or ended by the time limit.",
                policy="; ".join(f"{w:.0%} {p}" for p, _, w in POLICIES) + " (per episode; switches inside them)",
                recipe="collect/collect.py: behaviour schedule, return trips (half walked by the learned walker), "
                "event-balanced pieces of 250-1000 transitions", pools=[p.name for p in pools], events_supply=supply,
                events_picked=got)
    np.savez(out / "train.npz", frames=frames, actions=rec["act"].astype(np.int16), dones=done, truncated=trunc,
             rewards=rec["rew"].astype(np.float32), meta=json.dumps(meta))
    np.savez(out / "train_labels.npz", src=src, **lab, **{k: v for k, v in rec.items() if k != "done"}, done=done,
             trunc=trunc)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pool", "pack"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--states", action="store_true", help="also save the full state of every step (held-out pools)")
    ap.add_argument("--policies", default="policies", help="the folder with the POLICIES checkpoints")
    ap.add_argument("--pools", type=Path, nargs="*")
    a = ap.parse_args()
    if a.cmd == "pool":
        collect(a.out, a.seed, a.worlds, a.steps, *load_policies(a.policies), a.states)
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(write_set(a.pools, a.out), indent=1))
