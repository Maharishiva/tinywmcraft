"""The domain-free parts of the collector (collect.py), JAX and batched over worlds, plus a numpy packer.

Behave: on top of the drawn driver policy (w_policy per episode; another one takes over for U[switch] steps at hazard
1 / gap_switch), each episode draws an intensity c (calm 0.2 or chaos 1) that scales the hazards below.
  hot       at hazard c / gap_hot: U[hot] steps at temperature U[tau_hot] (or with p_uniform U[uniform] uniform
            steps, never a hot_ban action), then U[rec] recovery steps at U[tau_rec]; paused in danger.
  setup     a setup action that worked starts with p_detour * c a detour (U[detour] hot steps, the setups banned for
            U[cooldown]); else with p_repeat * c it is repeated U[rep_delay] steps later; in p_hold of the episodes
            its follow-up actions are banned for U[hold] steps (so the setup is used later, from an old place).
  breakers  per episode: swap pairs swapped with p_swap while the domain says they matter; forbid actions banned
            with p_forbid * c for U[ban] steps; at hazard c / gap_neglect a need is neglected for U[neglect] steps.
  forced    the domain's force_p proposes attempts from the state; one is taken with probability c * sum, at most
            once per U[force_gap] steps.
Trips: in native play, at hazard 1 / gap when safe (or when the caller asks, `now`, with p_now: an out-and-back of
scale >= now_scale), a trip draws a scale [lo, hi] of planner distances and a tour of waypoints: A the start, R a
remembered point (age max(min_age, lo) to age_span * hi, by salience), F a point at that distance, G one at half of
it; e.g. "R" back to an old place, "FA" out and back, "FGAG" A-B-C-A-C. The planner, or a learned walker for p_lead
of the trips, walks the legs (p_agent: the policy's own action); a waypoint may get a pause (p_pause, U[pause], hot
with p_hot), the end U[dwell] native steps. Danger yields to the policy. A trip is aborted at an episode end, danger
over yield_max, no path (a walker replans), no progress for `stuck` steps, or a leg over slack x distance + extra.
"""
from dataclasses import dataclass
from typing import Callable
import jax, jax.numpy as jnp, numpy as np

NATIVE, HOT, UNIFORM, RECOVER, REPEAT, FORCED = range(6)                    # behaviour record modes
WALK, PAUSE, DWELL = 1, 2, 3                                                # trip modes (0: no trip)
REACHED, TIMEOUT, NOPATH, STUCK, DANGER, EPEND = range(1, 7)
NEVER = -(1 << 30)


def _ri(k, n, lohi): return jax.random.randint(k, (n,), lohi[0], lohi[1] + 1)
def _u(k, n, lohi): return jax.random.uniform(k, (n,), minval=lohi[0], maxval=lohi[1])
def _sel(m, a, b): return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)
def _dec(x): return jnp.maximum(x - 1, 0)


@dataclass(frozen=True)
class Behave:
    n_actions: int
    w_policy: tuple = (1.0,)                                                 # the domain's lists
    setup: tuple = (); swaps: tuple = (); forbid: tuple = (); follow: tuple = (); hot_ban: tuple = (); n_needs: int = 0
    gap_switch: float = 800.0; switch: tuple = (50, 400); intensity: tuple = (0.2, 1.0)
    gap_hot: float = 200.0; hot: tuple = (5, 30); tau_hot: tuple = (2.0, 8.0); p_uniform: float = 0.35
    uniform: tuple = (3, 24); rec: tuple = (20, 80); tau_rec: tuple = (1.0, 1.5)
    p_detour: float = 0.4; detour: tuple = (25, 60); cooldown: tuple = (60, 250); p_repeat: float = 0.6
    rep_delay: tuple = (1, 40); p_hold: float = 0.75; hold: tuple = (32, 96); hold_self: tuple = (96, 256)
    p_defer: float = 0.5; p_swap: float = 0.5; p_forbid: float = 0.06; ban: tuple = (100, 800)
    gap_neglect: float = 16000.0; neglect: tuple = (300, 1500); force_gap: tuple = (4, 16)

    def init(self, n, key):
        z, zs = jnp.zeros((n,), jnp.int32), lambda m, dt=jnp.int32: jnp.zeros((n, m), dt)
        st = dict(mode=z, left=z, tau=jnp.ones((n,)), c=jnp.ones((n,)), since=z, fwait=z, base=z, drv=z, drv_left=z,
                  cool=z, rep_a=z - 1, rep_left=z, swap=zs(len(self.swaps), bool), ban=zs(len(self.forbid)),
                  holds=z.astype(bool), hold=zs(len(self.setup)), hold_self=zs(len(self.setup)), need=zs(self.n_needs),
                  defer=z.astype(bool))
        return self.new_episode(st, key, jnp.ones((n,), bool))

    def new_episode(self, st, key, start):
        n, k = start.shape[0], jax.random.split(key, 5)
        base = jax.random.categorical(k[0], jnp.log(jnp.asarray(self.w_policy)), shape=(n,))
        c = jnp.asarray(self.intensity)[jax.random.randint(k[4], (n,), 0, len(self.intensity))]
        zero = {key: st[key] * 0 for key in ("mode", "left", "fwait", "drv_left", "cool", "need", "hold", "hold_self")}
        ban = jax.random.bernoulli(k[2], self.p_forbid * c[:, None], st["ban"].shape) * jax.random.randint(
            k[3], st["ban"].shape, self.ban[0], self.ban[1] + 1)
        new = dict(st, **zero, tau=jnp.ones((n,)), c=c, since=st["since"] * 0 + 999, base=base, drv=base,
                   rep_a=st["rep_a"] * 0 - 1, swap=jax.random.bernoulli(k[1], self.p_swap, st["swap"].shape),
                   holds=jax.random.bernoulli(jax.random.fold_in(k[1], 1), self.p_hold, (n,)), ban=ban,
                   defer=jax.random.bernoulli(jax.random.fold_in(k[1], 2), self.p_defer, (n,)))
        return _sel(start, new, st)

    def act(self, st, logits, key, active, serve, force_p, calm, defer):
        """logits (D, n, A) of every policy; active (n, P): the swap pairs that matter; serve (n, K, A): the actions
        that serve each need; force_p (n, A); calm (n,): no danger; defer (n, A): held back in p_defer of episodes."""
        n = logits.shape[1]
        lg = jnp.take_along_axis(logits, st["drv"][None, :, None], 0)[0]
        for j, (a, b) in enumerate(self.swaps):
            on = st["swap"][:, j] & active[:, j]
            lg = lg.at[:, a].set(jnp.where(on, lg[:, b], lg[:, a])).at[:, b].set(jnp.where(on, lg[:, a], lg[:, b]))
        ban = jnp.zeros((n, self.n_actions), bool)
        for j, a in enumerate(self.forbid):
            ban = ban.at[:, a].set(st["ban"][:, j] > 0)
        for a in self.setup:
            ban = ban.at[:, a].set(ban[:, a] | (st["cool"] > 0))
        for j, acts in enumerate(self.follow):
            for a in acts:
                ban = ban.at[:, a].set(ban[:, a] | (st["hold"][:, j] > 0))
        for j, a in enumerate(self.setup):
            ban = ban.at[:, a].set(ban[:, a] | (st["hold_self"][:, j] > 0))
        ban = ban | (defer & st["defer"][:, None])
        if self.n_needs:
            ban = ban | ((st["need"] > 0)[:, :, None] & serve).any(1)
        lg = jnp.where(ban, -1e9, lg)
        hot = calm & (st["mode"] != NATIVE)
        for a in self.hot_ban:
            ban = ban.at[:, a].set(ban[:, a] | hot)
        lg = jnp.where((hot & (st["mode"] == UNIFORM))[:, None], jnp.where(ban, -1e9, 0.0),
                       lg / jnp.where(hot, st["tau"], 1.0)[:, None])
        k = jax.random.split(key, 3)
        a = jax.random.categorical(k[0], lg).astype(jnp.int32)
        rep = (st["rep_a"] >= 0) & (st["rep_left"] == 0)
        fp = jnp.where(ban, 0.0, force_p * st["c"][:, None])
        forced = ~rep & (st["fwait"] == 0) & (jax.random.uniform(k[1], (n,)) < fp.sum(-1))
        a = jnp.where(rep, st["rep_a"], jnp.where(forced, jax.random.categorical(k[2], jnp.log(fp + 1e-12)), a))
        mode = jnp.where(rep, REPEAT, jnp.where(forced, FORCED, st["mode"]))
        return a, dict(bmode=mode.astype(jnp.int8), drv=st["drv"].astype(jnp.int8), tau=st["tau"].astype(jnp.float16),
                       intensity=st["c"].astype(jnp.float16))

    def update(self, st, a, setup_ok, forced, key):
        n, k, c = a.shape[0], jax.random.split(key, 16), st["c"]
        mode, left = st["mode"], st["left"] - 1
        hot_end, rec_end = ((mode == HOT) | (mode == UNIFORM)) & (left <= 0), (mode == RECOVER) & (left <= 0)
        go = (mode == NATIVE) & jax.random.bernoulli(k[0], c / self.gap_hot)
        uni = go & jax.random.bernoulli(k[1], self.p_uniform, (n,))
        trig = setup_ok & (st["rep_a"] < 0) & jax.random.bernoulli(k[2], self.p_detour * c)
        rep = setup_ok & ~trig & (st["rep_a"] < 0) & jax.random.bernoulli(k[3], self.p_repeat * c)
        fired = (st["rep_a"] >= 0) & (st["rep_left"] == 0)
        mode = jnp.select([trig, uni, go, hot_end, rec_end], [HOT, UNIFORM, HOT, RECOVER, NATIVE], mode)
        left = jnp.select([trig, uni, go, hot_end], [_ri(k[4], n, self.detour), _ri(k[5], n, self.uniform),
                                                     _ri(k[6], n, self.hot), _ri(k[7], n, self.rec)], left)
        tau = jnp.select([trig | go, hot_end, rec_end], [_u(k[8], n, self.tau_hot), _u(k[9], n, self.tau_rec), 1.0],
                         st["tau"])
        sw, back = (st["drv_left"] <= 1) | jax.random.bernoulli(k[10], 1 / self.gap_switch, (n,)), st["drv_left"] == 1
        other = jax.random.categorical(k[11], jnp.log(jnp.asarray(self.w_policy)), shape=(n,))
        need = _dec(st["need"])
        start = jax.random.bernoulli(k[13], c[:, None] / self.gap_neglect, need.shape) & (need == 0)
        hold = (setup_ok & st["holds"])[:, None] & (a[:, None] == jnp.asarray(self.setup))
        draw = lambda i, lohi: jax.random.randint(jax.random.fold_in(k[14], i), st["hold"].shape, lohi[0], lohi[1] + 1)
        return dict(st, mode=mode, left=left, tau=tau, drv=jnp.where(sw, jnp.where(back, st["base"], other), st["drv"]),
                    drv_left=jnp.where(sw & ~back, _ri(k[12], n, self.switch), _dec(st["drv_left"])),
                    need=jnp.where(start, jax.random.randint(k[14], need.shape, *self.neglect), need),
                    ban=_dec(st["ban"]), since=jnp.where(jnp.isin(a, jnp.asarray(self.setup)), 0, st["since"] + 1),
                    fwait=jnp.where(forced, _ri(k[15], n, self.force_gap), _dec(st["fwait"])),
                    hold=jnp.where(hold, draw(1, self.hold), _dec(st["hold"])),
                    hold_self=jnp.where(hold, draw(2, self.hold_self), _dec(st["hold_self"])),
                    cool=jnp.where(trig, _ri(k[4], n, self.cooldown), _dec(st["cool"])),
                    rep_a=jnp.where(rep, a, jnp.where(fired, -1, st["rep_a"])),
                    rep_left=jnp.where(rep, _ri(k[5], n, self.rep_delay), _dec(st["rep_left"])))


@dataclass(frozen=True)
class Trips:
    point: Callable                       # s -> (P,) int32, where the agent is
    safe: Callable                        # s -> (may a trip start, may it go on)
    salience: Callable                    # s -> float, how distinctive the view is
    plan: Callable                        # (s, target) -> plan (a pytree), made at each leg start and on a replan
    step: Callable                        # (s, plan, key) -> (action, distance left, a path exists)
    dist: Callable                        # (plan, points (M, P)) -> (M,) planner distance from the plan's target
    sample: Callable                      # (plan, lo, hi, key) -> (a point at a distance in [lo, hi], found)
    P: int = 2; M: int = 256; gap: float = 30.0
    scales: tuple = ((2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127)); w_scale: tuple = (1, 2, 2, 0.6, 0.3, 0.15)
    tours: tuple = ("R", "FA", "GR", "FGAG", "RFA"); w_tour: tuple = (0.35, 0.3, 0.1, 0.15, 0.1)
    min_age: int = 4; age_span: float = 4.0; p_pause: float = 0.4; pause: tuple = (2, 24); p_hot: float = 0.5
    tau_act: tuple = (3.0, 8.0); dwell: tuple = (2, 12); short: int = 8; p_agent: float = 0.1; slack: float = 2.0
    extra: int = 12; stuck: int = 8; yield_max: int = 24; p_now: float = 0.3; now_scale: int = 2
    p_lead: float = 0.5; lead_slack: float = 3.0; lead_extra: int = 20; lead_stuck: int = 16

    def init(self, s):
        pt = jax.vmap(self.point)(s)
        n, z = pt.shape[0], jnp.zeros((pt.shape[0],), jnp.int32)
        st = dict(mem_pt=jnp.zeros((n, self.M, self.P), jnp.int32), mem_t=jnp.full((n, self.M), NEVER, jnp.int32),
                  mem_sal=jnp.zeros((n, self.M), jnp.float32), head=z, t=z, here=pt, mode=z, tour=z, scale=z, nwp=z,
                  wp=jnp.zeros((n, max(map(len, self.tours)), self.P), jnp.int32), wi=z, left=z, cap=z, budget=z,
                  hot=z.astype(bool), tau=jnp.ones((n,)), plan=jax.vmap(self.plan)(s, pt), legt=z, still=z, wait=z,
                  trip=z, end=z, lead=z.astype(bool), calm=jnp.ones((n,), bool))
        return self._remember(st, pt, jnp.ones((n,), bool), jax.vmap(self.salience)(s))

    def _remember(self, st, pt, start, sal):
        i, t, head = jnp.arange(pt.shape[0]), jnp.where(start, 0, st["t"]), jnp.where(start, 0, st["head"])
        mem_t = jnp.where(start[:, None], NEVER, st["mem_t"]).at[i, head].set(t)
        return dict(st, mem_pt=st["mem_pt"].at[i, head].set(pt), mem_t=mem_t, head=(head + 1) % self.M, t=t, here=pt,
                    mem_sal=st["mem_sal"].at[i, head].set(sal))

    def _target(self, st, wi):
        return jnp.take_along_axis(st["wp"], wi[:, None, None], 1)[:, 0]

    def leading(self, st):
        "Where a learned walker should walk now, and to which point."
        return (st["mode"] == WALK) & st["lead"] & st["calm"], self._target(st, st["wi"])

    def act(self, st, s, a_pol, logits, key, a_lead=None):
        "-> action (n,), from the planner?, record. a_pol: the behaviour's action; a_lead: the walker's action."
        n, k = a_pol.shape[0], jax.random.split(key, 3)
        a_plan, dist, ok = jax.vmap(self.step)(s, st["plan"], jax.random.split(k[0], n))
        a_plan = a_plan if a_lead is None else jnp.where(st["lead"], a_lead, a_plan)
        planner = (st["mode"] == WALK) & st["calm"] & ok & (dist > 0) & ~jax.random.bernoulli(k[1], self.p_agent, (n,))
        hot = (st["mode"] == PAUSE) & st["hot"] & st["calm"]
        a = jnp.where(planner, a_plan, jnp.where(hot, jax.random.categorical(k[2], logits / st["tau"][:, None]), a_pol))
        on = st["mode"] > 0
        return a.astype(jnp.int32), planner, dict(
            rmode=st["mode"].astype(jnp.int8), rtour=jnp.where(on, st["tour"], -1).astype(jnp.int8),
            rscale=jnp.where(on, st["scale"], -1).astype(jnp.int8), rtrip=st["trip"], rwi=st["wi"].astype(jnp.int8),
            rplanner=planner, rlead=st["lead"] & on, rtarget=self._target(st, st["wi"]).astype(jnp.int16))

    def update(self, st, s, a, planner, done, allow, now, key):
        """After action a took the worlds to s (done: s is the next episode's first state); allow: may a trip start;
        now: the caller asks for one. st["end"] says how a trip ended on this transition."""
        n, k = a.shape[0], jax.random.split(key, 16)
        bern = lambda i, p: jax.random.bernoulli(k[i], p, (n,))
        pt = jax.vmap(self.point)(s)
        moved = (pt != st["here"]).any(-1) & ~done
        st = self._remember(dict(st, t=st["t"] + 1), pt, done, jax.vmap(self.salience)(s))
        start_ok, keep_ok = jax.vmap(self.safe)(s)
        mode, walk = st["mode"], st["mode"] == WALK
        _, dist, ok = jax.vmap(self.step)(s, st["plan"], jax.random.split(k[0], n))
        at, last, lost = walk & ok & (dist == 0), st["wi"] >= st["nwp"] - 1, walk & ~ok
        still = jnp.where(walk & planner & ~moved, st["still"] + 1, jnp.where(moved, 0, st["still"]))
        wait = jnp.where((walk | (mode == PAUSE)) & ~keep_ok, st["wait"] + 1, 0)
        legt, left = st["legt"] + (walk & st["calm"]), st["left"] - 1
        stuck = still >= jnp.where(st["lead"], self.lead_stuck, self.stuck)
        end = jnp.select([done & (mode > 0), wait > self.yield_max, lost & ~st["lead"], walk & ~at & stuck,
                          walk & ~at & (legt > st["budget"]), at & last], [EPEND, DANGER, NOPATH, STUCK, TIMEOUT,
                                                                             REACHED], 0)
        pause = at & ~last & bern(1, self.p_pause)                # a waypoint: a pause or the next leg; the end: dwell
        next_leg = (at & ~last & ~pause) | ((mode == PAUSE) & (left <= 0))
        new_mode = jnp.where(end == REACHED, DWELL, jnp.where(pause, PAUSE, mode))
        new_mode = jnp.where(((mode == DWELL) & ((left <= 0) | done)) | ((end > 0) & (end != REACHED)), 0, new_mode)
        new_left = jnp.where(end == REACHED, _ri(k[2], n, self.dwell), jnp.where(pause, jnp.minimum(
            _ri(k[3], n, self.pause), jnp.maximum(st["cap"], self.pause[0])), left))
        wi = st["wi"] + next_leg
        now = now & bern(13, self.p_now)                          # a new trip: scale, tour, waypoints
        go = (mode == 0) & ~done & start_ok & (allow & bern(4, 1 / self.gap) | now)
        sc = jax.random.categorical(k[5], jnp.log(jnp.asarray(self.w_scale)), shape=(n,))
        sc = jnp.where(now, jnp.maximum(sc, self.now_scale), sc)
        lo, hi = (jnp.asarray([x[j] for x in self.scales])[sc] for j in (0, 1))

        def pick(_):
            ph = jax.vmap(self.plan)(s, pt)
            dm, age = jax.vmap(self.dist)(ph, st["mem_pt"]), st["t"][:, None] - st["mem_t"]
            fits = (st["mem_t"] > NEVER) & (dm >= lo[:, None]) & (dm <= hi[:, None])
            fits &= (age >= jnp.maximum(self.min_age, lo)[:, None]) & (age <= self.age_span * hi[:, None])
            w = jnp.where(fits, jnp.log(jnp.maximum(st["mem_sal"], 1e-6)), -jnp.inf)
            ir = jnp.argmax(w + jax.random.gumbel(k[6], w.shape), -1)
            f, has_f = jax.vmap(self.sample)(ph, lo, hi, jax.random.split(k[7], n))
            g, has_g = jax.vmap(self.sample)(ph, jnp.maximum(lo // 2, 1), jnp.maximum(hi // 2, 1),
                                             jax.random.split(k[8], n))
            found = jnp.stack([jnp.ones((n,), bool), fits.any(-1), has_f, has_g], 1)
            return jnp.stack([pt, st["mem_pt"][jnp.arange(n), ir], f, g], 1), found
        cand, has = jax.lax.cond(go.any(), pick, lambda _: (jnp.stack([pt] * 4, 1), jnp.zeros((n, 4), bool)), None)
        W, tours = st["wp"].shape[1], self.tours
        codes = jnp.asarray([["ARFG".index(c) for c in t] + [0] * (W - len(t)) for t in tours])
        tour = jnp.where(now, tours.index("FA"), jax.random.categorical(k[9], jnp.log(jnp.asarray(self.w_tour)),
                                                                          shape=(n,)))
        fine = (has | ~jnp.asarray([[c in t for c in "ARFG"] for t in tours])[tour]).all(-1)
        go = go & (fine | has[:, 2])
        tour = jnp.where(fine, tour, tours.index("FA"))
        wp = jnp.take_along_axis(cand, codes[tour][:, :, None], 1)
        nwp = jnp.asarray([len(t) for t in tours])[tour]
        g = lambda new, old: jnp.where(go.reshape(go.shape + (1,) * (jnp.ndim(old) - 1)), new, old)
        st = dict(st, tour=g(tour, st["tour"]), scale=g(sc, st["scale"]), wp=g(wp, st["wp"]), nwp=g(nwp, st["nwp"]),
                  lead=g(bern(14, self.p_lead), st["lead"]), trip=st["trip"] + go,
                  hot=g(bern(10, self.p_hot), st["hot"]),
                  cap=g(jnp.maximum(self.short, hi // 2), st["cap"]), tau=g(_u(k[11], n, self.tau_act), st["tau"]))
        wi = jnp.where(go, 0, wi)
        new_mode = jnp.where(go | (next_leg & (new_mode != 0)), WALK, new_mode)
        new_leg = go | (next_leg & (new_mode == WALK))
        need = new_leg | (walk & (new_mode == WALK) & ~new_leg & ((still == self.stuck // 2) | (lost & st["lead"])))
        target = self._target(st, wi)
        plan = jax.lax.cond(need.any(), lambda _: jax.vmap(self.plan)(s, target), lambda _: st["plan"], None)
        plan = _sel(need, plan, st["plan"])
        _, d0, ok0 = jax.vmap(self.step)(s, plan, jax.random.split(k[12], n))
        dead = new_leg & ~ok0
        end, new_mode = jnp.where(dead & (end == 0), NOPATH, end), jnp.where(dead, 0, new_mode)
        budget = jnp.where(st["lead"], jnp.ceil(self.lead_slack * d0).astype(jnp.int32) + self.lead_extra,
                           jnp.ceil(self.slack * d0).astype(jnp.int32) + self.extra)
        return dict(st, mode=new_mode, left=new_left, wi=wi, plan=plan, end=end.astype(jnp.int32),
                    still=jnp.where(new_leg, 0, still), budget=jnp.where(new_leg, budget, st["budget"]),
                    legt=jnp.where(new_leg, 0, legt), wait=jnp.where(new_mode == 0, 0, wait),
                    calm=keep_ok | (new_mode == 0) | (new_mode == DWELL))


def pieces(bnd, death, busy, quiet, idle_max=64, pad=16, seg=(250, 1000), min_len=48, seed=0):
    """Cut every stream (bnd (T, E): an episode ends here; death: by dying) into pieces: stretches with no busy
    transition for over idle_max steps are dropped but for pad steps at each end; the rest is split into pieces of about
    U[seg] transitions that end on a boundary or a quiet transition. -> [(stream, first, last, ends in a death)]."""
    T, E = bnd.shape
    rng, out, ts = np.random.default_rng(seed), [], np.arange(T)
    for e in range(E):
        b, q, y = bnd[:, e], quiet[:, e] & ~bnd[:, e], busy[:, e] | bnd[:, e]
        prev = np.maximum.accumulate(np.where(y, ts, -T))
        nxt = np.minimum.accumulate(np.where(y, ts, 2 * T)[::-1])[::-1]
        keep = y | (nxt - prev - 1 <= idle_max) | (np.minimum(ts - prev, nxt - ts) <= pad)
        s = 0
        for zz in np.flatnonzero(keep & (~np.r_[keep[1:], False] | b)):        # where a kept stretch ends
            while s < zz and not keep[s]:
                s += 1
            ends = np.flatnonzero(q[s:zz + 1]) + s
            z, a = zz if b[zz] or q[zz] else (ends[-1] if len(ends) else -1), s
            while z - a + 1 >= min_len:
                want = a + int(rng.integers(*seg))
                c = ends[(ends >= want) & (ends < z - min_len)]
                end = int(c[0]) if want < z - min_len and len(c) else z
                out.append((e, a, end, bool(death[end, e]) and end == z))
                a = end + 1
            s = zz + 1
    return out


def choose(segs, counts, total, target, weight, first, n0=300, seed=1):
    """Greedy picks that balance events, not frames: the `first` pieces, then each time the piece with the best gain per
    (n0 + its length) (n0 favours long pieces); got_k events of type k are worth weight_k * min(got_k / target_k, 1)
    plus a small log tail past the target. counts (S, K): events per piece."""
    n, tg, w = np.asarray([s[2] - s[1] + 1 for s in segs], float), np.maximum(target, 1.0), np.asarray(weight, float)
    value = lambda x: np.minimum(x / tg, 1.0) + 0.1 * np.log1p(np.maximum(x - tg, 0.0) / tg)
    used = np.asarray(first, bool).copy()
    picked, got, size = list(np.flatnonzero(used)), counts[used].sum(0).astype(float), n[used].sum()
    jitter = np.random.default_rng(seed).uniform(1.0, 1.01, len(segs))
    while size < total:
        gain = (w * (value(got + counts) - value(got))).sum(1) / (n0 + n) * jitter
        gain[used | (size + n > total)] = -1.0
        i = int(np.argmax(gain))
        if gain[i] < 0:
            break
        used[i], size, got = True, size + n[i], got + counts[i]
        picked.append(i)
    return picked, got


def pack(segs, picked, spare, t_out, e_out):
    """First-fit decreasing of the picked pieces into e_out streams of t_out + 1 frames, the free tails filled with the
    starts of spare pieces. -> src (t_out + 1, e_out, 2): (pool stream, frame) of every frame; done, truncated."""
    n = lambda i: segs[i][2] - segs[i][1] + 1
    bins, free = [[] for _ in range(e_out)], [t_out + 1] * e_out
    for i in sorted(picked, key=lambda i: -n(i)):
        j = next((j for j in range(e_out) if free[j] >= n(i)), None)
        if j is not None:
            bins[j].append(i)
            free[j] -= n(i)
    chosen = set(picked)
    spare = [i for i in spare if i not in chosen]
    src, done, trunc = np.zeros((t_out + 1, e_out, 2), np.int64), np.zeros((t_out, e_out), bool), \
        np.zeros((t_out, e_out), bool)
    for j, b in enumerate(bins):
        p = 0
        while p <= t_out:
            e, a, z, d = segs[b.pop(0) if b else spare.pop(0)]
            m = min(z - a + 1, t_out + 1 - p)
            src[p:p + m, j] = np.stack([np.full(m, e), np.arange(a, a + m)], 1)
            if p + m - 1 < t_out:
                done[p + m - 1, j], trunc[p + m - 1, j] = d, not d
            p += m
    return src, done, trunc
