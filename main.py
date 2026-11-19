#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (*.in) from $MAKEFUN_INPUT_DIR (else /input) and writes
one forecast file /output/<id>.out per case: five quantiles (5/25/50/75/95%)
of each trophy's Exophase earn rate at the case's target instant.

Model (written from scratch for this bounty; standard library only, no
network, no model calls, fully deterministic):

  A trophy's earn rate is earners / players-tracked.  Both move, and the
  page only refreshes every few hours, so the history is a staircase of
  refresh snapshots.  The forecaster separates the two moving parts:

  1. Players tracked N(t).  The distinct refresh snapshots give N at a
     handful of times.  log N is fitted as a power law in log(1 + hours
     since release) over the recent snapshots, then extrapolated to the
     target with the growth rate damped only mildly (the recent window
     already reflects the current, slower rate) and the total multiplier
     capped.  Between snapshots log N is interpolated in log(1 + t).

  2. Earners E_i(t).  A player who starts at time s earns trophy i
     after a random play time; the share of starters who have earned it
     by age a is lognormal, F_i(a) = Phi((ln a - ln m_i) / 0.7), with
     m_i the trophy's median time-to-earn (an early activity: under an
     hour; a chapter: a few hours; 100% completion: hundreds of hours).
     Only a share q_i of players ever earn it.  So
         E_i(t) = q_i * sum over arrival events of dN * F_i(t - s),
     and the rate is r_i(t) = q_i * mature_frac_i(t), the
     arrival-weighted mean of F_i over the tracked players' ages.
     Arrival events are the snapshot jumps (the first cohort is split
     over the 2 h before the first read: the game unlocked at local
     midnight across timezones, but a ~100 GB download means the first
     tracked players had only about an hour of play in them -- the
     launch-night rates of ~1% for the first chapter say so).  Future
     arrivals to the target come from the extrapolated N curve on a
     6 h grid.

     Per trophy, m_i is picked by grid search (0.5..256 h) minimising
     the recency-weighted squared error of log rate against
     log q_i + log mature_frac; log q_i is the weighted mean residual.
     Unlike an exponential maturation, the lognormal saturates: once
     the player base is older than a trophy's median time, its rate
     recovers towards q_i as growth slows -- that mean reversion is
     the whole edge over a persistence baseline (which assumes the
     rate stays) and over a log-time trend (which extrapolates the
     early dilution forever).

     q_i is shrunk 20-45% (in log space) towards the median q of its
     class -- platinum / story ("Chapter" or "final mission" in the
     description) / side -- because rare trophies have only a handful
     of earners and their own fit is noisy.  When the history shows no
     growth in N at all, no median is identifiable and a class prior
     median is used instead (story 4 h, side 32 h, platinum 256 h).

     Censored median: when several medians fit equally well and they
     imply target rates more than half a logit apart, the history
     cannot tell them apart (e.g. a trophy nobody has started earning
     yet).  The forecast then takes the log-median of the tied
     medians' target rates and widens the spread by their log-range,
     instead of trusting one arbitrary member of the tie.

  3. Spread.  Quantiles are the median on the logit scale plus normal
     z-scores times a per-trophy logit sd combining a random-walk term
     growing with the horizon, the trophy's own fit residual, a floor
     for model error, and an N-extrapolation term.

  4. No refresh ever seen.  When the history holds a single snapshot,
     no median is identifiable and the page may refresh before the
     target.  The forecast is then a two-component mixture in logit
     space: with probability 1-p no refresh (rate stays, small noise);
     with probability p one refresh lands, multiplying tracked players
     ~6x (launch night's first refresh: 638 -> 3868) while the
     newcomers have earned ~1 trophy each, so every trophy but the
     first keeps its earners and its rate is divided by ~6.  p grows
     with the horizon (mean refresh interval ~4 h).  Quantiles come
     from inverting the mixture CDF by bisection.  A trophy already
     at ~100% stays there.

  5. Edge cases.  Rates of 0 / missing reads are skipped in the fit.
     Everything is clamped to [0.01, 99.99] percent for the logit math.
"""

import glob
import json
import math
import os
from datetime import datetime
from statistics import NormalDist

# Candidate median times-to-earn, in hours: how long it takes a typical
# starter to earn the trophy.  Early activities are under an hour;
# chapters a few hours; long grinds and 100% completion hundreds.
MEDIAN_GRID_HOURS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# Candidate log-spreads of the time-to-earn distribution: a chapter
# takes a few hours for almost everyone (tight), a grind varies widely
# (loose).  The fit picks per trophy; a single fixed spread fits
# neither extreme.
SIGMA_GRID = [0.35, 0.7, 1.2]

# Class prior medians (hours) when the history identifies no median.
PRIOR_MEDIAN_HOURS = {"platinum": 256.0, "story": 4.0, "side": 32.0}

# First-refresh facts, from launch night as the brief records them:
# tracked players 638 -> 3868 (x6.06) while the newcomers had earned
# ~1 trophy each (total trophies 943 -> 4276, i.e. ~1.03 per new player).
FIRST_REFRESH_JUMP = 6.06
REFRESH_MEAN_INTERVAL_H = 4.0

# Spread constants (logit sd).
RW_PER_SQRT_DAY = 0.35      # random-walk widening with horizon
MODEL_FLOOR = 0.20          # irreducible model error
RESID_FLOOR = 0.15          # fit-residual floor (2-decimal rounding alone)
RESID_CAP = 0.90
N_UNCERT_PER_DOUBLING = 0.20
MIX_NO_REFRESH_SD = 0.15    # logit noise when no refresh lands
MIX_REFRESH_SD = 0.35       # logit noise when a refresh lands

NORMAL = NormalDist()


def parse_time(text):
    """ISO 8601 UTC string -> epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_frac(p):
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def inv_logit_frac(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700.0 else 1.0
    e = math.exp(x)
    return e / (1.0 + e)


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def weighted_mean(pairs):
    """pairs: [(value, weight)] -> weighted mean, or None when empty."""
    total_w = 0.0
    total = 0.0
    for value, weight in pairs:
        total += value * weight
        total_w += weight
    if total_w <= 0.0:
        return None
    return total / total_w


def least_squares_slope(xs, ys):
    """Slope of y on x by least squares; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        num += (x - mx) * (y - my)
        den += (x - mx) * (x - mx)
    if den <= 1e-12:
        return 0.0
    return num / den


def trophy_class(trophy):
    """Coarse class from tier + description words (generic, not per-name)."""
    if trophy.get("tier") == "platinum":
        return "platinum"
    desc = (trophy.get("description") or "").lower()
    if "chapter" in desc or "final mission" in desc:
        return "story"
    return "side"


class PlayersCurve:
    """Tracked-players curve N(t) from the refresh snapshots.

    t is hours since the game's release midnight (UTC).  log N is
    interpolated in log1p(t) between snapshots and extrapolated past the
    last snapshot with a mildly damped power law.
    """

    def __init__(self, states, target_h):
        # states: [(t_hours, n_players)] distinct snapshots, time order.
        self.states = states
        self.target_h = target_h
        self.t_last = states[-1][0]
        self.n_last = states[-1][1]
        # Power-law slope from the recent snapshots (last 36 h if enough).
        window = [s for s in states if s[0] >= self.t_last - 36.0]
        if len(window) < 2:
            window = states
        xs = [math.log1p(max(t, 0.0)) for t, _ in window]
        ys = [math.log(n) for _, n in window]
        alpha = least_squares_slope(xs, ys)
        # Launch growth decelerates, but the recent window already
        # reflects the current (slower) rate, so damp only mildly:
        # damping twice would double-count the slowdown.
        self.alpha_future = max(0.0, alpha) * 0.9
        self.u_last = math.log1p(max(self.t_last, 0.0))
        self.log_n_last = math.log(self.n_last)

    def log_n(self, t_h):
        if t_h <= self.states[0][0]:
            return math.log(self.states[0][1])
        if t_h >= self.t_last:
            u = math.log1p(max(t_h, 0.0))
            log_n = self.log_n_last + self.alpha_future * (u - self.u_last)
            # Cap the total multiplier over the last snapshot.
            cap = self.log_n_last + math.log(8.0)
            return min(log_n, cap)
        # Interpolate log N linearly in log1p(t) between bracketing states.
        for (t0, n0), (t1, n1) in zip(self.states, self.states[1:]):
            if t0 <= t_h <= t1:
                u0 = math.log1p(max(t0, 0.0))
                u1 = math.log1p(max(t1, 0.0))
                u = math.log1p(max(t_h, 0.0))
                frac = 0.0 if u1 <= u0 else (u - u0) / (u1 - u0)
                return math.log(n0) + frac * (math.log(n1) - math.log(n0))
        return math.log(self.n_last)


def distinct_snapshots(history):
    """Collapse consecutive identical reads into refresh snapshots.

    Returns [(epoch, n_players, rates)] with the time of the first read
    that showed each snapshot, in time order.
    """
    snaps = []
    prev_key = None
    for read in history:
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if not n or n <= 0:
            continue
        key = (n, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append((parse_time(read["at"]), n, list(rates)))
    return snaps


def arrival_events(states, target_h, curve):
    """Player arrival events [(t_hours, dN)] behind the N curve.

    Snapshot jumps are arrivals at the refresh times.  The first
    cohort is split over the 12 h before the first read: the game
    unlocked at local midnight across timezones, so the first tracked
    players had already been playing for hours.  Future arrivals to
    the target come from the extrapolated N curve on a 6 h grid.
    """
    events = []
    if not states:
        return events
    t0, n0 = states[0]
    for off in (-2.0, -1.0, -0.5, 0.0):
        events.append((t0 + off, n0 * 0.25))
    prev = n0
    for t, n in states[1:]:
        if n > prev:
            events.append((t, n - prev))
        prev = n
    t = states[-1][0]
    prev_n = states[-1][1]
    while t < target_h - 1e-9:
        t2 = min(t + 6.0, target_h)
        n2 = math.exp(curve.log_n(t2))
        if n2 > prev_n:
            events.append((t2, n2 - prev_n))
        prev_n = n2
        t = t2
    return events


def _earned_share(age_h, median_h, sigma):
    """Share of starters of age age_h who hold the trophy:
    lognormal CDF with median median_h and log-spread sigma."""
    if age_h <= 0.0:
        return 0.0
    z = (math.log(age_h) - math.log(median_h)) / sigma
    return normal_cdf(z)


def mature_frac(events, n_at_t, t_h, median_h, sigma):
    """Share of players tracked at t_h who have had time to earn it:
    the arrival-weighted mean of the earned share over their ages."""
    earned = 0.0
    for tj, dn in events:
        if tj <= t_h:
            earned += dn * _earned_share(t_h - tj, median_h, sigma)
    frac = earned / max(n_at_t, 1.0)
    return clamp(frac, 0.002, 1.0)


def _fit_at_median(obs, y, weights, events, median_h, sigma):
    """Weighted mean log q and SSE for one (median, sigma) candidate.

    obs: [(t_hours, rate_percent, n_players)] time order.
    """
    terms = []
    for (t, _rate, n), yi, w in zip(obs, y, weights):
        frac = mature_frac(events, n, t, median_h, sigma)
        terms.append((yi - math.log(frac), w))
    log_q = weighted_mean(terms)
    if log_q is None:
        return None, None, None
    sse = 0.0
    wsum = 0.0
    for value, w in terms:
        sse += w * (value - log_q) ** 2
        wsum += w
    resid = math.sqrt(sse / wsum) if wsum > 0 else 0.6
    return log_q, sse, resid


def fit_trophy(obs, events, forced_median=None):
    """Fit (log q, median, residual sd, median table) for one trophy.

    obs: [(t_hours, rate_percent, n_players)] usable observations.
    forced_median skips the grid search (used when no median is
    identifiable from the history).  The table holds one
    (median, log_q, sse, resid) row per candidate median.
    """
    if not obs:
        return None, 0.0, 0.6, []
    t_last = obs[-1][0]
    # Recency weights: halve every 24 h back from the last observation.
    weights = [0.5 ** ((t_last - t) / 24.0) for t, _r, _n in obs]
    y = [math.log(max(rate, 0.005) / 100.0) for _t, rate, _n in obs]
    table = []
    for median_h in MEDIAN_GRID_HOURS:
        for sigma in SIGMA_GRID:
            log_q, sse, resid = _fit_at_median(obs, y, weights, events,
                                                median_h, sigma)
            if log_q is not None:
                table.append(((median_h, sigma), log_q, sse, resid))
    if not table:
        return y[-1], (0.0, 0.7), 0.6, []
    if forced_median is not None:
        for (median_h, sigma), log_q, _sse, resid in table:
            if median_h == forced_median and sigma == 0.7:
                return log_q, (median_h, sigma), resid, table
        return y[-1], (forced_median, 0.7), 0.6, table
    best = min(table, key=lambda row: row[2])
    return best[1], best[0], best[3], table


def mixture_quantiles(last_rate_pct, p_refresh, levels):
    """Quantiles (percent) of the no-refresh / one-refresh mixture.

    Component A (weight 1-p): no refresh, rate stays near last_rate.
    Component B (weight p): one refresh, tracked players x6.06 while
    earners stay put, so the rate is last_rate / 6.06.  Both components
    are normal on the logit scale; the mixture CDF is inverted by
    bisection.
    """
    mean_a = logit_frac(last_rate_pct / 100.0)
    mean_b = logit_frac(last_rate_pct / FIRST_REFRESH_JUMP / 100.0)
    lo = mean_b - 5.0 * MIX_REFRESH_SD
    hi = mean_a + 5.0 * MIX_NO_REFRESH_SD

    def cdf(x):
        return ((1.0 - p_refresh)
                * normal_cdf((x - mean_a) / MIX_NO_REFRESH_SD)
                + p_refresh
                * normal_cdf((x - mean_b) / MIX_REFRESH_SD))

    out = []
    for level in levels:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < level:
                a = mid
            else:
                b = mid
        q = inv_logit_frac(0.5 * (a + b)) * 100.0
        out.append(round(clamp(q, 0.01, 99.99), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_single_snapshot(case, snaps, target, checkpoint):
    """Forecast when the history holds one snapshot only (mixture)."""
    trophies = case.get("trophies") or []
    horizon_h = max((target - checkpoint) / 3600.0, 0.0)
    p_refresh = 1.0 - math.exp(-horizon_h / REFRESH_MEAN_INTERVAL_H)
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    rates = snaps[0][2] if snaps else []
    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id")
        last = rates[idx] if idx < len(rates) and rates[idx] is not None \
            else None
        if last is None:
            rows.append({"id": tid, "q": [0.01, 0.05, 0.2, 1.0, 5.0]})
            continue
        if last >= 99.99:
            rows.append({"id": tid,
                         "q": [99.9, 99.95, 99.99, 99.995, 100.0]})
            continue
        rows.append({"id": tid,
                     "q": mixture_quantiles(last, p_refresh, levels)})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    horizon_days = max((target - checkpoint) / 86400.0, 0.0)

    game = case.get("game") or {}
    release = game.get("release_date") or "2026-11-19"
    t_release = parse_time(release + "T00:00:00Z")
    target_h = (target - t_release) / 3600.0

    snaps = distinct_snapshots(history)
    if len(snaps) <= 1:
        return forecast_single_snapshot(case, snaps, target, checkpoint)

    states = [((t - t_release) / 3600.0, n) for t, n, _ in snaps]
    curve = PlayersCurve(states, target_h)
    events = arrival_events(states, target_h, curve)
    # Is any median identifiable?  Only when N actually moved.
    n_first = states[0][1]
    n_max = max(n for _, n in states)
    median_identified = n_max > n_first * 1.05

    # Per-trophy observations in hours since release.
    per_trophy_obs = []
    for idx in range(len(trophies)):
        obs = []
        for t, n, rates in snaps:
            if idx < len(rates):
                rate = rates[idx]
                if rate is not None and rate > 0:
                    obs.append((((t - t_release) / 3600.0), rate, n))
        per_trophy_obs.append(obs)

    # First pass: fit every trophy, collect class medians for shrinkage.
    fits = []
    for trophy, obs in zip(trophies, per_trophy_obs):
        forced = None if median_identified else \
            PRIOR_MEDIAN_HOURS[trophy_class(trophy)]
        log_q, median_h, resid, table = fit_trophy(obs, events, forced)
        if log_q is None:
            last = obs[-1][1] if obs else 1.0
            log_q = math.log(max(last, 0.005) / 100.0)
            median_h, resid = (0.0, 0.7), 0.6
        fits.append({"log_q": log_q, "median": median_h, "resid": resid,
                     "table": table, "forced": forced is not None,
                     "n_obs": len(obs),
                     "last": obs[-1][1] if obs else None})
    class_log_q = {}
    for cls in ("platinum", "story", "side"):
        vals = sorted(f["log_q"] for f, tr in zip(fits, trophies)
                      if trophy_class(tr) == cls and f["n_obs"] >= 2)
        if vals:
            class_log_q[cls] = vals[len(vals) // 2]

    n_target = math.exp(curve.log_n(target_h))
    n_growth = math.log(max(n_target / max(curve.n_last, 1.0), 1.0))

    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    z_scores = [NORMAL.inv_cdf(clamp(level, 1e-6, 1 - 1e-6))
                for level in levels]

    rows = []
    for trophy, fit, obs in zip(trophies, fits, per_trophy_obs):
        tid = trophy.get("id")
        last = fit["last"]
        # A trophy everyone already holds stays held.
        if last is not None and last >= 99.99:
            rows.append({"id": tid, "q": [99.9, 99.95, 99.99, 99.995, 100.0]})
            continue
        # Shrink every candidate tau's log q towards the class median
        # (more when the own fit is thin), then floor at the best rate
        # already seen: the eventual share cannot sit below it.
        cls = trophy_class(trophy)
        blend = 0.0
        if cls in class_log_q:
            blend = 0.20 if fit["n_obs"] >= 5 else 0.45
        best_seen = None
        if obs:
            best_seen = math.log(max(max(r for _, r, _n in obs), 0.005)
                                 / 100.0)

        def shrunk_log_q(raw):
            value = raw
            if blend:
                value = (1.0 - blend) * value + blend * class_log_q[cls]
            if best_seen is not None:
                value = max(value, best_seen)
            return min(value, math.log(0.9999))

        def rate_at_target(med_sig, raw_log_q):
            median_h, sigma = med_sig
            frac = mature_frac(events, n_target, target_h, median_h, sigma)
            return math.exp(shrunk_log_q(raw_log_q)) * frac

        # Censored median: when several medians fit equally well and
        # they imply target rates more than half a logit apart, the
        # history cannot tell them apart.  Take the log-median of the
        # tied medians' target rates and widen the spread by their
        # log-range, instead of trusting one arbitrary member of the tie.
        spread_extra = 0.0
        table = fit["table"]
        if table and not fit["forced"]:
            best_sse = min(row[2] for row in table)
            tied = [row for row in table
                    if row[2] <= best_sse * 1.05 + 0.02]
            if len(tied) > 1:
                log_rates = sorted(
                    math.log(max(rate_at_target(med, lq), 1e-6))
                    for med, lq, _s, _r in tied)
                if log_rates[-1] - log_rates[0] > 0.5:
                    mid = log_rates[len(log_rates) // 2]
                    if len(log_rates) % 2 == 0:
                        mid = 0.5 * (mid
                                     + log_rates[len(log_rates) // 2 - 1])
                    median_frac = clamp(math.exp(mid), 0.0001, 0.9999)
                    spread_extra = min((log_rates[-1] - log_rates[0])
                                       / 4.0, 2.0)
                else:
                    median_frac = clamp(
                        rate_at_target(fit["median"], fit["log_q"]),
                        0.0001, 0.9999)
            else:
                median_frac = clamp(
                    rate_at_target(fit["median"], fit["log_q"]),
                    0.0001, 0.9999)
        else:
            median_frac = clamp(
                rate_at_target(fit["median"], fit["log_q"]),
                0.0001, 0.9999)
        # Spread on the logit scale.
        resid = clamp(fit["resid"], RESID_FLOOR, RESID_CAP)
        rw = RW_PER_SQRT_DAY * math.sqrt(horizon_days)
        n_unc = min(N_UNCERT_PER_DOUBLING * (n_growth / math.log(2.0)), 0.5)
        spread = math.sqrt(rw ** 2 + resid ** 2 + MODEL_FLOOR ** 2
                           + n_unc ** 2 + spread_extra ** 2)
        centre = logit_frac(median_frac)
        quantiles = []
        for z in z_scores:
            q = inv_logit_frac(centre + z * spread) * 100.0
            quantiles.append(round(clamp(q, 0.01, 99.99), 4))
        # Enforce non-decreasing after rounding.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": tid, "q": quantiles})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    if not case_files:
        raise SystemExit("no .in case files found in %s" % input_dir)
    for path in case_files:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = case.get("id") or os.path.splitext(
            os.path.basename(path))[0]
        forecast = forecast_case(case)
        out_path = os.path.join(output_dir, "%s.out" % case_id)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
