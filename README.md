# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Krisnaa19/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Krisnaa19/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z, a week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only, no network, no model calls,
fully deterministic) reads every case file (`*.in`) from
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one forecast file
`/output/<id>.out` per case.

A trophy's earn rate is earners / players-tracked, and both move — while
the page only refreshes every few hours. The forecaster separates them:

- **Players tracked N(t).** The distinct refresh snapshots give N at a
  handful of times. log N is fitted as a power law in log(1 + hours since
  release) over the recent snapshots, then extrapolated to the target
  with the growth rate damped only mildly (the recent window already
  reflects the current, slower rate) and the total multiplier capped.
- **Earners.** A player who starts at time s earns a trophy after a
  random play time; the share of starters who hold it by age a is
  lognormal, `Phi((ln a - ln m) / sigma)`, with `m` the trophy's median
  time-to-earn (an early activity: under an hour; a chapter: a few hours;
  100% completion: hundreds of hours) and `sigma` fitted per trophy
  (0.35 / 0.7 / 1.2). Only a share `q` of players ever earn it. The
  rate is `q` times the arrival-weighted mean of that share over the
  tracked players' ages. As growth slows, the rate recovers towards `q`
  — that mean reversion is the edge over a persistence baseline and over
  a log-time trend, which extrapolates the early dilution forever.
  `q` is shrunk towards its class median (platinum / story / side).
  When several medians fit equally well but imply very different target
  rates, the forecast takes their log-median and widens the spread.
- **Spread.** Quantiles are the median on the logit scale plus normal
  z-scores times a per-trophy logit sd (random-walk term growing with
  the horizon, fit residual, model floor, N-extrapolation term).
- **No refresh ever seen.** When the history holds a single snapshot,
  the forecast is a two-component mixture: no refresh (rate stays) vs
  one refresh (tracked players x6.06, earners stay put, rate / 6.06 —
  launch night's first refresh: 638 → 3,868 players while the newcomers
  had earned ~1 trophy each).

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/p00.out, out/p01.out, ...
```

## Validation

- Practice cases (p00/p01): pinball loss **0.66** against the
  reconstructed 04:30/05:00 refresh answers, vs **1.69** for the
  persistence baseline (house baseline ~1.51).
- Synthetic final-like cases (2 days of refreshes, known ground truth at
  Nov 26, three maturation worlds — sharp delays, exponential, lognormal;
  3 seeds each): mean loss **1.02 / 0.39 / 0.44**, vs persistence
  2.76 / 0.41 / 0.70 and the log-time curve baseline 3.57 / 0.25 / 0.44.
  The model is the most robust across worlds; it never scores
  catastrophically in any of them.
- Deterministic: two runs on the same case produce byte-identical
  output. Runs in ~0.1 s per practice case, ~0.4 s for a final-like
  case — far inside the 8 s cap.

## Layout

- `main.py` — the forecaster (the entrypoint).
- `README.md` — this file.
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry wallet.

---

<sub>Entered on [make.fun](https://make.fun/?ref=k9ejmyk6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
