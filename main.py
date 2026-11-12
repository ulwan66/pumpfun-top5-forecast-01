#!/usr/bin/env python3
"""Forecast the top 5 pump.fun coins by market cap at the answer reads.

Reads the house's 10-minute snapshots from $MAKEFUN_INPUT_DIR (or /input):
history.ndjson plus task.json. Writes one forecast to
$MAKEFUN_OUTPUT_DIR/forecast.json (or /output): per-place coin
probabilities, a top-5 ordering, and per-place market-cap size-bin
probabilities.

Approach, in brief: every coin's market cap is tracked in log10 SOL. From
the history we estimate how log caps move over one 10-minute step, pooled
by coin age and venue and shrunk toward global averages, with a separate
fitted crash component and a measured mean-reversion adjustment for the
coins currently leading. A Monte Carlo then draws each candidate's log
cap at every answer read, ranks the draws, and turns the simulated
frequencies into probabilities. A newcomer allowance, fitted from the
history's own rank churn, is left unnamed so the scorer spreads it over
chart coins the forecast does not name. Size forecasts blend the
simulated order statistics with the recent empirical distribution of
each place's size, shifted by its trend.

Standard library only, no network. Deterministic: one fixed seed, sorted
iteration everywhere, no wall-clock input.
"""

import json
import math
import os
import random
import sys
from bisect import bisect_right
from datetime import datetime, timezone

SEED = 20261112
N_PATHS = 4000
MAX_CANDIDATES = 400
MAX_ROWS_PER_PLACE = 80
MIN_NAMED_P = 0.001
PROB_SUM_CAP = 0.92
SIZE_SUM_CAP = 0.95
MIN_BIN_P = 0.002

# Prior for one 10-minute log10 step when the history says nothing.
PRIOR_DRIFT = 0.0
PRIOR_VOL = 0.30
# Shrinkage strengths (pseudo-counts) for pooled cells and single coins.
CELL_PRIOR_WEIGHT = 20.0
COIN_PRIOR_WEIGHT = 8.0
# A one-step move at or below this log10 drop counts as a crash, and is
# modelled separately from the ordinary diffusion.
JUMP_THRESHOLD = -0.25
# Drift stops accumulating past this many steps; volatility keeps growing
# a little longer. Extrapolating a hot streak 13 hours out is how a
# forecast goes wrong with confidence.
DRIFT_HORIZON_CAP = 24
VOL_HORIZON_CAP = 36
# Age bins in hours for the drift/volatility pooling.
AGE_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
# Rank bands for the mean-reversion adjustment: (upper rank, label).
RANK_BANDS = [(5, "top5"), (20, "top20"), (100, "top100"), (1 << 30, "rest")]


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds (float)."""
    if isinstance(text, (int, float)):
        return float(text)
    txt = str(text).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    return datetime.fromisoformat(txt).timestamp()


def log_cap_sol(row):
    """log10 of a row's market cap expressed in SOL."""
    try:
        lamports = int(row["market_cap_lamports"])
    except (KeyError, TypeError, ValueError):
        return None
    if lamports <= 0:
        return None
    return math.log10(lamports) - 9.0


def load_history(input_dir):
    """Return (marks, task): marks are parsed history lines in file order."""
    with open(os.path.join(input_dir, "task.json")) as fh:
        task = json.load(fh)
    marks = []
    path = os.path.join(input_dir, "history.ndjson")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    marks.append(json.loads(line))
    return marks, task


def build_series(marks):
    """Collect per-coin observations.

    Returns (series, mark_times): series maps a mint to a list of
    (mark_index, y, venue, rank, created_epoch) in mark order, and
    mark_times holds each mark's epoch time.
    """
    series = {}
    mark_times = []
    for idx, mark in enumerate(markets_safe(marks)):
        at = parse_time(mark.get("at", 0))
        mark_times.append(at)
        rows = mark.get("rows") or []
        for row in rows:
            mint = row.get("mint")
            if not mint:
                continue
            y = log_cap_sol(row)
            if y is None:
                continue
            created = parse_time(row["created_at"]) if row.get("created_at") else at
            venue = row.get("venue") or "curve"
            series.setdefault(mint, []).append(
                (idx, y, venue, row.get("rank"), created)
            )
    return series, mark_times


def markets_safe(marks):
    """Yield only dict marks; tolerate malformed lines gracefully."""
    for mark in marks:
        if isinstance(mark, dict):
            yield mark


def age_bin(age_hours):
    """Index of the age bin holding this coin age."""
    idx = 0
    for i, edge in enumerate(AGE_EDGES):
        if age_hours >= edge:
            idx = i
    return idx


def venue_group(venue):
    """Collapse venues into the two pricing regimes that behave alike."""
    if venue in ("pool", "migrating"):
        return "pool"
    return "curve"


def rank_band(rank):
    """Rank-band label for a rank, or 'rest' when the rank is unknown."""
    if rank is None:
        return "rest"
    for upper, label in RANK_BANDS:
        if rank <= upper:
            return label
    return "rest"


def mean_var(values):
    """(mean, variance) with a safe answer for tiny samples."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var


def collect_returns(series, mark_times):
    """One-step log-cap moves between consecutive marks, with context.

    Each record is (return, age_hours, venue_group, rank_band). Only
    pairs of observations exactly one mark apart are used, so a coin
    that drops out of the ranked rows and returns later contributes no
    stretched multi-step move.
    """
    records = []
    for mint in sorted(series):
        obs = series[mint]
        for prev, cur in zip(obs, obs[1:]):
            if cur[0] != prev[0] + 1:
                continue
            age = (mark_times[prev[0]] - prev[4]) / 3600.0
            records.append(
                (cur[1] - prev[1], age, venue_group(prev[2]), rank_band(prev[3]))
            )
    return records


class Estimator:
    """Pooled drift/volatility model with crash and mean-reversion terms."""

    def __init__(self, records):
        ordinary = [r for r in records if r[0] > JUMP_THRESHOLD]
        crashes = [r[0] for r in records if r[0] <= JUMP_THRESHOLD]
        self.global_mean, global_var = mean_var([r[0] for r in ordinary])
        if len(ordinary) < 10:
            self.global_mean, global_var = PRIOR_DRIFT, PRIOR_VOL ** 2
        self.global_vol = min(1.0, max(0.05, math.sqrt(max(global_var, 1e-6))))
        # Pooled cells keyed by (age bin, venue group), shrunk to global.
        cells = {}
        for r, age, venue, _band in ordinary:
            cells.setdefault((age_bin(age), venue), []).append(r)
        self.cells = {}
        for key, vals in cells.items():
            m, v = mean_var(vals)
            n = len(vals)
            shrunk_mean = (n * m + CELL_PRIOR_WEIGHT * self.global_mean) / (
                n + CELL_PRIOR_WEIGHT
            )
            shrunk_var = (n * v + CELL_PRIOR_WEIGHT * global_var) / (
                n + CELL_PRIOR_WEIGHT
            )
            self.cells[key] = (
                shrunk_mean,
                min(1.0, max(0.05, math.sqrt(max(shrunk_var, 1e-6)))),
            )
        # Mean reversion by current rank band, measured, then dampened.
        # The threshold is low on purpose: on a short history the top
        # band has few observations, and dropping them discards exactly
        # the regime the forecast cares about.
        bands = {}
        for r, _age, _venue, band in ordinary:
            bands.setdefault(band, []).append(r)
        self.band_adj = {}
        self.band_vol = {}
        for band, vals in bands.items():
            m, v = mean_var(vals)
            if len(vals) >= 8:
                self.band_adj[band] = 0.5 * (m - self.global_mean)
            else:
                self.band_adj[band] = 0.0
            n = len(vals)
            shrunk_var = (n * v + CELL_PRIOR_WEIGHT * global_var) / (
                n + CELL_PRIOR_WEIGHT
            )
            self.band_vol[band] = min(
                1.0, max(0.05, math.sqrt(max(shrunk_var, 1e-6)))
            )
        # Crash component: frequency and size of the excluded moves.
        total = len(records)
        if total > 0 and len(crashes) >= 5:
            self.jump_p = min(0.15, max(0.0, len(crashes) / total))
            jm, jv = mean_var(crashes)
            self.jump_mean = min(-0.1, jm)
            self.jump_sd = min(0.6, max(0.05, math.sqrt(max(jv, 1e-6))))
        else:
            self.jump_p, self.jump_mean, self.jump_sd = 0.02, -0.45, 0.20

    def cell_params(self, age_hours, venue):
        """Shrunk (drift, vol) for a coin's age/venue cell."""
        key = (age_bin(age_hours), venue_group(venue))
        if key in self.cells:
            return self.cells[key]
        # Fall back to any cell sharing the venue, else the global pair.
        for (b, v), params in sorted(self.cells.items()):
            if v == key[1]:
                return params
        return self.global_mean, self.global_vol

    def coin_params(self, obs, mark_times, last_idx):
        """(anchor_y, drift, vol, band_adj) for one candidate coin.

        The anchor is the coin's last observed log cap. Its own drift is
        shrunk hard toward its cell: a handful of steps says little.
        """
        own = []
        for prev, cur in zip(obs, obs[1:]):
            if cur[0] == prev[0] + 1 and cur[1] - prev[1] > JUMP_THRESHOLD:
                own.append(cur[1] - prev[1])
        last = obs[-1]
        age = (mark_times[last[0]] - last[4]) / 3600.0
        cell_mean, cell_vol = self.cell_params(age, last[2])
        n = len(own)
        if n:
            m, _v = mean_var(own)
            drift = (n * m + COIN_PRIOR_WEIGHT * cell_mean) / (n + COIN_PRIOR_WEIGHT)
        else:
            drift = cell_mean
        stale = last_idx - last[0]
        band = rank_band(last[3])
        # Volatility is the larger of the coin's age/venue cell and its
        # rank band's: the pooled average is dominated by quiet coins
        # and badly understates how much the leaders move.
        vol = max(cell_vol, self.band_vol.get(band, cell_vol))
        vol = vol * min(1.5, 1.0 + 0.05 * stale)
        vol = min(1.0, max(0.05, vol))
        return last[1], drift, vol, self.band_adj.get(band, 0.0)


def fit_newcomer_rate(marks):
    """Fit how fast unseen coins take top-5 slots, from rank churn.

    For each lag L, measure the share of top-5 slots at mark i+L held by
    mints absent from mark i's rows, then fit q(L) = 1 - exp(-lam*L) by
    grid search. Returns lam; q(h) is the newcomer allowance at horizon
    h steps.
    """
    top_sets = []
    row_sets = []
    for mark in marks:
        rows = mark.get("rows") or []
        ordered = sorted(rows, key=lambda r: r.get("rank", 1 << 30))
        top_sets.append({r["mint"] for r in ordered[:5] if r.get("mint")})
        row_sets.append({r["mint"] for r in rows if r.get("mint")})
    points = []
    n = len(marks)
    for lag in range(1, n):
        shares = []
        for i in range(n - lag):
            top = top_sets[i + lag]
            if not top:
                continue
            unseen = sum(1 for mint in top if mint not in row_sets[i])
            shares.append(unseen / len(top))
        if shares:
            points.append((lag, sum(shares) / len(shares)))
    if not points:
        return 0.02
    best_lam, best_err = 0.02, None
    lam = 0.001
    while lam <= 0.5:
        err = sum((1.0 - math.exp(-lam * lag) - share) ** 2 for lag, share in points)
        if best_err is None or err < best_err:
            best_err, best_lam = err, lam
        lam *= 1.15
    return best_lam


def select_candidates(series, marks, last_idx):
    """Candidate mints: last mark's rows plus recent top-20 alumni.

    Capped by best observed log cap so the simulation stays inside its
    time budget on the long final history.
    """
    last_rows = (marks[last_idx].get("rows") or []) if marks else []
    chosen = {r["mint"] for r in last_rows if r.get("mint")}
    recent = marks[max(0, last_idx - 2): last_idx + 1]
    for mark in recent:
        for row in mark.get("rows") or []:
            rank = row.get("rank")
            if rank is not None and rank <= 20 and row.get("mint"):
                chosen.add(row["mint"])
    scored = []
    for mint in chosen:
        obs = series.get(mint)
        if not obs:
            continue
        scored.append((max(o[1] for o in obs), mint))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [mint for _best, mint in scored[:MAX_CANDIDATES]]


def simulate(candidates, params, horizons, jump, n_places, edges, rng):
    """Monte Carlo over terminal log caps at each answer read.

    candidates: mints in fixed order. params: parallel list of
    (anchor_y, drift, vol, band_adj). horizons: read horizons in steps.
    jump: (p_per_step, mean, sd). Returns (place_counts, size_counts):
    place_counts[mint][place] counts simulated top-place finishes, and
    size_counts[place][bin] counts the simulated size at each place.
    Draws at different reads are independent terminal draws; the score
    averages over reads, so only the marginals matter.
    """
    jump_p, jump_mean, jump_sd = jump
    n_cand = len(candidates)
    # Per-horizon, per-candidate shift and spread, precomputed once.
    shifts = []
    spreads = []
    jump_probs = []
    for h in horizons:
        h_drift = min(h, DRIFT_HORIZON_CAP)
        h_vol = min(h, VOL_HORIZON_CAP)
        shifts.append([(drift + adj) * h_drift for (_a, drift, _v, adj) in params])
        spreads.append([vol * math.sqrt(h_vol) for (_a, _d, vol, _adj) in params])
        jump_probs.append(1.0 - (1.0 - jump_p) ** h_drift)
    anchors = [p[0] for p in params]
    place_counts = {mint: [0] * (n_places + 1) for mint in candidates}
    size_counts = {place: {} for place in range(1, n_places + 1)}
    gauss = rng.gauss
    rand = rng.random
    for _path in range(N_PATHS):
        for hi in range(len(horizons)):
            shift = shifts[hi]
            spread = spreads[hi]
            jp = jump_probs[hi]
            drawn = []
            for i in range(n_cand):
                y = anchors[i] + shift[i] + gauss(0.0, spread[i])
                if rand() < jp:
                    y += min(-0.05, jump_mean + gauss(0.0, jump_sd))
                drawn.append((y, candidates[i]))
            drawn.sort(key=lambda item: (-item[0], item[1]))
            for place in range(1, n_places + 1):
                if place - 1 >= len(drawn):
                    break
                y, mint = drawn[place - 1]
                place_counts[mint][place] += 1
                b = bisect_right(edges, y)
                bins = size_counts[place]
                bins[b] = bins.get(b, 0) + 1
    return place_counts, size_counts


def normal_cdf(x):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def empirical_size_dist(marks, place, mean_horizon_marks, edges):
    """Recency-weighted empirical distribution of one place's size.

    Takes the log cap at this rank in every mark that has one, weights
    recent marks more, and shifts the fitted normal by the series'
    trend extrapolated over the mean horizon. Returns {bin: p}.
    """
    values = []
    for idx, mark in enumerate(marks):
        rows = mark.get("rows") or []
        for row in rows:
            if row.get("rank") == place:
                y = log_cap_sol(row)
                if y is not None:
                    values.append((idx, y))
                break
    if len(values) < 2:
        return {}
    last_idx = len(marks) - 1
    weights = [0.85 ** (last_idx - idx) for idx, _y in values]
    wsum = sum(weights)
    mean = sum(w * y for w, (_i, y) in zip(weights, values)) / wsum
    var = sum(w * (y - mean) ** 2 for w, (_i, y) in zip(weights, values)) / wsum
    sd = max(0.15, math.sqrt(max(var, 1e-6)))
    # Least-squares trend per mark, extrapolated and clipped.
    n = len(values)
    mean_i = sum(idx for idx, _y in values) / n
    mean_y = sum(y for _i, y in values) / n
    denom = sum((idx - mean_i) ** 2 for idx, _y in values)
    slope = 0.0
    if denom > 0:
        slope = sum((idx - mean_i) * (y - mean_y) for idx, y in values) / denom
    shift = max(-0.5, min(0.5, slope * mean_horizon_marks))
    centre = mean + shift
    dist = {}
    bounds = [float("-inf")] + list(edges) + [float("inf")]
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        p_hi = 1.0 if hi == float("inf") else normal_cdf((hi - centre) / sd)
        p_lo = 0.0 if lo == float("-inf") else normal_cdf((lo - centre) / sd)
        p = p_hi - p_lo
        if p > 1e-9:
            dist[b] = p
    return dist


def floor6(p):
    """Round a probability down to 6 decimals, keeping sums honest."""
    return math.floor(p * 1e6) / 1e6


def shape_probabilities(place_counts, total_draws, newcomer_q, n_places,
                        chart_size):
    """Turn simulated frequencies into the probabilities list.

    Frequencies are scaled by (1 - q) so the newcomer allowance stays
    unnamed, capped as tail insurance, pruned of rows that score worse
    than leaving the coin unnamed, and floored to 6 decimals.
    """
    rows = []
    for place in range(1, n_places + 1):
        scored = []
        for mint in sorted(place_counts):
            freq = place_counts[mint][place] / total_draws
            if freq > 0:
                scored.append([mint, freq * (1.0 - newcomer_q)])
        total = sum(p for _m, p in scored)
        if total > PROB_SUM_CAP:
            scale = PROB_SUM_CAP / total
            for item in scored:
                item[1] *= scale
            total = PROB_SUM_CAP
        scored = [item for item in scored if item[1] >= MIN_NAMED_P]
        # Prune rows below the unnamed per-coin share, iterated to a
        # fixed point: naming such a coin scores worse than not naming it.
        for _round in range(3):
            total = sum(p for _m, p in scored)
            unnamed_slots = max(1, chart_size - len(scored))
            share = (1.0 - total) / unnamed_slots
            kept = [item for item in scored if item[1] >= share]
            if len(kept) == len(scored):
                break
            scored = kept
        scored.sort(key=lambda item: (-item[1], item[0]))
        for mint, p in scored[:MAX_ROWS_PER_PLACE]:
            p6 = floor6(p)
            if p6 > 0:
                rows.append({"mint": mint, "place": place, "p": p6})
    return rows


def shape_sizes(size_counts, total_draws, marks, mean_horizon_marks,
                n_places, edges):
    """Blend simulated order statistics with the empirical place sizes."""
    rows = []
    for place in range(1, n_places + 1):
        counts = size_counts.get(place, {})
        sim_total = sum(counts.values()) or 1
        mixed = {}
        for b, c in counts.items():
            mixed[b] = 0.65 * (c / sim_total)
        empirical = empirical_size_dist(marks, place, mean_horizon_marks, edges)
        for b, p in empirical.items():
            mixed[b] = mixed.get(b, 0.0) + 0.35 * p
        total = sum(mixed.values())
        if total <= 0:
            continue
        if total > SIZE_SUM_CAP:
            scale = SIZE_SUM_CAP / total
            for b in mixed:
                mixed[b] *= scale
        kept = {b: p for b, p in mixed.items() if p >= MIN_BIN_P}
        if not kept:
            best = max(mixed.items(), key=lambda kv: (kv[1], -kv[0]))
            kept = {best[0]: best[1]}
        for b in sorted(kept):
            p6 = floor6(kept[b])
            if p6 > 0:
                rows.append({"place": place, "bin": b, "p": p6})
    return rows


def order_top5(place_counts, total_draws, n_places, candidates):
    """Order candidates by expected place, then by place-1 frequency."""
    scored = []
    for mint in candidates:
        counts = place_counts[mint]
        freqs = [counts[place] / total_draws for place in range(1, n_places + 1)]
        expected = sum((k + 1) * f for k, f in enumerate(freqs))
        expected += (n_places + 1) * max(0.0, 1.0 - sum(freqs))
        scored.append((expected, -freqs[0], mint))
    scored.sort()
    return [mint for _e, _f, mint in scored[:n_places]]


def dummy_mints(count):
    """Distinct valid base58 placeholders for a history with no coins."""
    base = "1" * 43
    return [base + str(i + 1) for i in range(count)]


def fallback_forecast(marks, n_places):
    """A well-formed even-spread forecast for degenerate input."""
    known = []
    seen = set()
    for mark in reversed(marks):
        for row in mark.get("rows") or []:
            mint = row.get("mint")
            if mint and mint not in seen:
                seen.add(mint)
                known.append(mint)
    top = []
    if marks:
        last_rows = sorted(marks[-1].get("rows") or [],
                           key=lambda r: r.get("rank", 1 << 30))
        top = [r["mint"] for r in last_rows if r.get("mint")][:n_places]
    for mint in known:
        if len(top) >= n_places:
            break
        if mint not in top:
            top.append(mint)
    need = n_places - len(top)
    if need > 0:
        top.extend(dummy_mints(need))
    return {"probabilities": [], "sizes": [], "top5": top[:n_places]}


def build_forecast(marks, task):
    """Full pipeline: estimate, simulate, shape. Raises on bad input."""
    if not marks:
        raise ValueError("empty history")
    series, mark_times = build_series(marks)
    last_idx = len(marks) - 1
    candidates = select_candidates(series, marks, last_idx)
    if len(candidates) < 5:
        raise ValueError("too few candidates")
    n_places = int(task.get("places", 5))
    chart_size = int(task.get("chart_size", 100))
    edges = [float(e) for e in (task.get("size") or {}).get("edges", [])]
    if len(edges) < 2:
        edges = [2.0 + 0.05 * i for i in range(81)]
    cadence = float(task.get("cadence_minutes", 10)) * 60.0
    last_time = mark_times[last_idx]
    reads = task.get("reads") or []
    read_times = [parse_time(r) for r in reads]
    if not read_times:
        read_times = [last_time + 3600.0]
    horizons = [max(1, int(round((rt - last_time) / cadence))) for rt in read_times]
    mean_horizon_marks = sum(horizons) / len(horizons)
    records = collect_returns(series, mark_times)
    estimator = Estimator(records)
    params = [estimator.coin_params(series[m], mark_times, last_idx)
              for m in candidates]
    lam = fit_newcomer_rate(marks)
    newcomer_q = sum(1.0 - math.exp(-lam * h) for h in horizons) / len(horizons)
    newcomer_q = min(0.65, max(0.05, newcomer_q))
    rng = random.Random(SEED)
    place_counts, size_counts = simulate(
        candidates, params, horizons,
        (estimator.jump_p, estimator.jump_mean, estimator.jump_sd),
        n_places, edges, rng)
    total_draws = N_PATHS * len(horizons)
    probabilities = shape_probabilities(place_counts, total_draws, newcomer_q,
                                        n_places, chart_size)
    sizes = shape_sizes(size_counts, total_draws, marks, mean_horizon_marks,
                        n_places, edges)
    top5 = order_top5(place_counts, total_draws, n_places, candidates)
    if len(top5) < n_places:
        raise ValueError("top5 incomplete")
    return {"probabilities": probabilities, "sizes": sizes, "top5": top5}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        marks, task = load_history(input_dir)
        forecast = build_forecast(marks, task)
        out_name = task.get("output") or "forecast.json"
        n_places = int(task.get("places", 5))
    except Exception:
        forecast = None
        out_name = "forecast.json"
        n_places = 5
        try:
            marks, _task = load_history(input_dir)
        except Exception:
            marks = []
    if forecast is None:
        forecast = fallback_forecast(marks, n_places)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w") as fh:
        json.dump(forecast, fh, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
