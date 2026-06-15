"""Patrol-routing / dispatch optimization.

Turns the enforcement-priority ranking into concrete routes + schedules for a
fleet of patrol units leaving a depot within a shift-time budget — a
**prize-collecting VRP**: visit the locations that maximize captured priority
within each unit's time budget (it is fine, even expected, to skip low-value
stops). Two solvers, same interface:

  * OR-Tools routing (if ``ortools`` is installed) — disjunction penalties ∝
    priority, route-duration dimension, guided local search;
  * a dependency-free greedy + 2-opt fallback (deterministic, always available).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from curbiq import config as C
from curbiq.congestion import haversine_m


def _time_matrix(lat, lon, speed_kmph, dwell_s):
    """Seconds to travel i->j (+ service time at destination j, except depot=0)."""
    n = len(lat)
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    mps = speed_kmph * 1000.0 / 3600.0
    t = d / mps
    t[:, 1:] += dwell_s          # add dwell on arrival at any non-depot node (depot = idx 0)
    np.fill_diagonal(t, 0.0)
    return t


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def _solve_ortools(tm, priority, n_vehicles, depot, max_route_s, time_limit_s=5):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(priority)
    mgr = pywrapcp.RoutingIndexManager(n, n_vehicles, depot)
    routing = pywrapcp.RoutingModel(mgr)
    tm_int = np.rint(tm).astype(int)

    def cb(i, j):
        return int(tm_int[mgr.IndexToNode(i)][mgr.IndexToNode(j)])

    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    routing.AddDimension(idx, 0, int(max_route_s), True, "Time")
    routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(100)  # balance unit loads
    pmax = float(max(priority.max(), 1.0))
    for node in range(1, n):                       # depot can't be dropped
        penalty = int(priority[node] / pmax * 100000)  # skip high-priority = expensive
        routing.AddDisjunction([mgr.NodeToIndex(node)], penalty)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(time_limit_s)
    sol = routing.SolveWithParameters(p)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")
    routes = []
    for v in range(n_vehicles):
        i = routing.Start(v)
        route = []
        while not routing.IsEnd(i):
            node = mgr.IndexToNode(i)
            if node != depot:
                route.append(node)
            i = sol.Value(routing.NextVar(i))
        routes.append(route)
    return routes


def _two_opt(route, tm, depot):
    """2-opt on a single route bounded by the depot at both ends."""
    if len(route) < 3:
        return route
    path = [depot] + route + [depot]

    def length(p):
        return sum(tm[p[k]][p[k + 1]] for k in range(len(p) - 1))

    best = path
    improved = True
    while improved:
        improved = False
        for a in range(1, len(best) - 2):
            for b in range(a + 1, len(best) - 1):
                cand = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                if length(cand) + 1e-6 < length(best):
                    best = cand
                    improved = True
    return best[1:-1]


def _solve_greedy(tm, priority, n_vehicles, depot, max_route_s):
    n = len(priority)
    unvisited = set(range(n)) - {depot}
    routes = [[] for _ in range(n_vehicles)]
    spent = [0.0] * n_vehicles
    pos = [depot] * n_vehicles
    while unvisited:
        best = None
        for v in range(n_vehicles):
            for node in unvisited:
                t_to = tm[pos[v]][node]
                if spent[v] + t_to + tm[node][depot] <= max_route_s:
                    score = priority[node] / (t_to + 1.0)
                    if best is None or score > best[2]:
                        best = (v, node, score)
        if best is None:
            break
        v, node, _ = best
        routes[v].append(node)
        spent[v] += tm[pos[v]][node]
        pos[v] = node
        unvisited.discard(node)
    return [_two_opt(r, tm, depot) for r in routes]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def optimize_patrols(cells, n_units=C.PATROL_UNITS, top_k=C.PATROL_TOP_K,
                     depot=C.PATROL_DEPOT, shift_hours=C.PATROL_SHIFT_HOURS,
                     speed_kmph=C.PATROL_SPEED_KMPH, dwell_min=C.PATROL_DWELL_MIN,
                     shift_start=C.PATROL_SHIFT_START) -> dict:
    top = cells.sort_values("priority_score", ascending=False).head(top_k).reset_index(drop=True)
    # node 0 = depot, nodes 1..K = candidate stops
    lat = np.concatenate([[depot[0]], top["lat"].to_numpy()])
    lon = np.concatenate([[depot[1]], top["lon"].to_numpy()])
    priority = np.concatenate([[0.0], top["priority_score"].to_numpy()])
    tm = _time_matrix(lat, lon, speed_kmph, dwell_min * 60.0)
    max_route_s = shift_hours * 3600.0

    solver = "greedy"
    try:
        routes = _solve_ortools(tm, priority, n_units, 0, max_route_s)
        solver = "ortools"
    except Exception:
        routes = _solve_greedy(tm, priority, n_units, 0, max_route_s)

    try:
        start_dt = _dt.datetime.strptime(shift_start, "%H:%M")
    except ValueError:
        start_dt = _dt.datetime.strptime("17:30", "%H:%M")

    out_routes, covered, total_dist = [], set(), 0.0
    mps = speed_kmph * 1000.0 / 3600.0
    for v, route in enumerate(routes):
        stops, t_cursor, prev = [], 0.0, 0
        for node in route:
            t_cursor += tm[prev][node]
            covered.add(node)
            eta = (start_dt + _dt.timedelta(seconds=t_cursor)).strftime("%H:%M")
            total_dist += haversine_m(np.array([lat[prev]]), np.array([lon[prev]]),
                                      np.array([lat[node]]), np.array([lon[node]]))[0]
            stops.append({"seq": len(stops) + 1, "h3": top.iloc[node - 1]["h3"],
                          "lat": round(float(lat[node]), 5), "lon": round(float(lon[node]), 5),
                          "priority": round(float(priority[node]), 1), "eta": eta,
                          "top_offence": top.iloc[node - 1].get("top_offence")})
            prev = node
        if prev != 0:
            total_dist += haversine_m(np.array([lat[prev]]), np.array([lon[prev]]),
                                      np.array([lat[0]]), np.array([lon[0]]))[0]
        out_routes.append({
            "unit": f"Unit-{v + 1}", "n_stops": len(stops),
            "route_minutes": round(t_cursor / 60.0, 1),
            "priority_covered": round(sum(s["priority"] for s in stops), 1),
            "stops": stops,
        })

    return {
        "solver": solver,
        "n_units": n_units,
        "depot": {"lat": depot[0], "lon": depot[1]},
        "shift_start": shift_start,
        "shift_hours": shift_hours,
        "speed_kmph": speed_kmph,
        "dwell_min": dwell_min,
        "candidate_stops": int(len(top)),
        "stops_covered": int(len(covered)),
        "coverage_pct": round(100.0 * len(covered) / max(len(top), 1), 1),
        "total_priority_covered": round(float(priority[list(covered)].sum()), 1) if covered else 0.0,
        "total_distance_km": round(total_dist / 1000.0, 2),
        "routes": out_routes,
    }


if __name__ == "__main__":
    from curbiq.congestion import compute_congestion
    from curbiq.etl import load_processed
    from curbiq.forecast import run_forecast
    from curbiq.hotspots import compute_hotspots
    from curbiq.prioritize import run_prioritization

    df = load_processed()
    hot, _ = compute_hotspots(df)
    cis, _ = compute_congestion(df)
    fc = run_forecast(df)["forecast"]
    prio = run_prioritization(df, hot, cis, fc)["table"]
    r = optimize_patrols(prio)
    print(f"== patrol plan ({r['solver']}) ==")
    print(f"  {r['n_units']} units, {r['shift_hours']}h shift from {r['shift_start']} IST")
    print(f"  covered {r['stops_covered']}/{r['candidate_stops']} stops ({r['coverage_pct']}%), "
          f"{r['total_distance_km']} km total")
    for rt in r["routes"]:
        print(f"  {rt['unit']}: {rt['n_stops']} stops, {rt['route_minutes']} min, "
              f"priority {rt['priority_covered']}  -> first 3 ETAs "
              f"{[ (s['eta'], s['h3'][:7]) for s in rt['stops'][:3] ]}")
