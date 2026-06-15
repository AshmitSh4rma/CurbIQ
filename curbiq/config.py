"""Central configuration & domain constants for CurbIQ.

Every tunable parameter lives here so the ETL, analytics, API and docs share a
single source of truth. Values were derived from profiling the real Bengaluru
Traffic Police dataset (298,450 records, Nov 2023-Apr 2024) and from the
traffic-engineering literature (IRC PCU values, HCM side-friction, BPR). They
are refined by the research brief in ``docs/RESEARCH.md``.
"""
from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "police_violations.csv.gz"  # gzipped to save disk; pandas reads transparently
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"      # JSON/GeoJSON consumed by API + frontend
MODELS_DIR = ROOT / "models"
WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"

PROCESSED_PARQUET = PROCESSED_DIR / "violations.parquet"

for _d in (PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
SOURCE_TZ = "UTC"            # created_datetime carries a +00 offset
LOCAL_TZ = "Asia/Kolkata"    # Bengaluru, IST (UTC+5:30)

# IST congestion peak windows (half-open hour ranges). Used to weight congestion
# *impact* — NOT to describe enforcement timing (which is patrol-shift biased).
MORNING_PEAK = (8, 11)       # 08:00-10:59 IST
EVENING_PEAK = (17, 21)      # 17:00-20:59 IST

# --------------------------------------------------------------------------- #
# Geography (Bengaluru) — bounds taken from data profiling
# --------------------------------------------------------------------------- #
BBOX = {"lat_min": 12.80, "lat_max": 13.30, "lon_min": 77.44, "lon_max": 77.78}
CITY_CENTER = (12.9716, 77.5946)

# H3 resolutions. res 9 (~174 m edge, ~0.105 km^2) is street-block scale and is
# the PRIMARY unit for hotspot analysis; res 10 is fine-grained, res 8 is zonal.
H3_RES_PRIMARY = 9
H3_RES_FINE = 10
H3_RES_COARSE = 8

# --------------------------------------------------------------------------- #
# Offence taxonomy (code -> canonical label), from dataset profiling
# --------------------------------------------------------------------------- #
OFFENCE_LABELS: dict[int, str] = {
    104: "PARKING NEAR ROAD CROSSING",
    105: "PARKING ON FOOTPATH",
    106: "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    107: "PARKING IN A MAIN ROAD",
    108: "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE",
    109: "DOUBLE PARKING",
    110: "FAIL TO USE SAFETY BELTS",
    111: "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    112: "WRONG PARKING",
    113: "NO PARKING",
    115: "JUMPING TRAFFIC SIGNAL",
    116: "DEFECTIVE NUMBER PLATE",
    123: "CARRYING LENGTHY MATERIAL",
    124: "REFUSE TO GO FOR HIRE",
    125: "DEMANDING EXCESS FARE",
    130: "VIOLATING LANE DISCIPLINE",
    133: "USING BLACK FILM/OTHER MATERIALS",
    134: "U TURN PROHIBITED",
    135: "AGAINST ONE WAY/NO ENTRY",
    136: "OBSTRUCTING DRIVER",
    139: "PARKING OTHER THAN BUS STOP",
    140: "RIDER NOT WEARING HELMET",
    144: "WITHOUT SIDE MIRROR",
    146: "STOPPING ON WHITE/STOP LINE",
    147: "H T V PROHIBITED",
    237: "2W/3W - USING MOBILE PHONE",
    437: "OTHER - USING MOBILE PHONE",
}

# The parking offences that actually choke carriageways / footpaths / junctions.
PARKING_OFFENCE_CODES: set[int] = {104, 105, 106, 107, 108, 109, 111, 112, 113, 139}

# Carriageway-severity weight in [0, 1]: how directly the offence obstructs a
# moving traffic lane / intersection (1.0 = blocks an arterial through-lane).
OFFENCE_CARRIAGEWAY_SEVERITY: dict[int, float] = {
    107: 1.00,  # parking in a main road (arterial through-lane)
    109: 0.95,  # double parking (removes a live lane)
    106: 0.95,  # near traffic light / zebra (intersection throughput)
    104: 0.90,  # near road crossing (intersection)
    113: 0.85,  # no parking
    112: 0.80,  # wrong parking
    108: 0.75,  # opposite another parked vehicle (squeezes two-way road)
    111: 0.55,  # near bus stop / school / hospital
    139: 0.50,  # parking other than bus stop
    105: 0.35,  # footpath (pedestrian impact; less carriageway loss)
}
DEFAULT_SEVERITY = 0.50

# --------------------------------------------------------------------------- #
# Vehicle footprint (static lane-blockage), relative to a car (PCU-like).
# Anchored to IRC:106 / Indo-HCM PCU values and physical footprint. A parked
# heavy vehicle obstructs far more carriageway than a parked two-wheeler.
# --------------------------------------------------------------------------- #
VEHICLE_FOOTPRINT_PCU: dict[str, float] = {
    "MOPED": 0.30,
    "SCOOTER": 0.40,
    "MOTOR CYCLE": 0.40,
    "PASSENGER AUTO": 0.80,
    "GOODS AUTO": 0.90,
    "CAR": 1.00,
    "JEEP": 1.10,
    "MAXI-CAB": 1.20,
    "VAN": 1.30,
    "SCHOOL VEHICLE": 1.50,
    "LGV": 1.60,
    "TEMPO": 1.80,
    "MINI LORRY": 2.00,
    "TRACTOR": 2.50,
    "LORRY/GOODS VEHICLE": 3.00,
    "FACTORY BUS": 3.00,
    "PRIVATE BUS": 3.00,
    "BUS (BMTC/KSRTC)": 3.30,
    "TOURIST BUS": 3.30,
    "HGV": 3.50,
    "TANKER": 3.50,
    "OTHERS": 1.00,
}
DEFAULT_FOOTPRINT_PCU = 1.00

# Coarse vehicle category for reporting.
VEHICLE_CATEGORY: dict[str, str] = {
    "MOPED": "two_wheeler", "SCOOTER": "two_wheeler", "MOTOR CYCLE": "two_wheeler",
    "PASSENGER AUTO": "three_wheeler", "GOODS AUTO": "three_wheeler",
    "CAR": "light_4w", "JEEP": "light_4w", "MAXI-CAB": "light_4w", "VAN": "light_4w",
    "LGV": "lcv", "TEMPO": "lcv", "MINI LORRY": "lcv", "SCHOOL VEHICLE": "lcv",
    "LORRY/GOODS VEHICLE": "heavy", "HGV": "heavy", "TANKER": "heavy",
    "TRACTOR": "heavy",
    "BUS (BMTC/KSRTC)": "bus", "PRIVATE BUS": "bus", "TOURIST BUS": "bus",
    "FACTORY BUS": "bus",
    "OTHERS": "other",
}

# --------------------------------------------------------------------------- #
# Road-class inference from the free-text ``location`` address.
# Ordered (priority) patterns; first match wins. Weight in [0, 1] = how critical
# the road is to network throughput (a parked car on an arterial hurts far more
# than one in a residential layout).
# --------------------------------------------------------------------------- #
ROAD_CLASS_PATTERNS: list[tuple[str, str, float]] = [
    (r"\b(outer ring road|nice road|orr)\b", "ring_arterial", 1.00),
    (r"\bring road\b", "ring_arterial", 0.95),
    (r"\b(flyover|elevated|expressway|underpass|grade separator)\b", "grade_separated", 0.95),
    (r"\b(circle|signal|junction|cross junction)\b", "junction", 0.90),
    (r"\bmain road\b", "arterial", 0.85),
    (r"\b(bridge|overbridge|rob)\b", "bridge", 0.82),
    (r"\bmarket\b", "commercial", 0.80),
    (r"\b(station|metro|terminal|depot)\b", "transit_node", 0.75),
    (r"\b(\d+(st|nd|rd|th)?\s+)?cross( road)?\b", "local_connector", 0.45),
    (r"\b(layout|nagar|colony|block|enclave|garden|extension|township)\b", "residential", 0.35),
    (r"\b(road|rd|street|st|marg)\b", "collector", 0.55),
]
DEFAULT_ROAD_CLASS = "unknown"
DEFAULT_ROAD_WEIGHT = 0.50
_COMPILED_ROAD_PATTERNS = [(re.compile(p, re.I), name, w) for p, name, w in ROAD_CLASS_PATTERNS]

# --------------------------------------------------------------------------- #
# Validation confidence: how much to trust a record as a genuine violation.
# rejected records are likely false positives; duplicates are dropped from counts.
# --------------------------------------------------------------------------- #
VALIDATION_CONFIDENCE: dict[str, float] = {
    "approved": 1.00,
    "processing": 0.60,
    "created1": 0.60,
    "NULL": 0.60,        # not yet validated
    "rejected": 0.15,
    "duplicate": 0.00,
}
DEFAULT_CONFIDENCE = 0.60

# --------------------------------------------------------------------------- #
# Congestion model (traffic-engineering defaults; refined by docs/RESEARCH.md)
# --------------------------------------------------------------------------- #
# --- HCM 2000 Ch.16 on-street-parking capacity factor ----------------------
#   fp = (N - 0.1 - HCM_MANEUVER_BLOCK_S * Nm / 3600) / N
# where N = through-lanes/direction, Nm = parking maneuvers/hour (we proxy this
# with the cell's hourly violation rate). capacity_loss = 1 - fp.
HCM_MANEUVER_BLOCK_S = 18.0       # a maneuver blocks the adjacent lane ~18 s (HCM 2000)
HCM_MANEUVER_RATE_CAP = 180.0     # cap Nm at 180/h (HCM 2000)
HCM_FP_FLOOR = 0.05               # fp floored at 0.05
MAX_CAPACITY_LOSS = 0.60          # cap loss (Indian side-friction studies 17-66%)
# Detected violations under-count true parking maneuvers (enforcement is sparse).
# Maneuver rate is estimated as events per *active* peak-hour; operators who know
# their enforcement capture rate (e.g. catch 1 in 4) can raise this multiplier.
ENFORCEMENT_CAPTURE_MULT = 1.0

# Assumed through-lanes per direction by road class (drives the HCM N term).
ASSUMED_LANES: dict[str, int] = {
    "ring_arterial": 3, "grade_separated": 3, "arterial": 2, "bridge": 2,
    "junction": 2, "commercial": 2, "transit_node": 2, "collector": 2,
    "local_connector": 1, "residential": 1, "unknown": 1,
}
DEFAULT_LANES = 2

# Road-class capacity-loss lookup (curb-obstruction rule-of-thumb, 15-50%).
ROAD_CLASS_CAPACITY_LOSS: dict[str, float] = {
    "ring_arterial": 0.40, "grade_separated": 0.40, "arterial": 0.40, "bridge": 0.40,
    "junction": 0.30, "commercial": 0.30, "transit_node": 0.30, "collector": 0.30,
    "local_connector": 0.15, "residential": 0.15, "unknown": 0.15,
}
DEFAULT_ROAD_CAPACITY_LOSS = 0.20

# BPR volume-delay function:  t = t0 * (1 + ALPHA * (v/c) ** BETA)  (US BPR 1964)
BPR_ALPHA = 0.15
BPR_BETA = 4.0
BASELINE_VC_RATIO = 0.80          # assumed peak baseline v/c (research default)
VC_CLAMP = 2.0                    # clamp v/c' (beta=4 explodes past capacity)

# Junction-proximity exponential decay length (metres); ~one res-9 cell radius,
# honouring the KSP Datathon 100 m bottleneck rule.
JUNCTION_DECAY_D0_M = 150.0

# Composite Congestion Impact Score weights (sum to 1.0). Each factor is
# z-standardized across cells before weighting, then min-max scaled to 0-100.
CIS_WEIGHTS: dict[str, float] = {
    "severity_density": 0.35,   # severity-weighted violation density (obstruction x count)
    "junction_proximity": 0.25,
    "peak_overlap": 0.25,
    "road_capacity_loss": 0.15,
}

# --------------------------------------------------------------------------- #
# Hotspot statistics
# --------------------------------------------------------------------------- #
# Getis-Ord Gi* significance bands by |z|, with Benjamini-Hochberg FDR applied.
GI_Z_BANDS = {"90%": 1.65, "95%": 1.96, "99%": 2.58, "99.9%": 3.29}
GI_DISPLAY_Z = 3.29          # only flag |z| >= this as a high-confidence hotspot
FDR_Q = 0.05                 # false-discovery-rate target for hotspot significance
GI_NEIGHBOR_K = 1            # H3 grid-disk radius for spatial weights (k=1: self + 6 nbrs)
GI_SENSITIVITY_K = 2         # secondary radius reported for MAUP / smoothing sensitivity
MORAN_PERMUTATIONS = 999     # conditional permutations for Local Moran pseudo-p
MORAN_SEED = 42
DBSCAN_EPS_M = 120.0         # DBSCAN neighbourhood radius (metres)
DBSCAN_MIN_SAMPLES = 15
EMERGING_MIN_EVENTS_PER_BIN = 3   # min events/week for a cell to enter emerging analysis

# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
FORECAST_FREQ = "D"          # daily cell-level violation counts
FORECAST_RES = H3_RES_COARSE  # res 8: coarser/denser cells -> better forecast signal
FORECAST_HORIZON = 1          # predict next bucket (1 day ahead)
FORECAST_LAGS = [1, 2, 3, 7, 14, 28]
FORECAST_ROLLING = [3, 7, 14, 28]
FORECAST_EWM_HALFLIFE = 7
FORECAST_EMBARGO = 28         # walk-forward embargo (>= longest window & horizon)
FORECAST_MIN_TRAIN_DAYS = 90  # >= 3 months before first validation fold
HOTSPOT_LABEL_TOP_FRAC = 0.10 # binary "is hotspot" label = top 10% cells by count
LGBM_PARAMS = {
    "objective": "poisson",        # overdispersed, zero-inflated counts
    "metric": "poisson",
    "n_estimators": 3000,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42,        # reproducible builds (docs <-> artifacts stay in sync)
    "deterministic": True,
    "force_row_wise": True,
}
LGBM_EARLY_STOPPING = 100
# Prediction Accuracy Index (PAI): evaluate hotspot capture at top-k% of cells.
PAI_AREA_FRACS = [0.05, 0.20]

# --------------------------------------------------------------------------- #
# Enforcement prioritization & fairness
# --------------------------------------------------------------------------- #
# Exposure-adjusted ranking breaks the patrol -> record -> rank -> patrol loop:
#   adjusted_rate = raw_count / (exposure + alpha),  alpha = median(exposure)
UNDER_ENFORCEMENT_GAP_FLAG = 0.30   # |modeled - observed| percentile gap => blind spot
DISPARATE_IMPACT_FLAG = 0.80        # 4/5ths rule: group_rate/max_rate < 0.8 flagged
STAT_PARITY_FLAG = 0.10             # |P(enforced|A) - P(enforced|B)| > 0.10 flagged

# --------------------------------------------------------------------------- #
# Privacy (India DPDP Act 2023). Public artifacts are H3-only; raw points and
# plates never leave the processing layer.
# --------------------------------------------------------------------------- #
K_ANON_MIN = 5                      # suppress any public cell/bucket with count < 5
HASH_PLATES = True                  # SHA-256 + salt vehicle numbers in any export
PLATE_SALT = "curbiq-dpdp-v1"       # rotate in production via env

# --------------------------------------------------------------------------- #
# BTP real enforcement geography (for "real-world viability" alignment).
# --------------------------------------------------------------------------- #
BTP_ENFORCEMENT_GEOGRAPHY = {
    "corridors": 12, "junctions": 43, "major_roads": 99, "high_density_points": 154,
}

# --------------------------------------------------------------------------- #
# Patrol routing / dispatch optimization
# --------------------------------------------------------------------------- #
PATROL_UNITS = 6                 # enforcement vehicles available per shift
PATROL_SHIFT_HOURS = 4.0         # route-duration budget per unit
PATROL_SPEED_KMPH = 20.0         # assumed urban patrol speed
PATROL_DWELL_MIN = 8.0           # enforcement time spent per stop
PATROL_TOP_K = 60                # candidate stops = top-K priority cells
PATROL_DEPOT = (12.9766, 77.5993)  # dispatch origin (~BTP HQ / Cubbon Park)
PATROL_SHIFT_START = "17:30"     # IST start of the (currently missing) evening shift

LICENSE = "Apache-2.0"

# --------------------------------------------------------------------------- #
# Null sentinels used across the raw CSV
# --------------------------------------------------------------------------- #
NULL_TOKENS = {"", "NULL", "null", "None", "nan", "NaN"}
