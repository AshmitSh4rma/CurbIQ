"""Feature-engineering unit tests (curbiq.features)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from curbiq import config as C
from curbiq.features import (classify_road, max_carriageway_severity,
                             parse_int_list, parse_str_list, peak_overlap,
                             primary_offence, time_of_day_bucket)


# --------------------------------------------------------------------------- #
# parse_int_list / parse_str_list
# --------------------------------------------------------------------------- #
class TestParseIntList:
    def test_json_list(self):
        assert parse_int_list("[112,104]") == [112, 104]

    def test_single_scalar(self):
        assert parse_int_list("113") == [113]

    def test_string_ints_in_list(self):
        assert parse_int_list('["112", "104"]') == [112, 104]

    @pytest.mark.parametrize("raw", ["", "NULL", "null", "None", "nan", "NaN"])
    def test_null_tokens(self, raw):
        assert parse_int_list(raw) == []

    def test_none_and_nan(self):
        assert parse_int_list(None) == []
        assert parse_int_list(float("nan")) == []

    def test_garbage_returns_empty(self):
        assert parse_int_list("not-json-at-all{") == []

    def test_non_int_members_skipped(self):
        assert parse_int_list('[112, "abc", 104]') == [112, 104]


class TestParseStrList:
    def test_json_list(self):
        assert parse_str_list('["NO PARKING"]') == ["NO PARKING"]

    def test_multi(self):
        assert parse_str_list('["NO PARKING", "WRONG PARKING"]') == [
            "NO PARKING", "WRONG PARKING"]

    def test_bare_string_falls_back_to_itself(self):
        # Not valid JSON -> wrapped as a single-element list of the raw string.
        assert parse_str_list("NO PARKING") == ["NO PARKING"]

    @pytest.mark.parametrize("raw", ["", "NULL", "None", "nan"])
    def test_null_tokens(self, raw):
        assert parse_str_list(raw) == []

    def test_none(self):
        assert parse_str_list(None) == []


# --------------------------------------------------------------------------- #
# classify_road on known addresses
# --------------------------------------------------------------------------- #
class TestClassifyRoad:
    @pytest.mark.parametrize(
        "location,expected_class,expected_weight",
        [
            ("Outer Ring Road near KR Puram", "ring_arterial", 1.00),
            ("Sarjapur Ring Road", "ring_arterial", 0.95),
            ("Hebbal Flyover", "grade_separated", 0.95),
            ("Trinity Circle", "junction", 0.90),
            ("MG Main Road", "arterial", 0.85),
            ("Shivajinagar Market", "commercial", 0.80),
            ("Cubbon Park Metro Station", "transit_node", 0.75),
            ("Indiranagar 12th Cross", "local_connector", 0.45),
            ("Koramangala 4th Block Layout", "residential", 0.35),
            ("100 Feet Road", "collector", 0.55),
        ],
    )
    def test_known_addresses(self, location, expected_class, expected_weight):
        cls, weight = classify_road(location)
        assert cls == expected_class
        assert weight == pytest.approx(expected_weight)

    def test_priority_arterial_before_collector(self):
        # "main road" must win over the generic "road" collector pattern.
        assert classify_road("Old Madras Main Road")[0] == "arterial"

    @pytest.mark.parametrize("loc", [None, ""])
    def test_empty_falls_to_default(self, loc):
        cls, weight = classify_road(loc)
        assert cls == C.DEFAULT_ROAD_CLASS
        assert weight == pytest.approx(C.DEFAULT_ROAD_WEIGHT)

    def test_unmatched_falls_to_default(self):
        assert classify_road("zzzzz qqqqq")[0] == C.DEFAULT_ROAD_CLASS


# --------------------------------------------------------------------------- #
# peak_overlap monotonicity / structure
# --------------------------------------------------------------------------- #
class TestPeakOverlap:
    def test_weekday_peak_is_one(self):
        assert peak_overlap(9, False) == 1.0      # morning peak
        assert peak_overlap(18, False) == 1.0     # evening peak

    def test_weekend_peak_is_softer(self):
        assert peak_overlap(9, True) == 0.7
        assert peak_overlap(9, True) < peak_overlap(9, False)

    def test_ordering_peak_gt_shoulder_gt_midday_gt_night(self):
        peak = peak_overlap(9, False)
        shoulder = peak_overlap(7, False)   # m0-1 = 7
        midday = peak_overlap(13, False)
        night = peak_overlap(3, False)
        assert peak > shoulder > midday > night

    def test_bounded_in_unit_interval(self):
        for h in range(24):
            for w in (True, False):
                assert 0.0 <= peak_overlap(h, w) <= 1.0

    def test_night_floor(self):
        assert peak_overlap(2, False) == 0.25


# --------------------------------------------------------------------------- #
# footprint mapping (config-driven, verified through the lookup)
# --------------------------------------------------------------------------- #
class TestFootprint:
    def test_two_wheeler_below_car(self):
        assert C.VEHICLE_FOOTPRINT_PCU["MOTOR CYCLE"] < C.VEHICLE_FOOTPRINT_PCU["CAR"]

    def test_car_is_unit_reference(self):
        assert C.VEHICLE_FOOTPRINT_PCU["CAR"] == 1.0

    def test_heavy_above_car(self):
        assert C.VEHICLE_FOOTPRINT_PCU["LORRY/GOODS VEHICLE"] > C.VEHICLE_FOOTPRINT_PCU["CAR"]
        assert C.VEHICLE_FOOTPRINT_PCU["BUS (BMTC/KSRTC)"] > C.VEHICLE_FOOTPRINT_PCU["CAR"]

    def test_monotone_two_wheeler_to_bus(self):
        order = ["MOPED", "MOTOR CYCLE", "CAR", "VAN", "MINI LORRY",
                 "LORRY/GOODS VEHICLE", "HGV"]
        vals = [C.VEHICLE_FOOTPRINT_PCU[v] for v in order]
        assert vals == sorted(vals)

    def test_unknown_vehicle_falls_to_default(self):
        assert C.VEHICLE_FOOTPRINT_PCU.get("SPACESHIP", C.DEFAULT_FOOTPRINT_PCU) == 1.0


# --------------------------------------------------------------------------- #
# primary_offence severity selection
# --------------------------------------------------------------------------- #
class TestPrimaryOffence:
    def test_picks_highest_severity_parking_code(self):
        # 107 (1.00) beats 113 (0.85) and 105 (0.35).
        assert primary_offence([113, 107, 105]) == 107

    def test_picks_higher_of_two_parking_codes(self):
        # 108 (0.75) beats 105 (0.35).
        assert primary_offence([105, 108]) == 108

    def test_no_parking_codes_returns_first(self):
        # 110 / 116 are not parking offences -> fall back to first code.
        assert primary_offence([110, 116]) == 110

    def test_empty_returns_none(self):
        assert primary_offence([]) is None

    def test_max_carriageway_severity(self):
        assert max_carriageway_severity([113, 107, 105]) == pytest.approx(1.00)
        # non-parking-only -> default severity
        assert max_carriageway_severity([110, 116]) == pytest.approx(C.DEFAULT_SEVERITY)

    def test_max_severity_empty_is_default(self):
        assert max_carriageway_severity([]) == pytest.approx(C.DEFAULT_SEVERITY)


# --------------------------------------------------------------------------- #
# time_of_day_bucket sanity
# --------------------------------------------------------------------------- #
class TestTimeOfDayBucket:
    @pytest.mark.parametrize(
        "hour,bucket",
        [(0, "night"), (5, "night"), (6, "early_morning"), (9, "morning_peak"),
         (13, "midday"), (18, "evening_peak"), (22, "late_evening")],
    )
    def test_buckets(self, hour, bucket):
        assert time_of_day_bucket(hour) == bucket

    def test_all_hours_covered(self):
        assert all(isinstance(time_of_day_bucket(h), str) for h in range(24))
