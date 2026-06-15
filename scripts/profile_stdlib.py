"""Zero-dependency profiler for the raw BTP parking-violation CSV.

Runs on the system Python (stdlib only) so we can understand the schema
before the analytics venv finishes installing.
"""
import csv
import json
import sys
from collections import Counter, defaultdict

PATH = "/home/ashmit/Claude/CurbIQ/data/raw/police_violations.csv"
csv.field_size_limit(10**7)

NULL_SET = {"", "NULL", "null", "None", "nan"}


def is_null(v):
    return v is None or v.strip() in NULL_SET


def top(counter, k=20):
    return counter.most_common(k)


def main():
    nulls = Counter()
    uniq = defaultdict(set)
    UNIQ_CAP = 300
    vehicle_types = Counter()
    violation_types = Counter()
    offence_codes = Counter()
    stations = Counter()
    val_status = Counter()
    centers = Counter()
    junction_present = Counter()
    months = Counter()
    hours = Counter()
    scita = Counter()
    multi_label = Counter()
    lat_min = lat_max = lon_min = lon_max = None
    bad_geo = 0
    vehnum = set()
    vehnum_overflow = False
    n = 0

    with open(PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            n += 1
            for c in cols:
                v = row.get(c)
                if is_null(v):
                    nulls[c] += 1
                elif len(uniq[c]) < UNIQ_CAP:
                    uniq[c].add(v)

            vt = row.get("vehicle_type")
            if not is_null(vt):
                vehicle_types[vt] += 1

            raw = row.get("violation_type")
            if not is_null(raw):
                try:
                    arr = json.loads(raw)
                    multi_label[len(arr)] += 1
                    for x in arr:
                        violation_types[x] += 1
                except Exception:
                    violation_types["<parse_error>"] += 1

            oc = row.get("offence_code")
            if not is_null(oc):
                try:
                    for x in json.loads(oc):
                        offence_codes[str(x)] += 1
                except Exception:
                    pass

            st = row.get("police_station")
            if not is_null(st):
                stations[st] += 1

            vs = row.get("validation_status")
            val_status[vs if not is_null(vs) else "NULL"] += 1

            cc = row.get("center_code")
            if not is_null(cc):
                centers[cc] += 1

            jn = row.get("junction_name")
            has_junction = not is_null(jn) and jn.strip() != "No Junction"
            junction_present["has_junction" if has_junction else "no_junction"] += 1

            sc = row.get("data_sent_to_scita")
            scita[sc if not is_null(sc) else "NULL"] += 1

            cd = row.get("created_datetime")
            if not is_null(cd):
                months[cd[:7]] += 1
                hours[cd[11:13]] += 1

            try:
                la = float(row["latitude"])
                lo = float(row["longitude"])
                lat_min = la if lat_min is None else min(lat_min, la)
                lat_max = la if lat_max is None else max(lat_max, la)
                lon_min = lo if lon_min is None else min(lon_min, lo)
                lon_max = lo if lon_max is None else max(lon_max, lo)
                if not (12.6 < la < 13.3 and 77.3 < lo < 77.9):
                    bad_geo += 1
            except Exception:
                bad_geo += 1

            vn = row.get("vehicle_number")
            if not is_null(vn) and not vehnum_overflow:
                vehnum.add(vn)
                if len(vehnum) > 400000:
                    vehnum_overflow = True

    print("ROWS:", n)
    print("COLUMNS:", cols)
    print("NULL_COUNTS:", json.dumps(dict(nulls)))
    print("VEHICLE_TYPES:", top(vehicle_types, 40))
    print("VIOLATION_TYPES:", top(violation_types, 40))
    print("OFFENCE_CODES:", top(offence_codes, 40))
    print("MULTI_LABEL_DISTRIBUTION:", dict(sorted(multi_label.items())))
    print("TOP_STATIONS:", top(stations, 60))
    print("TOP_CENTERS:", top(centers, 40))
    print("JUNCTION_PRESENCE:", dict(junction_present))
    print("VALIDATION_STATUS:", dict(val_status))
    print("SCITA_FLAG:", dict(scita))
    print("MONTHS:", sorted(months.items()))
    print("HOURS:", sorted(hours.items()))
    print("GEO_BOUNDS lat[%s,%s] lon[%s,%s] bad_geo=%s" % (lat_min, lat_max, lon_min, lon_max, bad_geo))
    print("VEHICLE_NUMBER_UNIQUE:", len(vehnum), "overflow:", vehnum_overflow)
    print("UNIQUE_COUNTS (capped at %d):" % UNIQ_CAP,
          json.dumps({c: len(s) for c, s in uniq.items()}))


if __name__ == "__main__":
    sys.exit(main())
