"""
CRMLS multi-file geocoding pipeline.

Handles a directory containing a mix of:
    - Monthly files with both versions:      CRMLSSoldYYYYMM.csv + CRMLSSoldYYYYMM_filled.csv
    - Monthly files, non-filled only:        CRMLSSoldYYYYMM.csv
    - Monthly files, filled only:            CRMLSSoldYYYYMM_filled.csv
    - Date-range files, filled only:         CRMLSSoldYYYYMMDD_YYYYMMDD_filled.csv

Per-file treatment:
    - non-filled available (whether or not a _filled sibling exists):
        use the non-filled file as source of truth, ignore its _filled sibling
        entirely (avoids the mojibake risk for no benefit), geocode from scratch.
    - filled-only (no non-filled sibling):
        fix mojibake in text columns first, then geocode as a CONFIRMATION
        check against the coordinates it shipped with (not just a fill-the-gaps pass).

All files are filtered down to the model's scope -- PropertyType == "Residential"
and PropertySubType == "SingleFamilyResidence" -- BEFORE geocoding, since that's
the only subset that matters and it substantially cuts the geocoding workload.

Every row (not just ones missing coordinates) gets geocoded and compared against
its existing lat/lon, producing a coord_status flag:
    'original_confirmed'   original coord exists, geocode agrees (<= tolerance)
    'original_discrepancy' original coord exists, geocode disagrees (> tolerance)
    'filled_from_geocode'  no usable original coord, geocode succeeded
    'geocode_failed'       address could not be resolved

Per-source-file checkpointing means an interrupted run can be resumed without
re-geocoding rows already done, and without re-processing files already finished.

Requirements:
    pip install geopy pandas

Usage:
    python geocode_pipeline.py
"""

import glob
import os
import re
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError

# ---- config ----
DATA_DIR = "C:/Users/kikoh/Desktop/CAPropPredictor/CRMLSData"                                   # directory containing all CRMLSSold*.csv files
OUTPUT_DIR = "./geocoded_output"
CHECKPOINT_DIR = "./geocode_checkpoints"
ID_COL = "ListingKey"
ADDRESS_COL = "UnparsedAddress"
DISCREPANCY_THRESHOLD_KM = 1.0
CHECKPOINT_EVERY = 50
USER_AGENT = "crmls_geocode_validation_daniel_berkeley"

SCOPE_FILTER = {
    "PropertyType": "Residential",
    "PropertySubType": "SingleFamilyResidence",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=10)

# filename patterns
MONTH_RE = re.compile(r"CRMLSSold(\d{6})(_filled)?\.csv$")
RANGE_RE = re.compile(r"CRMLSSold(\d{8})_(\d{8})(_filled)?\.csv$")


def classify_files(data_dir):
    """
    Group files in data_dir by 'period' (month or date range) and decide
    treatment for each period.
    Returns a list of dicts: {period, treatment, source_path}
    """
    files = glob.glob(os.path.join(data_dir, "CRMLSSold*.csv"))
    periods = {}  # period_key -> {'filled': path or None, 'nonfilled': path or None}

    for f in files:
        name = os.path.basename(f)
        m_month = MONTH_RE.match(name)
        m_range = RANGE_RE.match(name)

        if m_range:
            period_key = f"{m_range.group(1)}_{m_range.group(2)}"
            is_filled = bool(m_range.group(3))
        elif m_month:
            period_key = m_month.group(1)
            is_filled = bool(m_month.group(2))
        else:
            print(f"  skipping unrecognized filename: {name}")
            continue

        periods.setdefault(period_key, {"filled": None, "nonfilled": None})
        if is_filled:
            periods[period_key]["filled"] = f
        else:
            periods[period_key]["nonfilled"] = f

    plan = []
    for period_key, versions in periods.items():
        if versions["nonfilled"] is not None:
            plan.append({"period": period_key, "treatment": "nonfilled_source", "source_path": versions["nonfilled"]})
        elif versions["filled"] is not None:
            plan.append({"period": period_key, "treatment": "filled_only", "source_path": versions["filled"]})
    return plan


def fix_mojibake(df):
    """Attempt to reverse UTF-8-decoded-as-Latin-1 corruption on all text columns."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        def try_fix(val):
            if not isinstance(val, str):
                return val
            try:
                fixed = val.encode("latin-1").decode("utf-8")
                return fixed
            except (UnicodeDecodeError, UnicodeEncodeError):
                return val
        df[col] = df[col].apply(try_fix)
    return df


def filter_scope(df):
    mask = pd.Series(True, index=df.index)
    for col, val in SCOPE_FILTER.items():
        mask &= (df[col] == val)
    return df[mask].copy()


def valid_coord(lat, lon):
    """Basic sanity check -- catches swapped lat/lon, placeholder junk (e.g. 0,0 or
    999), and anything outside physically possible ranges before it reaches geopy."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def geocode_and_validate(df, checkpoint_path):
    if os.path.exists(checkpoint_path):
        done = pd.read_csv(checkpoint_path)
        done_ids = set(done[ID_COL])
    else:
        done = pd.DataFrame(columns=[ID_COL, "geocoded_lat", "geocoded_lon", "distance_km", "coord_status"])
        done_ids = set()

    to_process = df[~df[ID_COL].isin(done_ids)]
    results = []
    skipped_transient = 0

    for i, (_, row) in enumerate(to_process.iterrows()):
        address = row.get(ADDRESS_COL)
        orig_lat, orig_lon = row.get("Latitude"), row.get("Longitude")
        geocoded_lat, geocoded_lon, dist, status = None, None, None, "geocode_failed"
        transient_failure = False

        if pd.notna(address):
            try:
                loc = geocode_fn(str(address) + ", CA")
                if loc is not None:
                    geocoded_lat, geocoded_lon = loc.latitude, loc.longitude
            except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
                # connection-related failure -- NOT the address's fault. Don't record
                # this row as done, so it gets retried automatically on the next run.
                print(f"    connection issue on '{address}': {e} -- will retry next run")
                transient_failure = True
            except Exception as e:
                # some other error (bad data, unexpected response, etc.) -- treat as
                # a genuine failure for this address, not a connection problem.
                print(f"    error geocoding '{address}': {e}")

        if transient_failure:
            skipped_transient += 1
            continue  # not added to results -> stays "to do" for next run

        if geocoded_lat is not None:
            geocoded_ok = valid_coord(geocoded_lat, geocoded_lon)
            original_ok = valid_coord(orig_lat, orig_lon)

            if original_ok and geocoded_ok:
                try:
                    dist = geodesic((orig_lat, orig_lon), (geocoded_lat, geocoded_lon)).km
                    status = "original_confirmed" if dist <= DISCREPANCY_THRESHOLD_KM else "original_discrepancy"
                except ValueError as e:
                    # belt-and-suspenders: shouldn't hit this given valid_coord above,
                    # but never let one bad row take down the whole run.
                    print(f"    distance calc failed for '{address}' ({orig_lat}, {orig_lon}) "
                          f"vs ({geocoded_lat}, {geocoded_lon}): {e}")
                    status = "distance_calc_error"
            elif not original_ok and pd.notna(orig_lat) and pd.notna(orig_lon):
                # there WAS an original coordinate, it's just garbage (out of range,
                # likely swapped lat/lon) -- flag it distinctly so it's easy to find
                status = "original_invalid_geocoded_used"
            else:
                status = "filled_from_geocode"

        results.append({
            ID_COL: row[ID_COL], "geocoded_lat": geocoded_lat, "geocoded_lon": geocoded_lon,
            "distance_km": dist, "coord_status": status,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(checkpoint_path, index=False)
            print(f"    checkpointed {len(results)} newly done ({i + 1}/{len(to_process)} attempted)")

    all_results = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    all_results.to_csv(checkpoint_path, index=False)
    if skipped_transient:
        print(f"  {skipped_transient} rows skipped due to connection issues -- re-run this "
              f"script to pick them up (they were NOT checkpointed as done).")
    return df.merge(all_results, on=ID_COL, how="left")


def process_period(period):
    print(f"\n=== Period {period['period']} ({period['treatment']}) ===")

    try:
        df = pd.read_csv(period["source_path"], low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"  {period['source_path']} is empty (no header/rows) -- skipping period.")
        return None

    if len(df) == 0:
        print(f"  {period['source_path']} has a header but 0 rows -- skipping period.")
        return None

    if period["treatment"] == "filled_only":
        df = fix_mojibake(df)

    df = filter_scope(df)
    print(f"  {len(df)} rows in model scope after filtering")

    if len(df) == 0:
        print(f"  no rows left in scope for this period -- skipping geocoding.")
        return None

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{period['period']}_checkpoint.csv")
    df = geocode_and_validate(df, checkpoint_path)

    out_path = os.path.join(OUTPUT_DIR, f"{period['period']}_geocoded.csv")
    df.to_csv(out_path, index=False)
    print(f"  saved {out_path}")
    return df


def main():
    plan = classify_files(DATA_DIR)
    print(f"Found {len(plan)} periods to process:")
    for p in plan:
        print(f"  {p['period']}: {p['treatment']} <- {p['source_path']}")

    all_dfs = [process_period(p) for p in plan]
    all_dfs = [d for d in all_dfs if d is not None]
    if not all_dfs:
        print("No periods produced any rows -- nothing to combine.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=[ID_COL], keep="first")
    after = len(combined)
    if before != after:
        print(f"\nDropped {before - after} duplicate ListingKey rows across files (safety net).")

    combined_path = os.path.join(OUTPUT_DIR, "model_dataset_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nFinal combined dataset: {len(combined)} rows -> {combined_path}")
    print(combined["coord_status"].value_counts())


if __name__ == "__main__":
    main()