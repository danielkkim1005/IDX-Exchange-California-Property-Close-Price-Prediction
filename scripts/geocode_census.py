"""
US Census Bureau Bulk Geocoder integration.

Replaces the Nominatim-based geocoder for cases where speed matters more than
match robustness. No API key, no per-second rate limit -- up to 10,000
addresses go in a single HTTP request instead of one request per row.

Tradeoffs versus the Nominatim version:
    - US addresses only (fine for this project, CA-only data).
    - Requires separately-parsed Street / City / State / ZIP columns, not a
      single free-text address string. This is why geocoding has to run
      upstream, before PostalCode and StateOrProvince get dropped from the
      pipeline -- both are still needed here.
    - Match rate on messy or ambiguous addresses tends to be lower than a
      commercial geocoder. Non-matches are left as "geocode_failed" rather
      than retried with a different strategy.
    - No per-row confirmation step -- this module is only meant to run on
      rows already identified as needing geocoding (missing coordinates or
      suspect placeholder clusters), not as a full-dataset confirmation pass
      the way the original Nominatim script was designed to do.

Output contract matches the Nominatim version's geocode_and_validate() so it
is a drop-in replacement: returns a dataframe with columns
    ListingKey, geocoded_lat, geocoded_lon, coord_status
where coord_status is one of:
    'filled_from_geocode'  -- Census returned a match
    'geocode_failed'       -- No match or a tie between multiple matches

Usage:
    import geocode_census as gc
    result = gc.geocode_and_validate_census(to_geocode, checkpoint_path="...")
"""

import io
import os
import time
import pandas as pd
import requests

CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
CENSUS_BENCHMARK = "Public_AR_Current"
MAX_BATCH_SIZE = 10000  # Census's documented hard limit per request
PAUSE_BETWEEN_BATCHES_SECONDS = 1  # courtesy pause for multi-batch runs, not a rate-limit requirement

ID_COL = "ListingKey"
STREET_COL = "UnparsedAddress"
CITY_COL = "City"
STATE_COL = "StateOrProvince"
ZIP_COL = "PostalCode"


def _build_batch_csv(df):
    """
    Builds the exact 5-column, no-header CSV the Census batch endpoint
    requires: Unique ID, Street address, City, State, ZIP.
    """
    batch_df = pd.DataFrame({
        "id": df[ID_COL],
        "street": df[STREET_COL],
        "city": df[CITY_COL],
        "state": df[STATE_COL] if STATE_COL in df.columns else "CA",
        "zip": df[ZIP_COL] if ZIP_COL in df.columns else "",
    })
    csv_buffer = io.StringIO()
    batch_df.to_csv(csv_buffer, index=False, header=False)
    return csv_buffer.getvalue()


def _parse_batch_response(response_text):
    """
    Parses the Census batch geocoder's CSV response.
    Columns: id, input_address, match_status, match_type, matched_address,
    coordinates ("lon,lat" -- note the order), tiger_line_id, side.
    """
    cols = ["id", "input_address", "match_status", "match_type",
            "matched_address", "coordinates", "tiger_line_id", "side"]
    result = pd.read_csv(io.StringIO(response_text), header=None, names=cols, dtype=str)

    def split_coords(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.Series([None, None])
        lon, lat = str(val).split(",")
        return pd.Series([float(lat), float(lon)])

    result[["matched_lat", "matched_lon"]] = result["coordinates"].apply(split_coords)
    result["coord_status"] = result["match_status"].apply(
        lambda s: "filled_from_geocode" if s == "Match" else "geocode_failed"
    )
    return result.rename(columns={"id": ID_COL})[[ID_COL, "matched_lat", "matched_lon", "coord_status"]]


def _geocode_one_batch(batch_df):
    csv_text = _build_batch_csv(batch_df)
    files = {"addressFile": ("batch.csv", csv_text, "text/csv")}
    data = {"benchmark": CENSUS_BENCHMARK}
    response = requests.post(CENSUS_BATCH_URL, files=files, data=data, timeout=180)
    response.raise_for_status()
    parsed = _parse_batch_response(response.text)
    return parsed.rename(columns={"matched_lat": "geocoded_lat", "matched_lon": "geocoded_lon"})


def geocode_and_validate_census(df, checkpoint_path):
    """
    Batches df into <=MAX_BATCH_SIZE chunks, geocodes each via the Census
    Bulk Geocoder, and checkpoints progress per batch (not per row -- each
    batch is already a single fast request, so per-row checkpointing isn't
    needed the way it was for the slow, rate-limited Nominatim version).

    Returns a dataframe with columns: ListingKey, geocoded_lat, geocoded_lon,
    coord_status -- same contract as the Nominatim version's
    geocode_and_validate(), so this is a drop-in replacement.
    """
    if os.path.exists(checkpoint_path):
        done = pd.read_csv(checkpoint_path)
        done_ids = set(done[ID_COL])
    else:
        done = pd.DataFrame({
            ID_COL: pd.Series(dtype="object"),
            "geocoded_lat": pd.Series(dtype="float64"),
            "geocoded_lon": pd.Series(dtype="float64"),
            "coord_status": pd.Series(dtype="object"),
        })
        done_ids = set()

    to_process = df[~df[ID_COL].isin(done_ids)]
    if len(to_process) == 0:
        print("Nothing left to geocode, all rows already checkpointed.")
        return done

    batches = [to_process.iloc[i:i + MAX_BATCH_SIZE] for i in range(0, len(to_process), MAX_BATCH_SIZE)]
    print(f"{len(to_process)} rows to geocode across {len(batches)} batch(es) "
          f"(up to {MAX_BATCH_SIZE} rows/request).")

    all_results = [done]
    for i, batch in enumerate(batches):
        print(f"  batch {i + 1}/{len(batches)} ({len(batch)} rows)...")
        try:
            batch_result = _geocode_one_batch(batch)
        except requests.exceptions.RequestException as e:
            print(f"    request failed: {e} -- stopping here, already-completed "
                  f"batches are checkpointed, rerun this cell to resume.")
            break

        all_results.append(batch_result)
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(checkpoint_path, index=False)

        n_matched = (batch_result["coord_status"] == "filled_from_geocode").sum()
        print(f"    {n_matched}/{len(batch)} matched")

        if i < len(batches) - 1:
            time.sleep(PAUSE_BETWEEN_BATCHES_SECONDS)

    return pd.concat(all_results, ignore_index=True)
