"""
CAPropPredictor -- Streamlit deployment app.

Loads the trained pipeline (models/lgbm_m5.pkl), collects raw property inputs
through a form, reconstructs every engineered feature the model expects using
the exact same logic as wk10_domain_features.ipynb, and predicts ClosePrice.

Company convention followed here (per Streamlit_Deployment_Tutorial.pdf and
Maya Rajesh's reference notebook, Maya_Rajesh_New.ipynb): model.pkl,
requirements.txt, and app.py all sit flat in the same directory/repo root
(the tutorial's GitHub Community Cloud deploy expects exactly this layout:
repo root, branch, "app.py" as the main file path). This app additionally
ships a small `comps_lookup.json` artifact since our pipeline's
ZipMedianPricePerSqft feature needs a fit-on-train lookup table that a flat
model.pkl alone doesn't carry -- our feature set diverges from Maya's
(no KMeans LocationCluster; we use Latitude/Longitude directly plus ZIP/area
comps and distance-to-employment-center), so there's no kmeans_model.pkl to
ship, but the tutorial's Version 2 address-lookup idea still applies: this
app geocodes a typed address to lat/long (and ZIP/city/county where they
match) using geopy/Nominatim -- a free alternative to the tutorial's Google
Maps API that needs no key setup -- so users aren't forced to type
coordinates by hand.

Feature reconstruction follows the AVM best-practices doc's Section 12
deployment guidance: every engineered feature is rebuilt from raw inputs
using training-time logic (not re-derived some other way), and the
constructed feature set is checked against the pipeline's expected schema
before predict() is ever called.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants carried over from wk10_domain_features.ipynb -- keep these in
# lockstep with that notebook. If the notebook changes how a feature is
# built, this app's predictions will silently stop matching training-time
# behavior unless this file is updated too.
# ---------------------------------------------------------------------------

MODEL_PATH = "models/lgbm_m5.pkl"
COMPS_LOOKUP_PATH = "comps_lookup.json"
DROPDOWN_CHOICES_PATH = "dropdown_choices.json"  # small (~96KB) extract of unique category values,
# committed to the repo -- CRMLSCleaned/housing_m5_pre_split.csv itself is gitignored (319K rows,
# too large to ship) and won't exist on a fresh clone or on Streamlit Community Cloud, so dropdown
# population can't depend on it directly.

MAJOR_CA_CENTERS = {
    "Los Angeles": (34.0522, -118.2437),
    "San Francisco": (37.7749, -122.4194),
    "San Diego": (32.7157, -117.1611),
    "San Jose": (37.3382, -121.8863),
    "Sacramento": (38.5816, -121.4944),
    "Fresno": (36.7378, -119.7871),
    "Oakland": (37.8044, -122.2712),
    "Long Beach": (33.7701, -118.1937),
    "Bakersfield": (35.3733, -119.0187),
    "Anaheim": (33.8366, -117.9143),
}

NUMERIC_MEDIAN_COLUMNS = [
    "Latitude", "Longitude", "ViewYN", "PoolPrivateYN", "AttachedGarageYN", "FireplaceYN", "NewConstructionYN",
    "ParkingTotal", "BathroomsTotalInteger", "BedroomsTotal",
    "LivingArea", "LotSizeSquareFeet", "YearBuilt", "Levels", "Stories",
    "PropertyAgeYears", "BedBathRatio",
    "SaleMonthSin", "SaleMonthCos", "MonthsSinceStart",
    "DistanceToNearestMajorCenter", "ZipMedianPricePerSqft",
    "AssociationFeeMissing", "GarageSpacesMissing",
]
NUMERIC_ZERO_FILL_COLUMNS = ["AssociationFee", "GarageSpaces"]
CATEGORICAL_COLUMNS = ["City", "CountyOrParish", "MLSAreaMajor", "SchoolDistrictJoined", "Flooring"]
FEATURE_COLUMNS = NUMERIC_MEDIAN_COLUMNS + NUMERIC_ZERO_FILL_COLUMNS + CATEGORICAL_COLUMNS

# LightGBM (m5), overall test-set metrics -- Deliverables/wk10_m5_metrics_summary.csv
MODEL_METRICS = {"r2": 0.9094, "mae": 152550, "mape": 0.1155, "mdape": 0.0797}


def _get_google_maps_api_key():
    """Reads google_maps_api_key from Streamlit secrets, if configured.
    Locally that's .streamlit/secrets.toml (gitignored -- never commit real
    keys); on Streamlit Community Cloud it's set via the app's
    Settings -> Secrets panel. Returns None (not an error) if unset, so the
    app falls back to the free geocoder rather than breaking."""
    try:
        return st.secrets["google_maps_api_key"]
    except Exception:
        return None


def _geocode_google(address, api_key):
    """Google Maps Geocoding API -- the service the deployment tutorial
    walks through. More accurate/reliable than Nominatim, but requires a
    Google Cloud API key with billing enabled."""
    import requests

    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": api_key, "region": "us"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return None, f"Google geocoding request failed: {exc}"

    status = data.get("status")
    if status != "OK":
        detail = f": {data['error_message']}" if data.get("error_message") else ""
        return None, f"Google geocoding returned '{status}'{detail}"

    result = data["results"][0]
    loc = result["geometry"]["location"]
    # Google returns address parts as a flat list of {long_name, types: [...]}
    # rather than a dict -- index by each component's primary type.
    components = {c["types"][0]: c["long_name"] for c in result.get("address_components", []) if c.get("types")}

    return {
        "latitude": loc["lat"],
        "longitude": loc["lng"],
        "postal_code": components.get("postal_code"),
        "city": components.get("locality") or components.get("sublocality") or components.get("postal_town"),
        "county": components.get("administrative_area_level_2"),
    }, None


def _geocode_nominatim(address):
    """Nominatim (OpenStreetMap) via geopy -- free, no API key required.
    Used automatically whenever no Google Maps key is configured, and kept
    as the always-available fallback either way.

    Retries once on HTTP 429 specifically. Nominatim's free-tier usage policy
    caps clients to roughly 1 request/second per IP; a 429 there means the
    server actively refused the request (not that it was slow), so a longer
    `timeout` would do nothing -- a short backoff-and-retry is the actual fix
    for a rate-limit that's often transient."""
    try:
        from geopy.exc import GeocoderServiceError, GeocoderTimedOut
        from geopy.geocoders import Nominatim
    except ImportError:
        return None, "geopy is not installed (see requirements.txt)."

    geolocator = Nominatim(user_agent="caproppredictor-streamlit-app", timeout=10)

    last_exc = None
    for attempt in range(2):
        try:
            location = geolocator.geocode(address, addressdetails=True, country_codes="us")
            last_exc = None
            break
        except GeocoderServiceError as exc:
            last_exc = exc
            if "429" not in str(exc) or attempt == 1:
                return None, f"geocoding service error: {exc}"
            time.sleep(1.5)  # back off past Nominatim's ~1 req/sec window, then retry once
        except GeocoderTimedOut as exc:
            return None, f"geocoding service error: {exc}"

    if last_exc is not None:
        return None, f"geocoding service error: {last_exc}"

    if location is None:
        return None, "address not found"

    addr = location.raw.get("address", {})
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "postal_code": addr.get("postcode"),
        "city": addr.get("city") or addr.get("town") or addr.get("village"),
        "county": addr.get("county"),
    }, None


def geocode_address(address):
    """Resolves a free-text address to lat/long (+ ZIP/city/county where the
    geocoder reports them). Uses the Google Maps Geocoding API when
    `google_maps_api_key` is configured in Streamlit secrets (see
    _get_google_maps_api_key), otherwise falls back to the free Nominatim
    (OpenStreetMap) geocoder -- so the app works with zero setup, and gets
    more accurate/reliable once a key is added. This exists purely to save
    the user from typing latitude/longitude by hand (per the deployment
    tutorial's Version 2 motivation); every field it fills stays editable,
    and geocoding failure never blocks the manual-entry path below it.

    Returns (result_dict_or_None, error_message_or_None).
    """
    api_key = _get_google_maps_api_key()
    if not api_key:
        return _geocode_nominatim(address)

    result, google_err = _geocode_google(address, api_key)
    if result is not None:
        return result, None

    # Fall through to Nominatim rather than failing outright on a transient
    # Google error (quota hit, network blip, bad key, etc.) -- but don't
    # discard *why* Google failed, or the only error the user ever sees is
    # whichever one Nominatim produces, which hides the real root cause.
    fallback_result, nominatim_err = _geocode_nominatim(address)
    if fallback_result is not None:
        return fallback_result, None
    return None, f"Google: {google_err} | Nominatim fallback: {nominatim_err}"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`. Run this app from the CAPropPredictor project root.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_comps_lookup():
    if not os.path.exists(COMPS_LOOKUP_PATH):
        st.error(
            f"`{COMPS_LOOKUP_PATH}` not found. This is the fit-on-train ZIP/area price-per-sqft "
            "lookup table generated by replaying wk10's Section 14 split against "
            "CRMLSCleaned/housing_m5_pre_split.csv -- it ships alongside app.py and must be present."
        )
        st.stop()
    with open(COMPS_LOOKUP_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dropdown_choices():
    """Populate categorical dropdowns from a small pre-extracted choices file
    (see DROPDOWN_CHOICES_PATH). Purely cosmetic -- TargetEncoder falls back to
    the global mean for any value typed that wasn't seen in training, so this
    doesn't gate what a user can enter; it just makes the common case a pick
    from a list instead of free text."""
    if not os.path.exists(DROPDOWN_CHOICES_PATH):
        return {c: [] for c in CATEGORICAL_COLUMNS + ["PostalCode"]}
    with open(DROPDOWN_CHOICES_PATH) as f:
        return json.load(f)


def zip_median_price_per_sqft(postal_code, mls_area_major, lookup):
    """Mirrors attach_price_per_sqft_comps' fallback chain: ZIP median -> area
    median -> global median. Uses the persisted, fit-on-train lookup table
    rather than the raw dataset, since the real feature at inference time must
    use exactly the same statistics the deployed model was trained against."""
    zip_table = lookup["zip_median_price_per_sqft"]
    area_table = lookup["area_median_price_per_sqft"]
    global_median = lookup["global_median_price_per_sqft"]

    if postal_code is not None and str(postal_code) in zip_table:
        return zip_table[str(postal_code)]
    if mls_area_major is not None and mls_area_major in area_table:
        return area_table[mls_area_major]
    return global_median


def months_since_start(sale_year, sale_month, lookup):
    epoch_year, epoch_month = (int(x) for x in lookup["months_since_start_epoch"].split("-"))
    return (sale_year * 12 + sale_month) - (epoch_year * 12 + epoch_month)


def build_feature_row(inputs, lookup):
    """Reconstructs every model feature from raw form inputs, using the same
    logic as wk10_domain_features.ipynb cells 9-20."""
    row = {}

    # --- passthrough intrinsic/location numerics ---
    row["Latitude"] = inputs["latitude"]
    row["Longitude"] = inputs["longitude"]
    row["ViewYN"] = int(inputs["view"])
    row["PoolPrivateYN"] = int(inputs["pool"])
    row["AttachedGarageYN"] = int(inputs["attached_garage"])
    row["FireplaceYN"] = int(inputs["fireplace"])
    row["NewConstructionYN"] = int(inputs["new_construction"])
    row["ParkingTotal"] = inputs["parking_total"]
    row["BathroomsTotalInteger"] = inputs["bathrooms"]
    row["BedroomsTotal"] = inputs["bedrooms"]
    row["LivingArea"] = inputs["living_area"]
    row["LotSizeSquareFeet"] = inputs["lot_size"]
    row["YearBuilt"] = inputs["year_built"]
    row["Levels"] = inputs["levels"]
    row["Stories"] = inputs["stories"]

    # --- Section 09: PropertyAgeYears / BedBathRatio ---
    row["PropertyAgeYears"] = inputs["sale_year"] - inputs["year_built"]
    row["BedBathRatio"] = (
        inputs["bedrooms"] / inputs["bathrooms"] if inputs["bathrooms"] and inputs["bathrooms"] > 0 else np.nan
    )

    # --- Section 10: cyclical month + time trend ---
    row["SaleMonthSin"] = np.sin(2 * np.pi * inputs["sale_month"] / 12)
    row["SaleMonthCos"] = np.cos(2 * np.pi * inputs["sale_month"] / 12)
    row["MonthsSinceStart"] = months_since_start(inputs["sale_year"], inputs["sale_month"], lookup)

    # --- Section 12: distance to nearest major CA employment center ---
    distances = [
        haversine_miles(inputs["latitude"], inputs["longitude"], lat, lon)
        for lat, lon in MAJOR_CA_CENTERS.values()
    ]
    row["DistanceToNearestMajorCenter"] = float(np.min(distances))

    # --- Section 11: ZIP/area price-per-sqft comps (fit-on-train lookup) ---
    row["ZipMedianPricePerSqft"] = zip_median_price_per_sqft(
        inputs["postal_code"], inputs["mls_area_major"], lookup
    )

    # --- Section 13: missing-indicator flags + zero-fill values ---
    row["AssociationFeeMissing"] = int(inputs["association_fee"] is None)
    row["GarageSpacesMissing"] = int(inputs["garage_spaces"] is None)
    row["AssociationFee"] = inputs["association_fee"] if inputs["association_fee"] is not None else 0
    row["GarageSpaces"] = inputs["garage_spaces"] if inputs["garage_spaces"] is not None else 0

    # --- categoricals (TargetEncoder handles unseen values via global mean) ---
    row["City"] = inputs["city"]
    row["CountyOrParish"] = inputs["county"]
    row["MLSAreaMajor"] = inputs["mls_area_major"]
    row["SchoolDistrictJoined"] = inputs["school_district"]
    row["Flooring"] = inputs["flooring"]

    df = pd.DataFrame([row])

    # The fitted SimpleImputer/StandardScaler steps were fit on training columns
    # that are all float64 (real-world data has NaNs, which forces float dtype
    # even for integer-valued columns like BedroomsTotal). A single-row form
    # input naturally comes back as int64 for whole-number fields, which
    # SimpleImputer.transform rejects as a dtype mismatch against the fitted
    # float64 statistics -- so every numeric feature is cast to float64 here,
    # and every categorical is cast to a plain object dtype with real NaN
    # (not Python None) for missing values, matching training-time dtypes.
    for c in NUMERIC_MEDIAN_COLUMNS + NUMERIC_ZERO_FILL_COLUMNS:
        df[c] = df[c].astype("float64")
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].where(df[c].notna(), np.nan).astype("object")

    return df


def main():
    st.set_page_config(page_title="CA Property Price Predictor", page_icon="\U0001F3E1", layout="centered")
    st.title("CA Property Price Predictor")
    st.caption(
        "Predicts ClosePrice with the LightGBM (m5) pipeline -- see wk10_domain_features.ipynb "
        "for how this model was built and validated."
    )

    with st.expander("Model accuracy on held-out test data", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{MODEL_METRICS['r2']:.3f}")
        c2.metric("MAE", f"${MODEL_METRICS['mae']:,.0f}")
        c3.metric("MAPE", f"{MODEL_METRICS['mape']:.1%}")
        c4.metric("MdAPE", f"{MODEL_METRICS['mdape']:.1%}")
        st.caption(
            "From Deliverables/wk10_m5_metrics_summary.csv, single most-recent-month test cutoff. "
            "See wk10's rolling-origin backtest and wk11's walk-forward stability check for how "
            "consistent this is across other months."
        )

    model = load_model()
    lookup = load_comps_lookup()
    choices = load_dropdown_choices()

    st.header("Property details")
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        living_area = st.number_input("Living area (sqft)", min_value=1, value=1800, step=50)
        lot_size = st.number_input("Lot size (sqft)", min_value=0, value=6000, step=100)
        year_built = st.number_input("Year built", min_value=1800, max_value=2026, value=1990, step=1)
    with col2:
        levels = st.number_input("Levels", min_value=1, max_value=5, value=1, step=1)
        stories = st.number_input("Stories", min_value=1, max_value=5, value=1, step=1)
        parking_total = st.number_input("Total parking spaces", min_value=0, value=2, step=1)
        garage_unknown = st.checkbox("Garage spaces not reported (leave missing)", value=False)
        garage_spaces = (
            None if garage_unknown
            else st.number_input("Garage spaces", min_value=0, value=2, step=1)
        )

    st.subheader("Amenities")
    a1, a2, a3, a4, a5 = st.columns(5)
    view = a1.checkbox("View")
    pool = a2.checkbox("Pool")
    attached_garage = a3.checkbox("Attached garage")
    fireplace = a4.checkbox("Fireplace")
    new_construction = a5.checkbox("New construction")

    st.subheader("HOA")
    hoa_unknown = st.checkbox("Association fee not reported (leave missing)", value=False)
    association_fee = (
        None if hoa_unknown
        else st.number_input("Monthly association fee ($)", min_value=0, value=0, step=10)
    )

    st.header("Location")
    st.caption(
        "Enter a property address to auto-fill latitude/longitude (and ZIP/city/county where they "
        "match our training data) -- per the deployment tutorial's Version 2 pattern, but using a "
        "free geocoder (Nominatim/OpenStreetMap via geopy) instead of the Google Maps API, so no key "
        "setup is required. Everything it fills stays editable, and every field below can also just "
        "be entered by hand."
    )
    address = st.text_input("Property address (optional)", placeholder="6175 Oneida Drive, San Jose, CA 95123")
    if st.button("Look up address"):
        if not address.strip():
            st.warning("Enter an address first.")
        else:
            with st.spinner("Geocoding address..."):
                result, err = geocode_address(address)
            if result is None:
                st.warning(f"Could not geocode that address ({err}). Enter latitude/longitude manually below.")
            else:
                st.session_state["geocoded_lat"] = result["latitude"]
                st.session_state["geocoded_lon"] = result["longitude"]
                st.session_state["geocoded_zip"] = result["postal_code"]
                st.session_state["geocoded_city"] = result["city"]
                st.session_state["geocoded_county"] = result["county"]
                st.success(f"Found: {result['latitude']:.5f}, {result['longitude']:.5f}")
                st.rerun()

    def _select_index(options, geocoded_value, strip_suffixes=()):
        if not geocoded_value:
            return 0
        value = geocoded_value
        for suffix in strip_suffixes:
            if value.endswith(suffix):
                value = value[: -len(suffix)].strip()
        for i, opt in enumerate(options):
            if opt.lower() == value.lower():
                return i
        return 0

    zip_options = [""] + choices.get("PostalCode", [])
    city_options = [""] + choices.get("City", [])
    county_options = [""] + choices.get("CountyOrParish", [])

    col3, col4 = st.columns(2)
    with col3:
        latitude = st.number_input(
            "Latitude", min_value=32.0, max_value=42.5,
            value=float(st.session_state.get("geocoded_lat", 34.05)), format="%.5f",
        )
        longitude = st.number_input(
            "Longitude", min_value=-125.0, max_value=-113.5,
            value=float(st.session_state.get("geocoded_lon", -118.24)), format="%.5f",
        )
        postal_code = st.selectbox(
            "ZIP / postal code", options=zip_options,
            index=_select_index(zip_options, st.session_state.get("geocoded_zip")),
            help="Used for the ZIP-level comps feature. Falls back to MLSAreaMajor, then a global median, if unknown.",
        )
    with col4:
        city = st.selectbox(
            "City", options=city_options,
            index=_select_index(city_options, st.session_state.get("geocoded_city")),
        )
        county = st.selectbox(
            "County", options=county_options,
            index=_select_index(county_options, st.session_state.get("geocoded_county"), strip_suffixes=(" County", " Parish")),
        )
        mls_area_major = st.selectbox("MLS area", options=[""] + choices.get("MLSAreaMajor", []))

    col5, col6 = st.columns(2)
    with col5:
        school_district = st.selectbox("School district", options=[""] + choices.get("SchoolDistrictJoined", []))
    with col6:
        flooring = st.selectbox("Flooring", options=[""] + choices.get("Flooring", []))

    st.header("Sale timing")
    col7, col8 = st.columns(2)
    with col7:
        sale_year = st.number_input("Sale year", min_value=2020, max_value=2030, value=2026, step=1)
    with col8:
        sale_month = st.number_input("Sale month", min_value=1, max_value=12, value=6, step=1)

    if st.button("Predict price", type="primary"):
        if year_built > sale_year:
            st.error("Year built can't be after the sale year.")
            return

        inputs = {
            "bedrooms": bedrooms, "bathrooms": bathrooms, "living_area": living_area,
            "lot_size": lot_size, "year_built": year_built, "levels": levels, "stories": stories,
            "parking_total": parking_total, "garage_spaces": garage_spaces,
            "view": view, "pool": pool, "attached_garage": attached_garage,
            "fireplace": fireplace, "new_construction": new_construction,
            "association_fee": association_fee,
            "latitude": latitude, "longitude": longitude,
            "postal_code": postal_code or None, "city": city or None, "county": county or None,
            "mls_area_major": mls_area_major or None, "school_district": school_district or None,
            "flooring": flooring or None,
            "sale_year": int(sale_year), "sale_month": int(sale_month),
        }

        X = build_feature_row(inputs, lookup)

        # Section 12 (AVM best-practices doc): verify the constructed feature
        # set matches the pipeline's expected schema before predicting.
        missing_cols = set(FEATURE_COLUMNS) - set(X.columns)
        extra_cols = set(X.columns) - set(FEATURE_COLUMNS)
        if missing_cols or extra_cols:
            st.error(
                f"Feature schema mismatch -- missing: {sorted(missing_cols)}, "
                f"unexpected: {sorted(extra_cols)}. This means app.py has drifted from the "
                "training notebook's feature list; do not trust the prediction below."
            )
            return

        X = X[FEATURE_COLUMNS]
        prediction = model.predict(X)[0]

        st.success(f"### Predicted sale price: ${prediction:,.0f}")
        st.caption(
            f"Typical error on held-out test data: MAE ≈ ${MODEL_METRICS['mae']:,.0f}, "
            f"MAPE ≈ {MODEL_METRICS['mape']:.1%}. Treat this as an estimate, not an appraisal."
        )

        with st.expander("Feature values used for this prediction"):
            st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
