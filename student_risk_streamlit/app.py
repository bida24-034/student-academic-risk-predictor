import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path
from difflib import SequenceMatcher
from io import BytesIO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Insight",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS / MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_risk_model.pkl"

CUTOFF_DAY = 42


@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)

    return (
        package["model"],
        package["threshold"],
        package["features"]
    )


try:
    model, threshold, model_features = load_model()
    model_loaded = True
    model_error = None

except Exception as e:
    model = None
    threshold = 0.4311
    model_features = []
    model_loaded = False
    model_error = str(e)


# ============================================================
# MODEL FEATURE DEFINITIONS
# ============================================================

EXPECTED_MODEL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "module_presentation_length",
    "total_clicks_week6",
    "active_days_week6",
    "resources_accessed_week6",
    "avg_score_week6",
    "min_score_week6",
    "max_score_week6",
    "assessments_submitted_week6",
    "late_submissions_week6",
    "has_score_week6",
    "date_registration",
    "registration_date_missing"
]

if not model_features:
    model_features = EXPECTED_MODEL_FEATURES.copy()


CATEGORICAL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]


NUMERIC_FEATURES = [
    feature
    for feature in model_features
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# RAW OULAD FILES
# ============================================================

RAW_OULAD_FILES = {
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv"
}


# ============================================================
# FLEXIBLE COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "id_student": [
        "id_student",
        "student_id",
        "studentid",
        "student_number",
        "studentnumber",
        "learner_id",
        "learnerid",
        "student_no",
        "studentno"
    ],

    "code_module": [
        "code_module",
        "module",
        "module_code",
        "course",
        "course_code",
        "subject",
        "subject_code",
        "programme",
        "program"
    ],

    "code_presentation": [
        "code_presentation",
        "presentation",
        "semester",
        "term",
        "session",
        "academic_term",
        "intake"
    ],

    "gender": [
        "gender",
        "sex"
    ],

    "region": [
        "home_area",
        "home_region",
        "residential_area",
        "student_region",
        "region",
        "location",
        "area",
        "district"
    ],

    "highest_education": [
        "education_background",
        "educational_background",
        "prior_education",
        "entry_qualification",
        "highest_education",
        "education",
        "education_level",
        "highest_qualification",
        "qualification"
    ],

    "imd_band": [
        "imd_band",
        "imd",
        "deprivation_band",
        "socioeconomic_band",
        "socioeconomic_status"
    ],

    "age_band": [
        "age_band",
        "age_group",
        "agegroup",
        "age_range"
    ],

    "num_of_prev_attempts": [
        "previous_course_attempts",
        "prior_course_attempts",
        "num_of_prev_attempts",
        "previous_attempts",
        "prev_attempts",
        "attempts",
        "number_of_attempts"
    ],

    "studied_credits": [
        "studied_credits",
        "credits",
        "credit_load",
        "registered_credits"
    ],

    "disability": [
        "special_support",
        "support_needs",
        "additional_support",
        "learning_support",
        "disability",
        "disabled",
        "disability_status"
    ],

    "module_presentation_length": [
        "course_length_days",
        "module_length_days",
        "term_length_days",
        "module_presentation_length",
        "course_length",
        "module_length",
        "presentation_length",
        "course_duration",
        "duration_days"
    ],

    "total_clicks_week6": [
        "week6_platform_clicks",
        "week6_clicks",
        "six_week_clicks",
        "early_platform_clicks",
        "total_clicks_week6",
        "total_clicks",
        "clicks",
        "click_count",
        "vle_clicks",
        "lms_clicks",
        "online_clicks"
    ],

    "active_days_week6": [
        "week6_active_days",
        "six_week_active_days",
        "early_active_days",
        "active_days_week6",
        "active_days",
        "days_active",
        "learning_days",
        "online_days",
        "login_days"
    ],

    "resources_accessed_week6": [
        "week6_resources_used",
        "week6_resources_accessed",
        "six_week_resources_used",
        "resources_accessed_week6",
        "resources_accessed",
        "resource_count",
        "resources_used",
        "unique_resources",
        "learning_resources"
    ],

    "avg_score_week6": [
        "week6_average_mark",
        "week6_average_score",
        "week6_avg_mark",
        "early_average_mark",
        "avg_score_week6",
        "average_score",
        "avg_score",
        "average_mark",
        "avg_mark",
        "mean_score",
        "mean_mark",
        "assessment_average"
    ],

    "min_score_week6": [
        "week6_lowest_mark",
        "week6_minimum_mark",
        "week6_min_score",
        "min_score_week6",
        "minimum_score",
        "min_score",
        "minimum_mark",
        "min_mark"
    ],

    "max_score_week6": [
        "week6_highest_mark",
        "week6_maximum_mark",
        "week6_max_score",
        "max_score_week6",
        "maximum_score",
        "max_score",
        "maximum_mark",
        "max_mark"
    ],

    "assessments_submitted_week6": [
        "week6_assessments_submitted",
        "week6_assignments_submitted",
        "early_submissions",
        "assessments_submitted_week6",
        "assessments_submitted",
        "assignments_submitted",
        "submissions",
        "submission_count",
        "assignments_done"
    ],

    "late_submissions_week6": [
        "week6_late_submissions",
        "week6_late_assignments",
        "early_late_submissions",
        "late_submissions_week6",
        "late_submissions",
        "late_assignments",
        "late_submission_count"
    ],

    "has_score_week6": [
        "week6_score_available",
        "week6_has_score",
        "early_score_available",
        "has_score_week6",
        "has_score",
        "score_available"
    ],

    "date_registration": [
        "date_registration",
        "registration_date",
        "days_registration",
        "registration_day"
    ],

    "registration_date_missing": [
        "registration_date_missing",
        "registration_missing"
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "dataset_loaded": False,
    "uploaded_dataset": None,
    "dataset_name": None,
    "input_mode": None,
    "preprocessing_report": None,
    "predictions_generated": False,
    "prediction_results": None,
    "sidebar_navigation": "Dashboard",
    "external_mapping": None,
    "mapping_confirmed": False
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# NAVIGATION HELPERS
# ============================================================

def go_to_page(page_name):
    st.session_state.sidebar_navigation = page_name


def reset_prediction_state():
    st.session_state.predictions_generated = False
    st.session_state.prediction_results = None


def clear_active_dataset():
    st.session_state.dataset_loaded = False
    st.session_state.uploaded_dataset = None
    st.session_state.dataset_name = None
    st.session_state.preprocessing_report = None
    st.session_state.predictions_generated = False
    st.session_state.prediction_results = None
    st.session_state.external_mapping = None
    st.session_state.mapping_confirmed = False
    st.session_state["validation_issues"] = None


# ============================================================
# DATA HELPERS
# ============================================================

def normalise_name(value):
    value = str(value).strip().lower()

    for character in [
        " ", "-", ".", "/", "\\", "(", ")", "[", "]"
    ]:
        value = value.replace(character, "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def numeric_series(series):
    return pd.to_numeric(
        series.replace("?", np.nan),
        errors="coerce"
    )


def safe_numeric_value(value, default=0):
    result = pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]

    if pd.isna(result):
        return default

    return float(result)


def similarity(a, b):
    a_norm = normalise_name(a)
    b_norm = normalise_name(b)

    sequence_score = SequenceMatcher(
        None,
        a_norm,
        b_norm
    ).ratio()

    a_tokens = set(a_norm.split("_"))
    b_tokens = set(b_norm.split("_"))

    token_score = (
        len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
        if a_tokens and b_tokens
        else 0
    )

    return max(sequence_score, token_score)


# ============================================================
# AUTOMATIC COLUMN MAPPING
# ============================================================

def suggest_column_mapping(columns):

    columns = list(columns)

    normalized_columns = {
        normalise_name(column): column
        for column in columns
    }

    suggestions = {}

    targets = [
        "id_student"
    ] + list(model_features)

    for target in targets:

        aliases = COLUMN_ALIASES.get(
            target,
            [target]
        )

        # Always include the model field name itself as a valid alias.
        aliases = list(dict.fromkeys(
            [target] + list(aliases)
        ))

        match = None

        # Exact normalized alias match
        for alias in aliases:

            alias_norm = normalise_name(alias)

            if alias_norm in normalized_columns:
                match = normalized_columns[alias_norm]
                break

        # Fuzzy match only when reasonably strong
        if match is None:

            best_column = None
            best_score = 0

            for column in columns:

                for alias in aliases:

                    score = similarity(
                        column,
                        alias
                    )

                    if score > best_score:
                        best_score = score
                        best_column = column

            if best_score >= 0.82:
                match = best_column

        suggestions[target] = match

    return suggestions


# ============================================================
# GENERIC DATA CLEANING
# ============================================================

def basic_clean_external_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    # Convert ? into missing values
    cleaned = cleaned.replace(
        ["?", "NA", "N/A", "null", "NULL", ""],
        np.nan
    )

    # Remove exact duplicate rows
    duplicate_rows = int(
        cleaned.duplicated().sum()
    )

    cleaned = (
        cleaned
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Strip whitespace from text columns
    for column in cleaned.select_dtypes(
        include="object"
    ).columns:

        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

        cleaned[column] = cleaned[column].replace(
            {
                "<NA>": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "": np.nan
            }
        )

        # Convert pandas nullable StringDtype back to ordinary object.
        # sklearn's fitted SimpleImputer expects np.nan, not pd.NA.
        cleaned[column] = cleaned[column].astype(object)
        cleaned[column] = cleaned[column].where(
            pd.notna(cleaned[column]),
            np.nan
        )

    report = {
        "Original rows": original_rows,
        "Exact duplicate rows removed": duplicate_rows,
        "Rows after basic cleaning": len(cleaned),
        "Uploaded columns": cleaned.shape[1]
    }

    return cleaned, report


# ============================================================
# PREPARED ANALYTICAL DATA CLEANING
# ============================================================

def clean_prepared_analytical_dataset(df):
    """
    Clean a dataset that already follows the model's analytical schema.
    The function standardises missing values/types and removes duplicates,
    but does not fabricate unavailable student information.
    """
    cleaned = df.copy()
    original_rows = len(cleaned)
    original_missing = int(cleaned.isna().sum().sum())

    # Standard missing-value markers
    cleaned = cleaned.replace(
        ["?", "NA", "N/A", "n/a", "null", "NULL", "None", ""],
        np.nan
    )

    # Remove exact duplicate rows
    exact_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Trim text values without converting genuine missing values into strings
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace("", np.nan)

    # Normalise categories that are known from the training structure
    if "gender" in cleaned.columns:
        cleaned["gender"] = (
            cleaned["gender"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({
                "female": "F", "f": "F", "woman": "F",
                "male": "M", "m": "M", "man": "M"
            })
            .astype(object)
        )
        cleaned["gender"] = cleaned["gender"].where(
            pd.notna(cleaned["gender"]), np.nan
        )

    if "disability" in cleaned.columns:
        cleaned["disability"] = (
            cleaned["disability"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({
                "yes": "Y", "y": "Y", "true": "Y", "1": "Y",
                "no": "N", "n": "N", "false": "N", "0": "N"
            })
            .astype(object)
        )
        cleaned["disability"] = cleaned["disability"].where(
            pd.notna(cleaned["disability"]), np.nan
        )

    if "imd_band" in cleaned.columns:
        cleaned["imd_band"] = cleaned["imd_band"].replace(
            {"10-20": "10-20%"}
        )

    # Model numeric fields should be numeric. Invalid text becomes NaN so the
    # fitted model pipeline can use its training-time imputation.
    for feature in NUMERIC_FEATURES:
        if feature in cleaned.columns:
            cleaned[feature] = pd.to_numeric(
                cleaned[feature], errors="coerce"
            )

    # Binary indicator consistency where the source information exists.
    if "avg_score_week6" in cleaned.columns:
        if "has_score_week6" not in cleaned.columns:
            cleaned["has_score_week6"] = (
                cleaned["avg_score_week6"].notna().astype(int)
            )
        else:
            hs = pd.to_numeric(cleaned["has_score_week6"], errors="coerce")
            invalid_hs = ~hs.isin([0, 1]) & hs.notna()
            hs.loc[invalid_hs] = np.nan
            cleaned["has_score_week6"] = hs

    if "date_registration" in cleaned.columns:
        if "registration_date_missing" not in cleaned.columns:
            cleaned["registration_date_missing"] = (
                cleaned["date_registration"].isna().astype(int)
            )
        else:
            rm = pd.to_numeric(
                cleaned["registration_date_missing"], errors="coerce"
            )
            invalid_rm = ~rm.isin([0, 1]) & rm.notna()
            rm.loc[invalid_rm] = np.nan
            cleaned["registration_date_missing"] = rm

    # Enrollment-level duplicate key check/removal.
    key = [
        c for c in
        ["id_student", "code_module", "code_presentation"]
        if c in cleaned.columns
    ]
    duplicate_keys_removed = 0
    if len(key) == 3:
        duplicate_keys_removed = int(
            cleaned.duplicated(subset=key, keep="first").sum()
        )
        if duplicate_keys_removed:
            cleaned = (
                cleaned
                .drop_duplicates(subset=key, keep="first")
                .reset_index(drop=True)
            )

    report = {
        "Original rows": original_rows,
        "Exact duplicate rows removed": exact_duplicates,
        "Duplicate enrollment keys removed": duplicate_keys_removed,
        "Rows after cleaning": len(cleaned),
        "Missing values before cleaning": original_missing,
        "Missing values after cleaning": int(cleaned.isna().sum().sum()),
    }

    return cleaned, report


# ============================================================
# BUILD STANDARD MODEL SCHEMA FROM MAPPING
# ============================================================

def build_standard_dataset(
    source_df,
    mapping
):

    standardized = pd.DataFrame(
        index=source_df.index
    )

    # --------------------------------------------------------
    # STUDENT ID
    # --------------------------------------------------------

    id_source = mapping.get(
        "id_student"
    )

    if id_source and id_source in source_df.columns:

        standardized["id_student"] = (
            source_df[id_source]
        )

    else:

        standardized["id_student"] = (
            np.arange(
                1,
                len(source_df) + 1
            )
        )

    # --------------------------------------------------------
    # DIRECTLY MAPPED MODEL FEATURES
    # --------------------------------------------------------

    directly_mapped = []

    unavailable = []

    for feature in model_features:

        source_column = mapping.get(
            feature
        )

        if (
            source_column
            and
            source_column in source_df.columns
        ):

            standardized[feature] = (
                source_df[source_column]
            )

            directly_mapped.append(
                feature
            )

        else:

            standardized[feature] = np.nan

            unavailable.append(
                feature
            )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        if feature in standardized.columns:

            standardized[feature] = (
                pd.to_numeric(
                    standardized[feature],
                    errors="coerce"
                )
            )

    # --------------------------------------------------------
    # DERIVE HAS SCORE
    # --------------------------------------------------------

    derived = []

    if (
        mapping.get("has_score_week6") is None
        and
        "avg_score_week6"
        in standardized.columns
    ):

        standardized[
            "has_score_week6"
        ] = (
            standardized[
                "avg_score_week6"
            ]
            .notna()
            .astype(int)
        )

        derived.append(
            "has_score_week6"
        )

        if "has_score_week6" in unavailable:
            unavailable.remove(
                "has_score_week6"
            )

    # --------------------------------------------------------
    # DERIVE REGISTRATION MISSING FLAG
    # --------------------------------------------------------

    if (
        mapping.get(
            "registration_date_missing"
        ) is None
        and
        mapping.get(
            "date_registration"
        ) is not None
    ):

        standardized[
            "registration_date_missing"
        ] = (
            standardized[
                "date_registration"
            ]
            .isna()
            .astype(int)
        )

        derived.append(
            "registration_date_missing"
        )

        if (
            "registration_date_missing"
            in unavailable
        ):
            unavailable.remove(
                "registration_date_missing"
            )

    # --------------------------------------------------------
    # PRESERVE USEFUL ORIGINAL FIELDS
    # --------------------------------------------------------

    for optional_column in [
        "final_result",
        "at_risk"
    ]:

        if optional_column in source_df.columns:

            standardized[
                optional_column
            ] = source_df[
                optional_column
            ]

    # --------------------------------------------------------
    # NORMALISE SIMPLE CATEGORIES
    # --------------------------------------------------------

    if "gender" in standardized.columns:

        gender_map = {
            "female": "F",
            "f": "F",
            "woman": "F",
            "male": "M",
            "m": "M",
            "man": "M"
        }

        standardized["gender"] = (
            standardized["gender"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(gender_map)
        )

    if "disability" in standardized.columns:

        disability_map = {
            "yes": "Y",
            "y": "Y",
            "true": "Y",
            "1": "Y",
            "no": "N",
            "n": "N",
            "false": "N",
            "0": "N"
        }

        standardized["disability"] = (
            standardized["disability"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(disability_map)
        )

    # --------------------------------------------------------
    # GUARANTEE COMPLETE STANDARDIZED SCHEMA
    # --------------------------------------------------------

    # The model pipeline expects all 22 column names to exist.
    # If a source field truly has no equivalent, keep that column as NaN
    # and report it as unavailable instead of inventing student information.
    for feature in model_features:
        if feature not in standardized.columns:
            standardized[feature] = np.nan

            if feature not in unavailable:
                unavailable.append(feature)

    # Predictable output order: identifier first, then model features.
    ordered_columns = ["id_student"] + [
        feature
        for feature in model_features
        if feature != "id_student"
    ]

    extra_columns = [
        column
        for column in standardized.columns
        if column not in ordered_columns
    ]

    standardized = standardized[
        [
            column
            for column in ordered_columns
            if column in standardized.columns
        ]
        + extra_columns
    ]

    # --------------------------------------------------------
    # COMPATIBILITY INFORMATION
    # --------------------------------------------------------

    mapped_count = len(
        set(directly_mapped)
    )

    derived_count = len(
        set(derived)
    )

    available_count = (
        mapped_count
        +
        derived_count
    )

    compatibility_percentage = (
        available_count
        /
        len(model_features)
        *
        100
    )

    compatibility = {
        "directly_mapped": sorted(
            set(directly_mapped)
        ),
        "derived": sorted(
            set(derived)
        ),
        "unavailable": sorted(
            set(unavailable)
        ),
        "available_count": available_count,
        "required_count": len(
            model_features
        ),
        "percentage": compatibility_percentage
    }

    return standardized, compatibility



# ============================================================
# DATA VALIDATION / PREMIUM CHART HELPERS
# ============================================================

VALIDATION_RANGES = {
    "num_of_prev_attempts": (0, None),
    "studied_credits": (0, None),
    "module_presentation_length": (1, None),
    "total_clicks_week6": (0, None),
    "active_days_week6": (0, None),
    "resources_accessed_week6": (0, None),
    "avg_score_week6": (0, 100),
    "min_score_week6": (0, 100),
    "max_score_week6": (0, 100),
    "assessments_submitted_week6": (0, None),
    "late_submissions_week6": (0, None),
    "has_score_week6": (0, 1),
    "registration_date_missing": (0, 1),
}

VALID_CATEGORIES = {
    "gender": {"F", "M"},
    "disability": {"Y", "N"},
}


def validate_model_dataset(df):
    """Transparent data-quality checks. Flags suspicious values; does not invent replacements."""
    issues = []

    for column, (minimum, maximum) in VALIDATION_RANGES.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        invalid = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid |= values < minimum

        if maximum is not None:
            invalid |= values > maximum

        count = int(invalid.fillna(False).sum())
        if count:
            expected = (
                f"{minimum} to {maximum}"
                if minimum is not None and maximum is not None
                else f">= {minimum}" if minimum is not None
                else f"<= {maximum}"
            )
            issues.append({
                "Field": column,
                "Issue": "Value outside expected range",
                "Affected rows": count,
                "Expected": expected,
            })

    for column, allowed in VALID_CATEGORIES.items():
        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str).str.strip()
        invalid_values = sorted(set(values) - allowed)

        if invalid_values:
            count = int(values.isin(invalid_values).sum())
            preview = ", ".join(invalid_values[:6])
            if len(invalid_values) > 6:
                preview += ", …"

            issues.append({
                "Field": column,
                "Issue": f"Unexpected categories: {preview}",
                "Affected rows": count,
                "Expected": ", ".join(sorted(allowed)),
            })

    score_columns = [
        column for column in
        ["avg_score_week6", "min_score_week6", "max_score_week6"]
        if column in df.columns
    ]

    if len(score_columns) == 3:
        min_s = pd.to_numeric(df["min_score_week6"], errors="coerce")
        avg_s = pd.to_numeric(df["avg_score_week6"], errors="coerce")
        max_s = pd.to_numeric(df["max_score_week6"], errors="coerce")
        inconsistent = (
            min_s.notna() & avg_s.notna() & max_s.notna()
            & ((min_s > avg_s) | (avg_s > max_s))
        )
        count = int(inconsistent.sum())
        if count:
            issues.append({
                "Field": "score_week6",
                "Issue": "min / average / max score relationship is inconsistent",
                "Affected rows": count,
                "Expected": "min <= average <= max",
            })

    return pd.DataFrame(
        issues,
        columns=["Field", "Issue", "Affected rows", "Expected"]
    )


def semantic_bar_chart(data, category_col, value_col, height=300, neutral=False):
    """Premium dark bar chart. Risk categories use red/green semantics."""
    plot_df = data[[category_col, value_col]].copy()
    plot_df[category_col] = plot_df[category_col].astype(str)

    if neutral:
        color = alt.value("#7EA7E8")
    else:
        color = alt.Color(
            f"{category_col}:N",
            scale=alt.Scale(
                domain=["At Risk", "Not At Risk"],
                range=["#FF626E", "#43D39E"]
            ),
            legend=None
        )

    chart = (
        alt.Chart(plot_df)
        .mark_bar(
            cornerRadiusTopLeft=7,
            cornerRadiusTopRight=7,
            opacity=0.96
        )
        .encode(
            x=alt.X(
                f"{category_col}:N",
                title=None,
                axis=alt.Axis(labelAngle=0, labelColor="#8997AA")
            ),
            y=alt.Y(
                f"{value_col}:Q",
                title=None,
                axis=alt.Axis(gridColor="#202A38", labelColor="#8997AA")
            ),
            color=color,
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title=category_col),
                alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.2f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            domain=False,
            ticks=False,
            titleColor="#8997AA",
            labelFontSize=11
        )
    )

    st.altair_chart(chart, use_container_width=True)


def probability_band_chart(series, height=300):
    plot_df = series.rename_axis("Probability Band").reset_index(name="Records")

    chart = (
        alt.Chart(plot_df)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            color="#7EA7E8"
        )
        .encode(
            x=alt.X(
                "Probability Band:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelAngle=-35, labelColor="#8997AA")
            ),
            y=alt.Y(
                "Records:Q",
                title=None,
                axis=alt.Axis(gridColor="#202A38", labelColor="#8997AA")
            ),
            tooltip=[
                alt.Tooltip("Probability Band:N"),
                alt.Tooltip("Records:Q", format=",")
            ]
        )
        .properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(domain=False, ticks=False)
    )

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# RAW OULAD PROCESSING
# ============================================================

def preprocess_raw_oulad(raw_files):

    assessments = raw_files[
        "assessments.csv"
    ].copy()

    courses = raw_files[
        "courses.csv"
    ].copy()

    student_assessment = raw_files[
        "studentAssessment.csv"
    ].copy()

    student_info = raw_files[
        "studentInfo.csv"
    ].copy()

    student_registration = raw_files[
        "studentRegistration.csv"
    ].copy()

    student_vle = raw_files[
        "studentVle.csv"
    ].copy()

    vle = raw_files[
        "vle.csv"
    ].copy()

    report = {}

    # --------------------------------------------------------
    # ORIGINAL COUNTS
    # --------------------------------------------------------

    report["Raw student records"] = len(
        student_info
    )

    report["Raw VLE rows"] = len(
        student_vle
    )

    report[
        "Raw assessment submissions"
    ] = len(
        student_assessment
    )

    # --------------------------------------------------------
    # REPLACE ?
    # --------------------------------------------------------

    tables = [
        assessments,
        courses,
        student_assessment,
        student_info,
        student_registration,
        student_vle,
        vle
    ]

    for table in tables:
        table.replace(
            "?",
            np.nan,
            inplace=True
        )

    # --------------------------------------------------------
    # REMOVE EXACT VLE DUPLICATES
    # --------------------------------------------------------

    duplicate_vle = int(
        student_vle.duplicated().sum()
    )

    student_vle = (
        student_vle
        .drop_duplicates()
        .copy()
    )

    report[
        "studentVle duplicates removed"
    ] = duplicate_vle

    # --------------------------------------------------------
    # CLEAN IMD
    # --------------------------------------------------------

    if "imd_band" in student_info.columns:

        student_info[
            "imd_band"
        ] = (
            student_info[
                "imd_band"
            ]
            .fillna("Unknown")
            .replace(
                {
                    "10-20": "10-20%"
                }
            )
        )

    # --------------------------------------------------------
    # NUMERIC TYPES
    # --------------------------------------------------------

    assessments["date"] = numeric_series(
        assessments["date"]
    )

    student_assessment[
        "score"
    ] = numeric_series(
        student_assessment["score"]
    )

    student_assessment[
        "date_submitted"
    ] = numeric_series(
        student_assessment[
            "date_submitted"
        ]
    )

    student_registration[
        "date_registration"
    ] = numeric_series(
        student_registration[
            "date_registration"
        ]
    )

    student_vle["date"] = numeric_series(
        student_vle["date"]
    )

    student_vle[
        "sum_click"
    ] = numeric_series(
        student_vle["sum_click"]
    )

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    key = [
        "code_module",
        "code_presentation",
        "id_student"
    ]

    # ========================================================
    # VLE — FIRST SIX WEEKS
    # ========================================================

    vle_week6 = student_vle[
        student_vle["date"]
        <= CUTOFF_DAY
    ].copy()

    report[
        "VLE rows through Day 42"
    ] = len(vle_week6)

    vle_features = (
        vle_week6
        .groupby(
            key,
            as_index=False
        )
        .agg(
            total_clicks_week6=(
                "sum_click",
                "sum"
            ),
            active_days_week6=(
                "date",
                "nunique"
            ),
            resources_accessed_week6=(
                "id_site",
                "nunique"
            )
        )
    )

    # ========================================================
    # ASSESSMENTS — FIRST SIX WEEKS
    # ========================================================

    assessment_details = (
        student_assessment
        .merge(
            assessments[
                [
                    "code_module",
                    "code_presentation",
                    "id_assessment",
                    "assessment_type",
                    "date",
                    "weight"
                ]
            ],
            on="id_assessment",
            how="left"
        )
    )

    assessment_week6 = (
        assessment_details[
            (
                assessment_details[
                    "date"
                ] <= CUTOFF_DAY
            )
            &
            (
                assessment_details[
                    "date_submitted"
                ] <= CUTOFF_DAY
            )
        ]
        .copy()
    )

    report[
        "Assessment submissions through Day 42"
    ] = len(
        assessment_week6
    )

    assessment_week6[
        "late_submission"
    ] = (
        assessment_week6[
            "date_submitted"
        ]
        >
        assessment_week6[
            "date"
        ]
    ).astype(int)

    assessment_features = (
        assessment_week6
        .groupby(
            key,
            as_index=False
        )
        .agg(
            avg_score_week6=(
                "score",
                "mean"
            ),
            min_score_week6=(
                "score",
                "min"
            ),
            max_score_week6=(
                "score",
                "max"
            ),
            assessments_submitted_week6=(
                "id_assessment",
                "count"
            ),
            late_submissions_week6=(
                "late_submission",
                "sum"
            )
        )
    )

    # ========================================================
    # MASTER DATASET
    # ========================================================

    master = student_info.copy()

    master = master.merge(
        courses[
            [
                "code_module",
                "code_presentation",
                "module_presentation_length"
            ]
        ],
        on=[
            "code_module",
            "code_presentation"
        ],
        how="left"
    )

    master = master.merge(
        vle_features,
        on=key,
        how="left"
    )

    master = master.merge(
        assessment_features,
        on=key,
        how="left"
    )

    master = master.merge(
        student_registration[
            key
            +
            ["date_registration"]
        ],
        on=key,
        how="left"
    )

    # --------------------------------------------------------
    # NO EARLY VLE ACTIVITY = ZERO ACTIVITY
    # --------------------------------------------------------

    for column in [
        "total_clicks_week6",
        "active_days_week6",
        "resources_accessed_week6"
    ]:

        master[column] = (
            master[column]
            .fillna(0)
        )

    # --------------------------------------------------------
    # ASSESSMENT COUNTS
    # --------------------------------------------------------

    master[
        "has_score_week6"
    ] = (
        master[
            "avg_score_week6"
        ]
        .notna()
        .astype(int)
    )

    master[
        "assessments_submitted_week6"
    ] = (
        master[
            "assessments_submitted_week6"
        ]
        .fillna(0)
    )

    master[
        "late_submissions_week6"
    ] = (
        master[
            "late_submissions_week6"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # REGISTRATION FLAG
    # --------------------------------------------------------

    master[
        "registration_date_missing"
    ] = (
        master[
            "date_registration"
        ]
        .isna()
        .astype(int)
    )

    # --------------------------------------------------------
    # HISTORICAL LABEL
    # --------------------------------------------------------

    if "final_result" in master.columns:

        master["at_risk"] = (
            master[
                "final_result"
            ]
            .isin(
                [
                    "Fail",
                    "Withdrawn"
                ]
            )
            .astype(int)
        )

    report[
        "Analytical records created"
    ] = len(master)

    report[
        "Unique students"
    ] = master[
        "id_student"
    ].nunique()

    report[
        "Duplicate analytical keys"
    ] = int(
        master.duplicated(
            subset=key
        ).sum()
    )

    return master, report


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #070B12;
    --surface: #0D131D;
    --surface-2: #111925;
    --surface-3: #151F2D;
    --line: #222D3C;
    --line-soft: rgba(126,167,232,.12);
    --text: #F4F7FB;
    --muted: #8997AA;
    --muted-2: #5E6D80;
    --blue: #7EA7E8;
    --blue-2: #9BC0FF;
    --green: #43D39E;
    --red: #FF626E;
    --amber: #F2B84B;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 72% -8%, rgba(73,120,197,.18), transparent 26%),
        radial-gradient(circle at 20% 8%, rgba(74,104,162,.07), transparent 20%),
        linear-gradient(180deg, #090D14 0%, var(--bg) 55%, #060A10 100%);
    color: var(--text);
}

.block-container {
    max-width: 1540px;
    padding: 1.15rem 2.35rem 4rem 2.35rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(17,25,37,.98), rgba(8,13,21,.99));
    border-right: 1px solid rgba(126,167,232,.12);
    box-shadow: 18px 0 45px rgba(0,0,0,.20);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: .8rem;
}

.sidebar-brand {
    padding: 10px 8px 23px 8px;
}

.brand-row {
    display:flex;
    align-items:center;
    gap:12px;
}

.brand-logo {
    width:39px;
    height:39px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#07101B;
    font-size:.86rem;
    font-weight:850;
    background:linear-gradient(145deg,#D5E5FF 0%,#8DB4F1 45%,#5F84BE 100%);
    box-shadow:0 10px 28px rgba(84,135,212,.28), inset 0 1px 0 rgba(255,255,255,.55);
}

.brand-name {
    color:#F5F8FC;
    font-size:.90rem;
    font-weight:760;
    letter-spacing:-.015em;
}

.brand-subtitle {
    color:#65758A;
    font-size:.59rem;
    margin-top:2px;
}

.sidebar-label {
    color:#526176;
    font-size:.55rem;
    font-weight:760;
    text-transform:uppercase;
    letter-spacing:.11em;
    margin:5px 8px 9px 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap:4px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius:10px;
    padding:7px 9px;
    transition:all .18s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:rgba(126,167,232,.06);
}

.sidebar-model {
    margin-top:24px;
    padding:14px;
    background:linear-gradient(145deg,rgba(20,30,44,.95),rgba(13,20,30,.96));
    border:1px solid rgba(126,167,232,.14);
    border-radius:13px;
    box-shadow:0 14px 35px rgba(0,0,0,.18);
}

.sidebar-model-status {
    color:#C3CFDC;
    font-size:.63rem;
    font-weight:700;
}

.sidebar-model-dot {
    display:inline-block;
    width:7px;
    height:7px;
    background:var(--green);
    border-radius:50%;
    margin-right:7px;
    box-shadow:0 0 12px rgba(67,211,158,.65);
}

.sidebar-model-detail {
    color:#68788D;
    font-size:.57rem;
    line-height:1.8;
    margin-top:8px;
}

.topbar {
    min-height:45px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:0 2px 14px 2px;
    border-bottom:1px solid rgba(126,167,232,.10);
    margin-bottom:24px;
}

.topbar-page {
    color:#75859A;
    font-size:.63rem;
    font-weight:650;
}

.topbar-status {
    color:#78889C;
    font-size:.59rem;
    padding:7px 10px;
    border:1px solid rgba(126,167,232,.12);
    background:rgba(16,24,35,.58);
    border-radius:999px;
}

.topbar-dot {
    color:var(--green);
    margin-right:6px;
    text-shadow:0 0 10px rgba(67,211,158,.7);
}

.page-title {
    color:#F7F9FC;
    font-size:1.55rem;
    font-weight:780;
    letter-spacing:-.035em;
    line-height:1.15;
}

.page-subtitle {
    color:#75859A;
    font-size:.70rem;
    line-height:1.65;
    margin-top:6px;
    margin-bottom:24px;
    max-width:820px;
}

.quick-header {
    color:#9AA8B9;
    font-size:.61rem;
    font-weight:760;
    text-transform:uppercase;
    letter-spacing:.095em;
    margin-top:28px;
    margin-bottom:11px;
}

.panel,
.info-card,
.upload-ready-card {
    padding:19px;
    border-radius:14px;
    background:
        linear-gradient(145deg,rgba(20,29,42,.98),rgba(13,20,30,.98));
    border:1px solid rgba(126,167,232,.13);
    box-shadow:0 18px 45px rgba(0,0,0,.17), inset 0 1px 0 rgba(255,255,255,.018);
}

.panel-title,
.info-title {
    color:#D5DEE9;
    font-size:.75rem;
    font-weight:720;
}

.panel-subtitle {
    color:#63748A;
    font-size:.59rem;
    margin-top:4px;
}

.empty-page {
    margin-top:26px;
    min-height:260px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    background:
        radial-gradient(circle at 50% 0%,rgba(126,167,232,.09),transparent 45%),
        linear-gradient(145deg,#111925,#0C121B);
    border:1px solid rgba(126,167,232,.13);
    border-radius:15px;
    text-align:center;
    box-shadow:0 20px 55px rgba(0,0,0,.18);
}

.empty-page-title {
    color:#D4DCE7;
    font-size:.85rem;
    font-weight:720;
}

.empty-page-text {
    color:#68788D;
    font-size:.66rem;
    margin-top:7px;
    max-width:560px;
    line-height:1.65;
}

.status-card {
    min-height:118px;
    padding:16px 17px;
    border-radius:14px;
    border:1px solid rgba(126,167,232,.13);
    background:
        linear-gradient(145deg,rgba(20,29,42,.98),rgba(13,20,30,.98));
    box-shadow:0 16px 38px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.02);
    transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}

.status-card:hover {
    transform:translateY(-2px);
    border-color:rgba(126,167,232,.24);
    box-shadow:0 20px 48px rgba(0,0,0,.22);
}

.status-red {
    border-color:rgba(255,98,110,.25);
    background:
        radial-gradient(circle at 100% 0%,rgba(255,98,110,.09),transparent 38%),
        linear-gradient(145deg,#171A24,#10151E);
}

.status-green {
    border-color:rgba(67,211,158,.23);
    background:
        radial-gradient(circle at 100% 0%,rgba(67,211,158,.08),transparent 38%),
        linear-gradient(145deg,#141C24,#10161E);
}

.status-amber {
    border-color:rgba(242,184,75,.23);
    background:
        radial-gradient(circle at 100% 0%,rgba(242,184,75,.07),transparent 38%),
        linear-gradient(145deg,#171B23,#10151E);
}

.status-blue {
    border-color:rgba(126,167,232,.24);
    background:
        radial-gradient(circle at 100% 0%,rgba(126,167,232,.10),transparent 40%),
        linear-gradient(145deg,#151D2A,#101721);
}

.status-label {
    color:#7D8DA1;
    font-size:.55rem;
    font-weight:680;
    text-transform:uppercase;
    letter-spacing:.075em;
}

.status-value {
    color:#F6F8FB;
    font-size:1.38rem;
    font-weight:790;
    letter-spacing:-.035em;
    margin-top:8px;
}

.status-note {
    color:#637389;
    font-size:.57rem;
    margin-top:5px;
}

.risk-text { color:var(--red); font-weight:780; }
.safe-text { color:var(--green); font-weight:780; }
.amber-text { color:var(--amber); font-weight:780; }
.blue-text { color:var(--blue-2); font-weight:780; }
.mapping-good { color:var(--green); }
.mapping-missing { color:var(--amber); }

.small-note {
    color:#718096;
    font-size:.62rem;
    line-height:1.7;
    padding:10px 12px;
    border-left:2px solid rgba(126,167,232,.35);
    background:rgba(126,167,232,.035);
    border-radius:0 9px 9px 0;
}

.model-row {
    display:flex;
    justify-content:space-between;
    gap:20px;
    padding:10px 0;
    border-bottom:1px solid rgba(126,167,232,.09);
    color:#74849A;
    font-size:.63rem;
}

.model-row-value {
    color:#C3CEDB;
    font-weight:650;
}

div[data-testid="stMetric"] {
    min-height:110px;
    background:
        linear-gradient(145deg,rgba(20,29,42,.98),rgba(13,20,30,.98));
    border:1px solid rgba(126,167,232,.13);
    padding:15px 16px;
    border-radius:14px;
    box-shadow:0 15px 38px rgba(0,0,0,.15);
}

div[data-testid="stMetric"] label {
    color:#7D8DA1 !important;
    font-size:.60rem !important;
}

div[data-testid="stMetricValue"] {
    color:#F5F8FC !important;
    font-weight:760 !important;
    letter-spacing:-.03em;
}

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {
    border-radius:10px;
    border:1px solid rgba(126,167,232,.18);
    background:linear-gradient(180deg,#182334,#121B28);
    color:#DCE6F2;
    min-height:39px;
    font-weight:650;
    box-shadow:0 8px 20px rgba(0,0,0,.12);
    transition:all .18s ease;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {
    border-color:rgba(126,167,232,.42);
    color:#FFFFFF;
    transform:translateY(-1px);
    box-shadow:0 11px 26px rgba(0,0,0,.20), 0 0 22px rgba(85,130,200,.08);
}

div[data-testid="stFileUploader"] {
    padding:10px;
    border-radius:14px;
    background:rgba(14,21,31,.60);
    border:1px solid rgba(126,167,232,.10);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius:10px !important;
    background:#101824 !important;
    border-color:#263244 !important;
}

[data-testid="stDataFrame"] {
    border:1px solid rgba(126,167,232,.11);
    border-radius:13px;
    overflow:hidden;
    box-shadow:0 16px 40px rgba(0,0,0,.13);
}

[data-testid="stAlert"] {
    border-radius:12px;
}

hr {
    border-color:rgba(126,167,232,.09) !important;
}

#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

[data-testid="stHeader"] {
    background:transparent;
}

::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-track { background:#090E15; }
::-webkit-scrollbar-thumb {
    background:#263244;
    border-radius:999px;
    border:2px solid #090E15;
}
::-webkit-scrollbar-thumb:hover { background:#35455D; }

@media (max-width: 900px) {
    .block-container {
        padding-left:1rem;
        padding-right:1rem;
    }
    .page-title { font-size:1.3rem; }
}

/* Final UI polish */
.page-subtitle { margin-bottom: 1.35rem !important; }
.quick-header { margin-top: 1.65rem !important; margin-bottom: 0.8rem !important; }
div[data-testid="stMetric"] { min-height: 112px; }
div[data-testid="stMetricValue"] { font-size: 2rem !important; line-height: 1.08 !important; }
.status-value { font-size: 2rem !important; line-height: 1.08 !important; }
.prediction-risk { color: #ff6b6b; font-weight: 700; }
.prediction-safe { color: #51cf66; font-weight: 700; }


.intervention-loading-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F3F6FA;
}
.intervention-loading-note {
    margin-top: 7px;
    font-size: 0.86rem;
    color: #AEB9C7;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
<div class="brand-row">
<div class="brand-logo">S</div>
<div>
<div class="brand-name">Student Insight</div>
<div class="brand-subtitle">Academic Risk Intelligence</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Workspace</div>',
        unsafe_allow_html=True
    )

    pages = [
        "Dashboard",
        "Upload Data",
        "Data Quality",
        "Predictions",
        "Students",
        "Interventions",
        "Analytics",
        "Model Information"
    ]

    page = st.radio(
        "Navigation",
        pages,
        key="sidebar_navigation",
        label_visibility="collapsed"
    )

    if model_loaded:

        st.markdown(
            f"""
<div class="sidebar-model">
<div class="sidebar-model-status">
<span class="sidebar-model-dot"></span>
Prediction model online
</div>
<div class="sidebar-model-detail">
Logistic Regression<br>
{len(model_features)} input features<br>
Threshold · {threshold:.4f}
</div>
</div>
""",
            unsafe_allow_html=True
        )

    else:
        st.error(
            "Prediction model unavailable"
        )


# ============================================================
# TOP BAR
# ============================================================

status_text = (
    "System ready"
    if model_loaded
    else "Model unavailable"
)

st.markdown(
    f"""
<div class="topbar">
<div class="topbar-page">{page}</div>
<div class="topbar-status">
<span class="topbar-dot">●</span>
{status_text}
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="page-title">Academic Risk Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Monitor early academic risk, student engagement and intervention readiness.
</div>
""",
        unsafe_allow_html=True
    )

    dataset = (
        st.session_state.uploaded_dataset
        if st.session_state.dataset_loaded
        else None
    )

    predictions = (
        st.session_state.prediction_results
        if st.session_state.predictions_generated
        else None
    )

    total_records = (
        len(dataset)
        if dataset is not None
        else None
    )

    at_risk = (
        int(
            predictions[
                "predicted_at_risk"
            ].sum()
        )
        if predictions is not None
        else None
    )

    risk_rate = (
        at_risk
        /
        len(predictions)
        *
        100
        if predictions is not None
        and len(predictions) > 0
        else None
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
<div class="status-card status-blue">
<div class="status-label">Student Records</div>
<div class="status-value">
{f"{total_records:,}" if total_records is not None else "—"}
</div>
<div class="status-note">
{"Active analytical dataset" if total_records is not None else "Awaiting dataset"}
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
<div class="status-card status-red">
<div class="status-label">Predicted At Risk</div>
<div class="status-value risk-text">
{f"{at_risk:,}" if at_risk is not None else "—"}
</div>
<div class="status-note">
{"Requires review" if at_risk is not None else "Awaiting predictions"}
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
<div class="status-card status-amber">
<div class="status-label">Predicted Risk Rate</div>
<div class="status-value amber-text">
{f"{risk_rate:.1f}%" if risk_rate is not None else "—"}
</div>
<div class="status-note">
Current active dataset
</div>
</div>
""",
            unsafe_allow_html=True
        )

    if predictions is not None:

        st.markdown(
            '<div class="quick-header">Risk Distribution</div>',
            unsafe_allow_html=True
        )

        chart_df = (
            predictions[
                "predicted_risk"
            ]
            .value_counts()
            .reindex(
                [
                    "At Risk",
                    "Not At Risk"
                ],
                fill_value=0
            )
            .rename_axis(
                "Prediction"
            )
            .reset_index(
                name="Records"
            )
        )

        semantic_bar_chart(
            chart_df,
            "Prediction",
            "Records",
            height=280
        )

    else:

        st.markdown(
            """
<div class="empty-page">
<div class="empty-page-title">No prediction analysis available</div>
<div class="empty-page-text">
Upload student data and generate predictions to populate the risk dashboard.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="quick-header">Quick Actions</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.button(
            "＋ Upload Data",
            use_container_width=True,
            on_click=go_to_page,
            args=("Upload Data",)
        )

    with q2:
        st.button(
            "Run Predictions",
            use_container_width=True,
            on_click=go_to_page,
            args=("Predictions",)
        )

    with q3:
        st.button(
            "Review Interventions",
            use_container_width=True,
            on_click=go_to_page,
            args=("Interventions",)
        )


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "Upload Data":

    st.markdown(
        '<div class="page-title">Upload & Prepare Data</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Use a prepared analytical dataset or the university's raw
student-system files that follow the structure used to train the model.
</div>
""",
        unsafe_allow_html=True
    )

    if st.session_state.dataset_loaded:
        active_name = st.session_state.dataset_name or "Active student dataset"
        active_mode = st.session_state.input_mode or "Prepared dataset"

        st.markdown(
            f"""
<div class="upload-ready-card">
<b>Active Data</b><br>
{active_name}
</div>
<div style="height: 14px;"></div>
""",
            unsafe_allow_html=True
        )

        mem1, mem2 = st.columns([1, 1])

        with mem1:
            st.button(
                "✓ Saved in Session",
                disabled=True,
                use_container_width=True,
                key="session_saved_indicator"
            )

        with mem2:
            if st.button(
                "Delete Active Data",
                use_container_width=True,
                key="delete_active_data"
            ):
                clear_active_dataset()
                st.session_state.pop("prepared_signature", None)
                st.session_state.pop("raw_signature", None)
                st.rerun()

        st.caption(
            "Session storage keeps the active dataset while you move "
            "between pages. It is not permanent database storage."
        )

        st.markdown(
            "<div style='height: 18px;'></div>",
            unsafe_allow_html=True
        )

    upload_mode_options = [
        "Prepared Analytical Dataset",
        "Raw Student Data Files"
    ]

    remembered_upload_mode = st.session_state.get(
        "last_upload_mode",
        "Prepared Analytical Dataset"
    )

    upload_mode = st.radio(
        "Data source",
        upload_mode_options,
        index=(
            upload_mode_options.index(remembered_upload_mode)
            if remembered_upload_mode in upload_mode_options
            else 0
        ),
        horizontal=True
    )

    st.session_state.last_upload_mode = upload_mode

    # ========================================================
    # PREPARED DATA
    # ========================================================

    if upload_mode == "Prepared Analytical Dataset":

        st.info(
            "Use this option when the CSV already contains the model's "
            "analytical features. If the file still contains duplicates, "
            "unclean values or inconsistent types, you can clean it here "
            "before prediction."
        )

        uploaded_file = st.file_uploader(
            "Upload prepared CSV",
            type=["csv"],
            key="prepared_upload"
        )

        if uploaded_file is not None:
            try:
                signature = (uploaded_file.name, uploaded_file.size)

                if st.session_state.get("prepared_signature") != signature:
                    uploaded_file.seek(0)
                    original_df = pd.read_csv(uploaded_file)

                    st.session_state["prepared_original_df"] = original_df.copy()
                    st.session_state.uploaded_dataset = original_df.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_loaded = True
                    st.session_state.input_mode = "Prepared Analytical Dataset"
                    st.session_state.preprocessing_report = None
                    st.session_state.prepared_signature = signature
                    st.session_state["prepared_cleaned"] = False
                    reset_prediction_state()

                df = st.session_state.uploaded_dataset

                validation_issues = validate_model_dataset(df)
                exact_duplicates = int(df.duplicated().sum())

                enrollment_key = [
                    "id_student", "code_module", "code_presentation"
                ]
                duplicate_keys = (
                    int(df.duplicated(subset=enrollment_key).sum())
                    if all(c in df.columns for c in enrollment_key)
                    else 0
                )

                missing = [
                    feature
                    for feature in model_features
                    if feature not in df.columns
                ]

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Records", f"{len(df):,}")
                with c2:
                    st.metric("Exact Duplicates", f"{exact_duplicates:,}")
                with c3:
                    st.metric("Data Issues", f"{len(validation_issues):,}")
                with c4:
                    st.metric(
                        "Model Features",
                        f"{len(model_features)-len(missing)}/{len(model_features)}"
                    )

                needs_cleaning = (
                    exact_duplicates > 0
                    or duplicate_keys > 0
                    or not validation_issues.empty
                )

                if needs_cleaning and not st.session_state.get("prepared_cleaned", False):
                    st.warning(
                        "This prepared dataset still has data-quality issues. "
                        "You can clean it before continuing."
                    )

                    if st.button(
                        "Clean Prepared Dataset",
                        type="primary",
                        key="clean_prepared_dataset"
                    ):
                        cleaned_df, cleaning_report = (
                            clean_prepared_analytical_dataset(
                                st.session_state["prepared_original_df"]
                            )
                        )

                        st.session_state.uploaded_dataset = cleaned_df
                        st.session_state.preprocessing_report = cleaning_report
                        st.session_state["prepared_cleaned"] = True
                        reset_prediction_state()
                        st.rerun()

                elif st.session_state.get("prepared_cleaned", False):
                    st.success(
                        "Prepared dataset cleaned. Data Quality will show the "
                        "remaining checks before prediction."
                    )

                    report = st.session_state.preprocessing_report or {}
                    if report:
                        st.dataframe(
                            pd.DataFrame(
                                list(report.items()),
                                columns=["Cleaning Step", "Result"]
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                    st.download_button(
                        "↓ Download Cleaned Prepared Dataset",
                        df.to_csv(index=False).encode("utf-8"),
                        "cleaned_prepared_analytical_dataset.csv",
                        "text/csv",
                        key="download_cleaned_prepared"
                    )

                if not missing:
                    st.success(
                        "Required model columns are present."
                    )
                    st.button(
                        "Continue to Data Quality →",
                        on_click=go_to_page,
                        args=("Data Quality",),
                        key="prepared_to_quality"
                    )
                else:
                    st.error(
                        "This dataset is missing required model features. "
                        "Cleaning cannot invent missing model columns."
                    )
                    st.write(missing)

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:
                st.error(f"Dataset could not be read: {e}")


    # ========================================================
    # RAW OULAD
    # ========================================================

    else:

        st.info(
            "Upload the seven raw university student-data CSV files. "
            "They must follow the same table and column structure used "
            "to train this system. Records and values may change."
        )

        uploads = st.file_uploader(
            "Upload raw student-data CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="oulad_upload"
        )

        if uploads:

            names = {
                file.name
                for file in uploads
            }

            missing_files = (
                RAW_OULAD_FILES
                -
                names
            )

            file_status = pd.DataFrame(
                [
                    {
                        "File": filename,
                        "Status": (
                            "Ready"
                            if filename in names
                            else "Missing"
                        )
                    }
                    for filename
                    in sorted(
                        RAW_OULAD_FILES
                    )
                ]
            )

            st.dataframe(
                file_status,
                use_container_width=True,
                hide_index=True
            )

            if missing_files:

                st.warning(
                    "Missing: "
                    +
                    ", ".join(
                        sorted(
                            missing_files
                        )
                    )
                )

            else:

                if st.button(
                    "Clean & Build Analytical Dataset",
                    type="primary"
                ):

                    try:

                        with st.spinner(
                            "Cleaning OULAD data and creating Week-6 features..."
                        ):

                            raw_files = {}

                            for file in uploads:

                                file.seek(0)

                                raw_files[
                                    file.name
                                ] = pd.read_csv(
                                    file
                                )

                            master, report = (
                                preprocess_raw_oulad(
                                    raw_files
                                )
                            )

                            missing = [
                                feature
                                for feature in model_features
                                if feature
                                not in master.columns
                            ]

                            if missing:

                                st.error(
                                    "Processing completed but "
                                    "required model features are missing."
                                )

                                st.write(
                                    missing
                                )

                            else:

                                st.session_state.uploaded_dataset = master
                                st.session_state.dataset_name = "processed_student_week6.csv"
                                st.session_state.dataset_loaded = True
                                st.session_state.input_mode = "Raw Student Data Files"
                                st.session_state.preprocessing_report = report

                                reset_prediction_state()

                                st.success(
                                    "OULAD processing completed successfully."
                                )

                    except Exception as e:

                        st.error(
                            "Raw OULAD preprocessing failed."
                        )

                        st.exception(e)

        if (
            st.session_state.dataset_loaded
            and
            st.session_state.input_mode
            == "Raw Student Data Files"
        ):

            master = (
                st.session_state.uploaded_dataset
            )

            report = (
                st.session_state.preprocessing_report
                or {}
            )

            st.markdown(
                '<div class="quick-header">Processed Dataset</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Analytical Records",
                    f"{len(master):,}"
                )

            with c2:
                st.metric(
                    "Unique Students",
                    f"{master['id_student'].nunique():,}"
                )

            with c3:
                st.metric(
                    "Model Features",
                    f"{len(model_features)}/{len(model_features)}"
                )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Processing Step": report.keys(),
                        "Result": report.values()
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            st.dataframe(
                master.head(10),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "↓ Download Cleaned Analytical Dataset",
                master.to_csv(
                    index=False
                ).encode("utf-8"),
                "processed_student_week6.csv",
                "text/csv"
            )

            st.button(
                "Continue to Data Quality →",
                on_click=go_to_page,
                args=("Data Quality",)
            )



# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.markdown(
        '<div class="page-title">Data Quality</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Review completeness, model compatibility and preprocessing quality
before academic risk prediction.
</div>
""",
        unsafe_allow_html=True
    )

    if not st.session_state.dataset_loaded:

        st.markdown(
            """
<div class="empty-page">
<div class="empty-page-title">No active dataset</div>
<div class="empty-page-text">
Upload and prepare student data before running data-quality checks.
</div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='height: 24px;'></div>",
            unsafe_allow_html=True
        )

        st.button(
            "Upload Data →",
            on_click=go_to_page,
            args=("Upload Data",)
        )

    else:

        df = (
            st.session_state.uploaded_dataset
        )

        missing_values = int(
            df.isna().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        available_features = sum(
            feature in df.columns
            for feature in model_features
        )

        # Composite enrollment key used throughout the analytical dataset.
        key_columns = [
            column for column in
            ["id_student", "code_module", "code_presentation"]
            if column in df.columns
        ]

        duplicate_key_rows = (
            int(df.duplicated(subset=key_columns).sum())
            if len(key_columns) == 3
            else 0
        )

        unique_students = (
            int(df["id_student"].nunique(dropna=True))
            if "id_student" in df.columns
            else 0
        )

        # Validate the dtypes expected by the deployed model.
        categorical_expected = [
            "code_module", "code_presentation", "gender", "region",
            "highest_education", "imd_band", "age_band", "disability"
        ]
        numerical_expected = [
            feature for feature in model_features
            if feature not in categorical_expected
        ]

        dtype_rows = []
        dtype_pass_count = 0

        for feature in model_features:
            if feature not in df.columns:
                dtype_rows.append({
                    "Feature": feature,
                    "Expected Type": "Missing",
                    "Detected Type": "—",
                    "Type Check": "Fail"
                })
                continue

            detected = str(df[feature].dtype)

            if feature in categorical_expected:
                valid_type = (
                    pd.api.types.is_object_dtype(df[feature])
                    or pd.api.types.is_string_dtype(df[feature])
                    or pd.api.types.is_categorical_dtype(df[feature])
                )
                expected = "Categorical / text"
            else:
                converted = pd.to_numeric(df[feature], errors="coerce")
                original_non_missing = int(df[feature].notna().sum())
                converted_non_missing = int(converted.notna().sum())
                valid_type = (
                    original_non_missing == 0
                    or converted_non_missing == original_non_missing
                )
                expected = "Numeric"

            if valid_type:
                dtype_pass_count += 1

            dtype_rows.append({
                "Feature": feature,
                "Expected Type": expected,
                "Detected Type": detected,
                "Type Check": "Pass" if valid_type else "Review"
            })

        # Basic validity/range checks. These assess structural plausibility,
        # not whether a university's source values are factually correct.
        validity_checks = []

        def add_range_check(feature, lower=None, upper=None):
            if feature not in df.columns:
                return
            values = pd.to_numeric(df[feature], errors="coerce")
            invalid = pd.Series(False, index=df.index)
            if lower is not None:
                invalid = invalid | (values < lower)
            if upper is not None:
                invalid = invalid | (values > upper)
            validity_checks.append({
                "Check": feature,
                "Rule": (
                    f">= {lower}" if upper is None
                    else f"{lower} to {upper}" if lower is not None
                    else f"<= {upper}"
                ),
                "Issues": int(invalid.fillna(False).sum()),
                "Status": "Pass" if int(invalid.fillna(False).sum()) == 0 else "Review"
            })

        add_range_check("total_clicks_week6", 0, None)
        add_range_check("active_days_week6", 0, None)
        add_range_check("resources_accessed_week6", 0, None)
        add_range_check("avg_score_week6", 0, 100)
        add_range_check("min_score_week6", 0, 100)
        add_range_check("max_score_week6", 0, 100)
        add_range_check("assessments_submitted_week6", 0, None)
        add_range_check("late_submissions_week6", 0, None)
        add_range_check("has_score_week6", 0, 1)
        add_range_check("registration_date_missing", 0, 1)

        validity_issue_count = sum(
            row["Issues"] for row in validity_checks
        )

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            st.metric("Records", f"{len(df):,}")
        with q2:
            st.metric("Missing Values", f"{missing_values:,}")
        with q3:
            st.metric("Duplicate Rows", f"{duplicate_rows:,}")
        with q4:
            st.metric(
                "Model Schema",
                f"{available_features}/{len(model_features)}"
            )

        q5, q6, q7, q8 = st.columns(4)

        with q5:
            st.metric("Unique Students", f"{unique_students:,}")
        with q6:
            st.metric("Duplicate Enrollment Keys", f"{duplicate_key_rows:,}")
        with q7:
            st.metric(
                "Correct Data Types",
                f"{dtype_pass_count}/{len(model_features)}"
            )
        with q8:
            if st.session_state.input_mode == "Raw Student Data Files":
                total_accuracy_checks = (
                    len(model_features)
                    + len(validity_checks)
                    + 2
                )
                passed_accuracy_checks = (
                    dtype_pass_count
                    + sum(
                        1 for row in validity_checks
                        if row["Status"] == "Pass"
                    )
                    + (1 if duplicate_rows == 0 else 0)
                    + (1 if duplicate_key_rows == 0 else 0)
                )
                data_accuracy_pct = (
                    passed_accuracy_checks
                    / total_accuracy_checks
                    * 100
                    if total_accuracy_checks
                    else 0
                )
                st.metric(
                    "Accuracy",
                    f"{data_accuracy_pct:.1f}%"
                )
            else:
                st.metric("Accuracy", "—")

        if st.session_state.input_mode == "Raw Student Data Files":
            st.caption(
                "Accuracy is a post-cleaning data-quality score based on "
                "required data types, plausible value ranges, duplicate-row "
                "checks and enrollment-key uniqueness. It measures structural "
                "accuracy/validity of the raw-data pipeline; it does not verify "
                "source values against an external university ground-truth system."
            )
        else:
            st.caption(
                "Accuracy is shown as — for an already prepared analytical "
                "dataset because the app did not observe its original raw-data "
                "cleaning process."
            )

        if (
            st.session_state.input_mode == "Raw Student Data Files"
            and st.session_state.preprocessing_report
        ):
            st.markdown(
                '<div class="quick-header">Raw-File Cleaning Summary</div>',
                unsafe_allow_html=True
            )
            cleaning_report_df = pd.DataFrame(
                {
                    "Cleaning / Processing Check":
                        st.session_state.preprocessing_report.keys(),
                    "Result":
                        st.session_state.preprocessing_report.values()
                }
            )
            st.dataframe(
                cleaning_report_df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            '<div class="quick-header">Data Type Validation</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame(dtype_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="quick-header">Accuracy & Validity Checks</div>',
            unsafe_allow_html=True
        )

        if validity_checks:
            st.dataframe(
                pd.DataFrame(validity_checks),
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            '<div class="quick-header">Feature Completeness</div>',
            unsafe_allow_html=True
        )

        feature_quality = []

        for feature in model_features:

            if feature not in df.columns:

                feature_quality.append(
                    {
                        "Feature": feature,
                        "Available": "No",
                        "Missing Values": "—",
                        "Missing %": "—"
                    }
                )

            else:

                missing = int(
                    df[feature]
                    .isna()
                    .sum()
                )

                missing_pct = (
                    missing
                    /
                    len(df)
                    *
                    100
                    if len(df)
                    else 0
                )

                feature_quality.append(
                    {
                        "Feature": feature,
                        "Available": "Yes",
                        "Missing Values": missing,
                        "Missing %":
                            f"{missing_pct:.1f}%"
                    }
                )

        st.dataframe(
            pd.DataFrame(
                feature_quality
            ),
            use_container_width=True,
            hide_index=True
        )

        missing_columns = [
            feature
            for feature in model_features
            if feature not in df.columns
        ]

        if missing_columns:

            st.error(
                "Required model columns are missing."
            )

            st.write(
                missing_columns
            )

        else:

            st.success(
                "The standardized dataset contains all "
                "22 model input columns."
            )

            if (
                st.session_state.input_mode
                == "Prepared Analytical Dataset"
            ):

                compatibility = (
                    st.session_state.get(
                        "external_compatibility",
                        {}
                    )
                )

                unavailable = (
                    compatibility.get(
                        "unavailable",
                        []
                    )
                )

                if unavailable:

                    st.warning(
                        "Some standardized columns contain "
                        "missing information because no equivalent "
                        "field existed in the uploaded dataset."
                    )

            st.button(
                "Continue to Predictions →",
                on_click=go_to_page,
                args=("Predictions",)
            )


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "Predictions":

    st.markdown(
        '<div class="page-title">Academic Risk Predictions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Apply the deployed Logistic Regression model to the active
standardized student dataset.
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='height: 18px;'></div>",
        unsafe_allow_html=True
    )

    if not st.session_state.dataset_loaded:

        st.warning(
            "Upload student data first."
        )

        st.button(
            "Upload Data →",
            on_click=go_to_page,
            args=("Upload Data",)
        )

    elif not model_loaded:

        st.error(
            f"Model unavailable: {model_error}"
        )

    else:

        df_prediction = (
            st.session_state.uploaded_dataset.copy()
        )

        missing_columns = [
            feature
            for feature in model_features
            if feature not in df_prediction.columns
        ]

        if missing_columns:

            st.error(
                "Prediction cannot run because required standardized "
                "columns are missing from the active dataset."
            )

            st.write(missing_columns)

            st.info(
                "Return to Upload Data → Prepared Analytical Dataset, confirm the "
                "mapping, then click Validate & Prepare Dataset before prediction."
            )

        else:

            st.markdown(
                """
<div class="upload-ready-card">
<b>Prediction dataset ready</b><br>
The active dataset has been standardized to the deployed
model schema.
</div>
<div style="height: 26px;"></div>
""",
                unsafe_allow_html=True
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Records",
                    f"{len(df_prediction):,}"
                )

            with p2:
                st.metric(
                    "Model Features",
                    len(model_features)
                )

            with p3:
                st.metric(
                    "Decision Threshold",
                    f"{threshold:.4f}"
                )

            if st.button(
                "Run Academic Risk Prediction",
                type="primary"
            ):

                try:

                    pred_X = (
                        df_prediction[
                            model_features
                        ]
                        .copy()
                    )

                    # ----------------------------------------
                    # RESTORE GENUINELY UNAVAILABLE SCORES
                    # ----------------------------------------

                    if (
                        "has_score_week6"
                        in pred_X.columns
                    ):

                        no_score = (
                            pd.to_numeric(
                                pred_X[
                                    "has_score_week6"
                                ],
                                errors="coerce"
                            )
                            == 0
                        )

                        for score_column in [
                            "avg_score_week6",
                            "min_score_week6",
                            "max_score_week6"
                        ]:

                            if (
                                score_column
                                in pred_X.columns
                            ):

                                pred_X.loc[
                                    no_score,
                                    score_column
                                ] = np.nan

                    # ----------------------------------------
                    # RESTORE MISSING REGISTRATION DATE
                    # ----------------------------------------

                    if (
                        "registration_date_missing"
                        in pred_X.columns
                        and
                        "date_registration"
                        in pred_X.columns
                    ):

                        registration_missing = (
                            pd.to_numeric(
                                pred_X[
                                    "registration_date_missing"
                                ],
                                errors="coerce"
                            )
                            == 1
                        )

                        pred_X.loc[
                            registration_missing,
                            "date_registration"
                        ] = np.nan

                    # ----------------------------------------
                    # FINAL SKLEARN-SAFE TYPE CLEANING
                    # ----------------------------------------
                    # External CSVs can contain pandas pd.NA values after
                    # text cleaning/mapping. The deployed sklearn imputer
                    # was trained with np.nan, so normalize missing values
                    # and dtypes before calling predict_proba.

                    categorical_model_features = [
                        "code_module",
                        "code_presentation",
                        "gender",
                        "region",
                        "highest_education",
                        "imd_band",
                        "age_band",
                        "disability"
                    ]

                    numerical_model_features = [
                        feature
                        for feature in model_features
                        if feature not in categorical_model_features
                    ]

                    for feature in categorical_model_features:
                        if feature in pred_X.columns:
                            pred_X[feature] = (
                                pred_X[feature]
                                .astype(object)
                                .where(
                                    pd.notna(pred_X[feature]),
                                    np.nan
                                )
                            )

                    for feature in numerical_model_features:
                        if feature in pred_X.columns:
                            pred_X[feature] = pd.to_numeric(
                                pred_X[feature],
                                errors="coerce"
                            ).astype(float)

                    # Defensive conversion: eliminate any remaining pd.NA
                    # anywhere in the matrix while preserving np.nan.
                    pred_X = pred_X.astype(object).where(
                        pd.notna(pred_X),
                        np.nan
                    )

                    # Put numerical columns back to numeric dtype after the
                    # dataframe-wide missing-value normalization.
                    for feature in numerical_model_features:
                        if feature in pred_X.columns:
                            pred_X[feature] = pd.to_numeric(
                                pred_X[feature],
                                errors="coerce"
                            )

                    with st.spinner(
                        "Generating academic risk predictions..."
                    ):

                        risk_probability = (
                            model.predict_proba(
                                pred_X
                            )[:, 1]
                        )

                        risk_prediction = (
                            risk_probability
                            >= threshold
                        ).astype(int)

                    results = (
                        df_prediction.copy()
                    )

                    results[
                        "risk_probability"
                    ] = risk_probability

                    results[
                        "risk_probability_pct"
                    ] = (
                        risk_probability
                        *
                        100
                    ).round(2)

                    results[
                        "predicted_at_risk"
                    ] = risk_prediction

                    results[
                        "predicted_risk"
                    ] = (
                        pd.Series(
                            risk_prediction,
                            index=results.index
                        )
                        .map(
                            {
                                1: "At Risk",
                                0: "Not At Risk"
                            }
                        )
                    )

                    st.session_state.prediction_results = results
                    st.session_state.predictions_generated = True

                    st.success(
                        "Predictions generated successfully."
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(e)

            if (
                st.session_state.predictions_generated
                and
                st.session_state.prediction_results
                is not None
            ):

                results = (
                    st.session_state.prediction_results
                )

                at_risk_count = int(
                    results[
                        "predicted_at_risk"
                    ].sum()
                )

                safe_count = (
                    len(results)
                    -
                    at_risk_count
                )

                rate = (
                    at_risk_count
                    /
                    len(results)
                    *
                    100
                )

                r1, r2, r3 = st.columns(3)

                with r1:

                    st.markdown(
                        f"""
<div class="status-card status-red">
<div class="status-label">At Risk</div>
<div class="status-value risk-text">{at_risk_count:,}</div>
<div class="status-note">Predicted intervention candidates</div>
</div>
""",
                        unsafe_allow_html=True
                    )

                with r2:

                    st.markdown(
                        f"""
<div class="status-card status-green">
<div class="status-label">Not At Risk</div>
<div class="status-value safe-text">{safe_count:,}</div>
<div class="status-note">Lower predicted academic risk</div>
</div>
""",
                        unsafe_allow_html=True
                    )

                with r3:

                    st.markdown(
                        f"""
<div class="status-card status-amber">
<div class="status-label">Risk Rate</div>
<div class="status-value amber-text">{rate:.1f}%</div>
<div class="status-note">Current dataset</div>
</div>
""",
                        unsafe_allow_html=True
                    )

                display_columns = [
                    column
                    for column in [
                        "id_student",
                        "code_module",
                        "code_presentation",
                        "risk_probability_pct",
                        "predicted_risk"
                    ]
                    if column in results.columns
                ]

                st.markdown(
                    '<div class="quick-header">Prediction Results</div>',
                    unsafe_allow_html=True
                )

                view_choice = st.radio(
                    "Rows to display",
                    ["First 100", "View All"],
                    horizontal=True,
                    key="prediction_rows_view"
                )

                prediction_view = (
                    results[display_columns]
                    if view_choice == "View All"
                    else results[display_columns].head(100)
                )

                def highlight_prediction_status(value):
                    if value == "At Risk":
                        return (
                            "background-color: rgba(220, 53, 69, 0.22); "
                            "color: #ff8a8a; font-weight: 700;"
                        )
                    if value == "Not At Risk":
                        return (
                            "background-color: rgba(40, 167, 69, 0.20); "
                            "color: #79e08b; font-weight: 700;"
                        )
                    return ""

                if "predicted_risk" in prediction_view.columns:
                    styled_prediction_view = (
                        prediction_view.style.map(
                            highlight_prediction_status,
                            subset=["predicted_risk"]
                        )
                    )
                    st.dataframe(
                        styled_prediction_view,
                        use_container_width=True,
                        hide_index=True,
                        height=620 if view_choice == "View All" else 420
                    )
                else:
                    st.dataframe(
                        prediction_view,
                        use_container_width=True,
                        hide_index=True,
                        height=620 if view_choice == "View All" else 420
                    )

                st.markdown(
                    "<div style='height: 12px;'></div>",
                    unsafe_allow_html=True
                )

                st.download_button(
                    "↓ Download Prediction Results",
                    results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    "student_risk_predictions.csv",
                    "text/csv"
                )


            st.warning(
                "Prediction guidance: Risk classifications are probabilistic "
                "decision-support outputs, not guaranteed academic outcomes. "
                "Use them to prioritize review and early support alongside "
                "academic judgement and other available student information."
            )


# ============================================================
# STUDENTS
# ============================================================

elif page == "Students":

    st.markdown(
        '<div class="page-title">Student Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Investigate an individual student or enrollment record,
their predicted risk and available early learning indicators.
</div>
""",
        unsafe_allow_html=True
    )

    if (
        not st.session_state.predictions_generated
        or
        st.session_state.prediction_results
        is None
    ):

        st.warning(
            "Generate predictions first."
        )

        st.button(
            "Go to Predictions →",
            on_click=go_to_page,
            args=("Predictions",)
        )

    else:

        students_df = (
            st.session_state.prediction_results.copy()
        )

        student_ids = (
            students_df[
                "id_student"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_student = st.selectbox(
            "Student ID",
            student_ids
        )

        student_rows = (
            students_df[
                students_df[
                    "id_student"
                ]
                == selected_student
            ]
            .copy()
        )

        if len(student_rows) > 1:

            student_rows[
                "_enrollment"
            ] = (
                student_rows[
                    "code_module"
                ].astype(str)
                +
                " · "
                +
                student_rows[
                    "code_presentation"
                ].astype(str)
            )

            selected_enrollment = (
                st.selectbox(
                    "Enrollment",
                    student_rows[
                        "_enrollment"
                    ].tolist()
                )
            )

            row = (
                student_rows[
                    student_rows[
                        "_enrollment"
                    ]
                    == selected_enrollment
                ]
                .iloc[0]
            )

        else:

            row = (
                student_rows.iloc[0]
            )

        probability = (
            safe_numeric_value(
                row.get(
                    "risk_probability_pct",
                    np.nan
                ),
                np.nan
            )
        )

        prediction = row.get(
            "predicted_risk",
            "Unknown"
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Student ID",
                str(
                    selected_student
                )
            )

        with s2:

            if prediction == "At Risk":

                st.markdown(
                    f"""
<div class="status-card status-red">
<div class="status-label">Prediction</div>
<div class="status-value risk-text">At Risk</div>
</div>
""",
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
<div class="status-card status-green">
<div class="status-label">Prediction</div>
<div class="status-value safe-text">Not At Risk</div>
</div>
""",
                    unsafe_allow_html=True
                )

        with s3:
            st.metric(
                "Risk Probability",
                (
                    f"{probability:.1f}%"
                    if not pd.isna(
                        probability
                    )
                    else "N/A"
                )
            )

        with s4:
            st.metric(
                "Module",
                str(
                    row.get(
                        "code_module",
                        "N/A"
                    )
                )
            )

        st.markdown(
            '<div class="quick-header">Early Learning Indicators</div>',
            unsafe_allow_html=True
        )

        indicator_fields = {
            "VLE Clicks":
                "total_clicks_week6",
            "Active Days":
                "active_days_week6",
            "Resources Accessed":
                "resources_accessed_week6",
            "Average Score":
                "avg_score_week6",
            "Assessments Submitted":
                "assessments_submitted_week6",
            "Late Submissions":
                "late_submissions_week6",
            "Previous Attempts":
                "num_of_prev_attempts",
            "Studied Credits":
                "studied_credits"
        }

        indicator_rows = []

        has_score = safe_numeric_value(
            row.get(
                "has_score_week6",
                np.nan
            ),
            np.nan
        )

        for label, field in indicator_fields.items():

            value = row.get(
                field,
                np.nan
            )

            if (
                field
                == "avg_score_week6"
                and
                has_score == 0
            ):
                value = "Not available"

            indicator_rows.append(
                {
                    "Indicator": label,
                    "Value": value
                }
            )

        st.dataframe(
            pd.DataFrame(
                indicator_rows
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="quick-header">Observed Support Signals</div>',
            unsafe_allow_html=True
        )

        signals = []

        active_days = safe_numeric_value(
            row.get(
                "active_days_week6",
                np.nan
            ),
            np.nan
        )

        clicks = safe_numeric_value(
            row.get(
                "total_clicks_week6",
                np.nan
            ),
            np.nan
        )

        avg_score = safe_numeric_value(
            row.get(
                "avg_score_week6",
                np.nan
            ),
            np.nan
        )

        late = safe_numeric_value(
            row.get(
                "late_submissions_week6",
                np.nan
            ),
            np.nan
        )

        if (
            not pd.isna(active_days)
            and active_days <= 5
        ):
            signals.append(
                "Limited early learning activity"
            )

        if (
            not pd.isna(clicks)
            and clicks == 0
        ):
            signals.append(
                "No recorded VLE activity"
            )

        if (
            has_score == 1
            and
            not pd.isna(avg_score)
            and
            avg_score < 50
        ):
            signals.append(
                "Low available assessment score"
            )

        if (
            not pd.isna(late)
            and late > 0
        ):
            signals.append(
                "Late assessment submissions"
            )

        if has_score == 0:
            signals.append(
                "No early assessment score available"
            )

        if signals:

            for signal in signals:
                st.warning(signal)

        else:

            st.success(
                "No additional operational warning signals "
                "were identified by the current review rules."
            )

        st.caption(
            "These indicators are descriptive support signals. "
            "They do not prove why the model produced its prediction."
        )

        st.button(
            "Review Early Interventions →",
            on_click=go_to_page,
            args=("Interventions",)
        )


# ============================================================
# INTERVENTIONS
# ============================================================

elif page == "Interventions":

    st.markdown(
        '<div class="page-title">Early Interventions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Prioritise the complete predicted at-risk population and
review support recommendations for individual cases.
</div>
""",
        unsafe_allow_html=True
    )

    if (
        not st.session_state.predictions_generated
        or
        st.session_state.prediction_results
        is None
    ):

        st.warning(
            "Generate predictions before reviewing interventions."
        )

        st.button(
            "Go to Predictions →",
            on_click=go_to_page,
            args=("Predictions",)
        )

    else:

        intervention_data = (
            st.session_state.prediction_results.copy()
        )

        at_risk_students = (
            intervention_data[
                intervention_data[
                    "predicted_at_risk"
                ]
                == 1
            ]
            .copy()
            .reset_index(drop=True)
        )

        if at_risk_students.empty:

            st.success(
                "No students are currently predicted At Risk."
            )

        else:

            def create_support_profile(row):

                needs = []
                actions = []

                active_days = safe_numeric_value(
                    row.get(
                        "active_days_week6",
                        np.nan
                    ),
                    np.nan
                )

                clicks = safe_numeric_value(
                    row.get(
                        "total_clicks_week6",
                        np.nan
                    ),
                    np.nan
                )

                resources = safe_numeric_value(
                    row.get(
                        "resources_accessed_week6",
                        np.nan
                    ),
                    np.nan
                )

                score = safe_numeric_value(
                    row.get(
                        "avg_score_week6",
                        np.nan
                    ),
                    np.nan
                )

                has_score = safe_numeric_value(
                    row.get(
                        "has_score_week6",
                        np.nan
                    ),
                    np.nan
                )

                submissions = safe_numeric_value(
                    row.get(
                        "assessments_submitted_week6",
                        np.nan
                    ),
                    np.nan
                )

                late = safe_numeric_value(
                    row.get(
                        "late_submissions_week6",
                        np.nan
                    ),
                    np.nan
                )

                previous = safe_numeric_value(
                    row.get(
                        "num_of_prev_attempts",
                        np.nan
                    ),
                    np.nan
                )

                if (
                    not pd.isna(active_days)
                    and active_days <= 5
                ):
                    needs.append(
                        "Low engagement"
                    )
                    actions.append(
                        "Engagement outreach"
                    )

                if (
                    not pd.isna(clicks)
                    and clicks == 0
                ):
                    needs.append(
                        "No VLE activity"
                    )

                    if (
                        "Engagement outreach"
                        not in actions
                    ):
                        actions.append(
                            "Engagement outreach"
                        )

                if (
                    not pd.isna(resources)
                    and resources == 0
                ):
                    needs.append(
                        "No resource access"
                    )

                if has_score == 0:

                    needs.append(
                        "No early score"
                    )

                    actions.append(
                        "Assessment check-in"
                    )

                elif (
                    not pd.isna(score)
                    and score < 50
                ):

                    needs.append(
                        "Low assessment score"
                    )

                    actions.append(
                        "Academic support"
                    )

                if (
                    not pd.isna(submissions)
                    and submissions == 0
                ):

                    needs.append(
                        "No assessments submitted"
                    )

                    if (
                        "Academic support"
                        not in actions
                    ):
                        actions.append(
                            "Academic support"
                        )

                if (
                    not pd.isna(late)
                    and late > 0
                ):

                    needs.append(
                        "Late submissions"
                    )

                    actions.append(
                        "Submission support"
                    )

                if (
                    not pd.isna(previous)
                    and previous > 0
                ):

                    needs.append(
                        "Previous attempts"
                    )

                    actions.append(
                        "Academic advisor review"
                    )

                if not needs:
                    needs = [
                        "General academic review"
                    ]

                if not actions:
                    actions = [
                        "General academic review"
                    ]

                return (
                    ", ".join(
                        dict.fromkeys(
                            needs
                        )
                    ),
                    ", ".join(
                        dict.fromkeys(
                            actions
                        )
                    )
                )

            support_profiles = (
                at_risk_students.apply(
                    create_support_profile,
                    axis=1
                )
            )

            at_risk_students[
                "Observed Support Need"
            ] = [
                result[0]
                for result in support_profiles
            ]

            at_risk_students[
                "Recommended Intervention"
            ] = [
                result[1]
                for result in support_profiles
            ]

            at_risk_students[
                "Status"
            ] = "Needs Review"

            at_risk_students = (
                at_risk_students
                .sort_values(
                    "risk_probability",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            total_at_risk = len(
                at_risk_students
            )

            unique_at_risk = (
                at_risk_students[
                    "id_student"
                ].nunique()
            )

            average_probability = (
                at_risk_students[
                    "risk_probability"
                ].mean()
                *
                100
            )

            i1, i2, i3 = st.columns(3)

            with i1:

                st.markdown(
                    f"""
<div class="status-card status-red">
<div class="status-label">At-Risk Enrollments</div>
<div class="status-value risk-text">{total_at_risk:,}</div>
</div>
""",
                    unsafe_allow_html=True
                )

            with i2:
                st.metric(
                    "Unique Students",
                    f"{unique_at_risk:,}"
                )

            with i3:
                st.metric(
                    "Average Risk Probability",
                    f"{average_probability:.1f}%"
                )

            st.markdown(
                '<div class="quick-header">At-Risk Intervention Queue</div>',
                unsafe_allow_html=True
            )

            intervention_display = (
                at_risk_students.copy()
            )

            intervention_display[
                "Risk Probability (%)"
            ] = (
                intervention_display[
                    "risk_probability"
                ]
                *
                100
            ).round(2)

            display_columns = [
                column
                for column in [
                    "id_student",
                    "code_module",
                    "code_presentation",
                    "Risk Probability (%)",
                    "Observed Support Need",
                    "Recommended Intervention",
                    "Status"
                ]
                if column
                in intervention_display.columns
            ]

            # Scrollable intervention queue:
            # fixed-height container (vertical scroll) + horizontal scroll,
            # with complete wrapped text and no truncation.
            queue_table = intervention_display[display_columns].copy()
            # Fast virtualised intervention queue.
            # Native Streamlit rendering keeps rows compact and uses scrollbars,
            # instead of rendering thousands of wrapped HTML rows.
            st.dataframe(
                queue_table,
                use_container_width=True,
                hide_index=True,
                height=560,
                column_config={
                    "id_student": st.column_config.TextColumn(
                        "Student ID", width="small"
                    ),
                    "code_module": st.column_config.TextColumn(
                        "Module", width="small"
                    ),
                    "code_presentation": st.column_config.TextColumn(
                        "Presentation", width="small"
                    ),
                    "Risk Probability (%)": st.column_config.NumberColumn(
                        "Risk Probability (%)",
                        format="%.1f%%",
                        width="small"
                    ),
                    "Observed Support Need": st.column_config.TextColumn(
                        "Observed Support Need", width="large"
                    ),
                    "Recommended Intervention": st.column_config.TextColumn(
                        "Recommended Intervention", width="large"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status", width="small"
                    ),
                }
            )

            st.caption(
                "Scroll horizontally or vertically to review the queue. "
                "The table is virtualised for faster loading; use Review Individual Case "
                "below to read the complete support need and recommendation for one student."
            )

            st.markdown(
                '<div class="quick-header">Review Individual Case</div>',
                unsafe_allow_html=True
            )

            case_options = list(
                range(
                    len(
                        at_risk_students
                    )
                )
            )

            selected_case = st.selectbox(
                "Select at-risk enrollment",
                case_options,
                format_func=lambda idx:
                    (
                        f"{at_risk_students.loc[idx, 'id_student']} · "
                        f"{at_risk_students.loc[idx, 'code_module']} · "
                        f"{at_risk_students.loc[idx, 'code_presentation']}"
                    )
            )

            case = (
                at_risk_students.loc[
                    selected_case
                ]
            )

            c1, c2 = st.columns(2)

            with c1:
                st.error(
                    "Observed Support Need\n\n"
                    +
                    str(
                        case[
                            "Observed Support Need"
                        ]
                    )
                )

            with c2:
                st.success(
                    "Recommended Intervention\n\n"
                    +
                    str(
                        case[
                            "Recommended Intervention"
                        ]
                    )
                )

            st.download_button(
                "↓ Download Intervention Queue",
                intervention_display.to_csv(
                    index=False
                ).encode("utf-8"),
                "at_risk_student_interventions.csv",
                "text/csv"
            )

            st.caption(
                "The machine-learning model determines the At-Risk "
                "classification. Intervention suggestions are transparent "
                "operational support rules and require human review."
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="page-title">Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
<div class="page-subtitle">
Explore academic risk, early-warning problems, engagement, academic history,
student profile patterns and module-level differences.
</div>
""",
        unsafe_allow_html=True
    )

    if (
        not st.session_state.predictions_generated
        or st.session_state.prediction_results is None
    ):
        st.info("Run predictions first to unlock analytics.")
        st.button(
            "Go to Predictions →",
            on_click=go_to_page,
            args=("Predictions",)
        )

    else:
        analytics_df = st.session_state.prediction_results.copy()

        if "predicted_risk" not in analytics_df.columns and "predicted_at_risk" in analytics_df.columns:
            analytics_df["predicted_risk"] = np.where(
                analytics_df["predicted_at_risk"] == 1,
                "At Risk",
                "Not At Risk"
            )

        if "predicted_at_risk" not in analytics_df.columns and "predicted_risk" in analytics_df.columns:
            analytics_df["predicted_at_risk"] = (
                analytics_df["predicted_risk"].eq("At Risk").astype(int)
            )

        if "risk_probability_pct" not in analytics_df.columns and "risk_probability" in analytics_df.columns:
            analytics_df["risk_probability_pct"] = (
                pd.to_numeric(analytics_df["risk_probability"], errors="coerce") * 100
            )

        # -------------------- Filters --------------------
        st.markdown('<div class="quick-header">Analysis Filters</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)

        module_options = ["All"]
        if "code_module" in analytics_df.columns:
            module_options += sorted(
                analytics_df["code_module"].dropna().astype(str).unique().tolist()
            )

        presentation_options = ["All"]
        if "code_presentation" in analytics_df.columns:
            presentation_options += sorted(
                analytics_df["code_presentation"].dropna().astype(str).unique().tolist()
            )

        with f1:
            selected_module = st.selectbox("Module", module_options, key="analytics_module")
        with f2:
            selected_presentation = st.selectbox(
                "Presentation", presentation_options, key="analytics_presentation"
            )

        filtered = analytics_df.copy()
        if selected_module != "All" and "code_module" in filtered.columns:
            filtered = filtered[filtered["code_module"].astype(str) == selected_module]
        if selected_presentation != "All" and "code_presentation" in filtered.columns:
            filtered = filtered[
                filtered["code_presentation"].astype(str) == selected_presentation
            ]

        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            # -------------------- Overview --------------------
            total_records = len(filtered)
            unique_students = (
                filtered["id_student"].nunique()
                if "id_student" in filtered.columns else total_records
            )
            at_risk_count = int(filtered["predicted_at_risk"].sum())
            risk_rate = (
                at_risk_count / total_records * 100 if total_records else 0
            )

            st.markdown('<div class="quick-header">Risk Overview</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Enrollment Records", f"{total_records:,}")
            k2.metric("Unique Students", f"{unique_students:,}")
            k3.metric("At-Risk Enrollments", f"{at_risk_count:,}")
            k4.metric("Risk Rate", f"{risk_rate:.1f}%")

            distribution = (
                filtered["predicted_risk"]
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Records")
            )
            distribution["Share (%)"] = (
                distribution["Records"] / distribution["Records"].sum() * 100
            )
            distribution["Visible Label"] = distribution.apply(
                lambda row: (
                    f"{row['Prediction']}  "
                    f"{int(row['Records']):,}  "
                    f"({row['Share (%)']:.1f}%)"
                ),
                axis=1
            )

            donut_base = alt.Chart(distribution).encode(
                theta=alt.Theta("Records:Q", stack=True),
                color=alt.Color(
                    "Prediction:N",
                    scale=alt.Scale(
                        domain=["At Risk", "Not At Risk"],
                        range=["#E66B6B", "#59B883"]
                    ),
                    legend=alt.Legend(title=None)
                )
            )

            donut_arcs = donut_base.mark_arc(
                innerRadius=62,
                outerRadius=112
            ).encode(
                tooltip=[
                    alt.Tooltip("Prediction:N", title="Prediction"),
                    alt.Tooltip("Records:Q", title="Records", format=","),
                    alt.Tooltip("Share (%):Q", title="Share", format=".1f")
                ]
            )

            donut_labels = donut_base.mark_text(
                radius=145,
                size=13,
                fontWeight="bold"
            ).encode(
                text=alt.Text("Visible Label:N"),
                color=alt.value("#DCE4EE")
            )

            dist_chart = (
                (donut_arcs + donut_labels)
                .properties(height=340, title="Prediction Distribution")
            )
            st.altair_chart(dist_chart, use_container_width=True)

            # -------------------- Early warning problems --------------------
            st.markdown(
                '<div class="quick-header">Main Early-Warning Problems</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "These are observable Week-6 warning signals among students predicted At Risk. "
                "They are associated with risk and do not prove cause-and-effect. "
                "A student may appear in more than one problem category."
            )

            at_risk_df = filtered[filtered["predicted_at_risk"] == 1].copy()
            problem_rows = []

            def _num(col):
                if col in at_risk_df.columns:
                    return pd.to_numeric(at_risk_df[col], errors="coerce")
                return pd.Series(np.nan, index=at_risk_df.index)

            problem_masks = {
                "Low engagement": _num("active_days_week6") <= 5,
                "No VLE activity": _num("total_clicks_week6") == 0,
                "No resource access": _num("resources_accessed_week6") == 0,
                "No early score": _num("has_score_week6") == 0,
                "Low Week-6 score (<50)": (
                    (_num("has_score_week6") == 1)
                    & (_num("avg_score_week6") < 50)
                ),
                "No assessments submitted": _num("assessments_submitted_week6") == 0,
                "Late submissions": _num("late_submissions_week6") > 0,
                "Previous attempts": _num("num_of_prev_attempts") > 0,
            }

            at_risk_unique = (
                at_risk_df["id_student"].nunique()
                if "id_student" in at_risk_df.columns else len(at_risk_df)
            )

            for problem, mask in problem_masks.items():
                affected = at_risk_df.loc[mask.fillna(False)]
                count = (
                    affected["id_student"].nunique()
                    if "id_student" in affected.columns else len(affected)
                )
                share = count / at_risk_unique * 100 if at_risk_unique else 0
                problem_rows.append({
                    "Problem": problem,
                    "At-Risk Students": int(count),
                    "Share of At-Risk Students (%)": float(share)
                })

            problem_df = pd.DataFrame(problem_rows).sort_values(
                "At-Risk Students", ascending=False
            )

            if not problem_df.empty:
                problem_chart = (
                    alt.Chart(problem_df)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        y=alt.Y("Problem:N", sort="-x", title=None),
                        x=alt.X("At-Risk Students:Q", title="Unique at-risk students"),
                        color=alt.Color("Problem:N", scale=alt.Scale(scheme="tableau10"), legend=None),
                        tooltip=[
                            "Problem:N",
                            alt.Tooltip("At-Risk Students:Q", format=","),
                            alt.Tooltip("Share of At-Risk Students (%):Q", format=".1f")
                        ]
                    )
                    .properties(height=360, title="Most Common Early-Warning Problems")
                )
                st.altair_chart(problem_chart, use_container_width=True)

                highest_problem = problem_df.iloc[0]
                lowest_problem = problem_df.iloc[-1]
                c1, c2 = st.columns(2)
                c1.info(
                    f"Most common warning signal: **{highest_problem['Problem']}** — "
                    f"{int(highest_problem['At-Risk Students']):,} unique at-risk students "
                    f"({highest_problem['Share of At-Risk Students (%)']:.1f}%)."
                )
                c2.info(
                    f"Least common observed warning signal: **{lowest_problem['Problem']}** — "
                    f"{int(lowest_problem['At-Risk Students']):,} unique at-risk students "
                    f"({lowest_problem['Share of At-Risk Students (%)']:.1f}%)."
                )

            # -------------------- Engagement and assessment --------------------
            st.markdown(
                '<div class="quick-header">Engagement & Assessment Patterns</div>',
                unsafe_allow_html=True
            )

            comparison_features = [
                ("active_days_week6", "Active Days"),
                ("total_clicks_week6", "VLE Clicks"),
                ("resources_accessed_week6", "Resources Accessed"),
                ("avg_score_week6", "Week-6 Score"),
                ("assessments_submitted_week6", "Assessments Submitted"),
                ("late_submissions_week6", "Late Submissions"),
            ]

            comparison_rows = []
            for col, label in comparison_features:
                if col not in filtered.columns:
                    continue
                temp = filtered[["predicted_risk", col]].copy()
                temp[col] = pd.to_numeric(temp[col], errors="coerce")
                if col == "avg_score_week6" and "has_score_week6" in filtered.columns:
                    temp.loc[
                        pd.to_numeric(filtered["has_score_week6"], errors="coerce") == 0,
                        col
                    ] = np.nan
                grouped = temp.groupby("predicted_risk")[col].mean()
                for group, value in grouped.items():
                    comparison_rows.append({
                        "Measure": label,
                        "Prediction": group,
                        "Average": value
                    })

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)

                for measure in comp_df["Measure"].dropna().unique():
                    measure_df = comp_df[comp_df["Measure"] == measure].copy()
                    measure_chart = (
                        alt.Chart(measure_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=72)
                        .encode(
                            x=alt.X(
                                "Prediction:N",
                                title=None,
                                axis=alt.Axis(labelAngle=0)
                            ),
                            y=alt.Y(
                                "Average:Q",
                                title=f"Average {measure}"
                            ),
                            color=alt.Color(
                                "Prediction:N",
                                scale=alt.Scale(
                                    domain=["At Risk", "Not At Risk"],
                                    range=["#E66B6B", "#59B883"]
                                ),
                                legend=alt.Legend(title=None, orient="top")
                            ),
                            tooltip=[
                                alt.Tooltip("Prediction:N", title="Prediction"),
                                alt.Tooltip("Average:Q", title=f"Average {measure}", format=".2f")
                            ]
                        )
                        .properties(
                            height=300,
                            title=f"Average {measure}: At Risk vs Not At Risk"
                        )
                    )
                    st.altair_chart(measure_chart, use_container_width=True)
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # -------------------- Academic history --------------------
            st.markdown(
                '<div class="quick-header">Academic History & Workload</div>',
                unsafe_allow_html=True
            )
            academic_features = [
                ("num_of_prev_attempts", "Previous Attempts"),
                ("studied_credits", "Studied Credits"),
            ]
            academic_rows = []
            for col, label in academic_features:
                if col in filtered.columns:
                    tmp = filtered[["predicted_risk", col]].copy()
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                    vals = tmp.groupby("predicted_risk")[col].mean()
                    for group, value in vals.items():
                        academic_rows.append({
                            "Measure": label,
                            "Prediction": group,
                            "Average": value
                        })
            if academic_rows:
                academic_df = pd.DataFrame(academic_rows)

                for measure in academic_df["Measure"].dropna().unique():
                    measure_df = academic_df[academic_df["Measure"] == measure].copy()
                    acad_chart = (
                        alt.Chart(measure_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=72)
                        .encode(
                            x=alt.X(
                                "Prediction:N",
                                title=None,
                                axis=alt.Axis(labelAngle=0)
                            ),
                            y=alt.Y(
                                "Average:Q",
                                title=f"Average {measure}"
                            ),
                            color=alt.Color(
                                "Prediction:N",
                                scale=alt.Scale(
                                    domain=["At Risk", "Not At Risk"],
                                    range=["#E66B6B", "#59B883"]
                                ),
                                legend=alt.Legend(title=None, orient="top")
                            ),
                            tooltip=[
                                alt.Tooltip("Prediction:N", title="Prediction"),
                                alt.Tooltip("Average:Q", title=f"Average {measure}", format=".2f")
                            ]
                        )
                        .properties(
                            height=300,
                            title=f"Average {measure}: At Risk vs Not At Risk"
                        )
                    )
                    st.altair_chart(acad_chart, use_container_width=True)
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # -------------------- Reusable categorical risk analysis --------------------
            st.markdown(
                '<div class="quick-header">Student Profile Risk Patterns</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "These are descriptive group differences for monitoring and fairness. "
                "They should not be interpreted as causes of academic failure."
            )

            def categorical_risk_table(data, column):
                work = data[[column, "predicted_at_risk"]].dropna().copy()
                if work.empty:
                    return pd.DataFrame()
                work[column] = work[column].astype(str)
                out = (
                    work.groupby(column, as_index=False)
                    .agg(
                        Records=("predicted_at_risk", "size"),
                        At_Risk=("predicted_at_risk", "sum")
                    )
                )
                out["Risk Rate (%)"] = out["At_Risk"] / out["Records"] * 100
                return out.sort_values("Risk Rate (%)", ascending=False)

            profile_specs = [
                ("age_band", "Risk by Age Band"),
                ("gender", "Risk by Gender"),
                ("highest_education", "Risk by Highest Education"),
                ("imd_band", "Risk by IMD Band"),
                ("disability", "Risk by Disability Status"),
                ("region", "Risk by Region"),
            ]

            for column, title in profile_specs:
                if column not in filtered.columns:
                    continue

                profile_df = categorical_risk_table(filtered, column)
                if profile_df.empty:
                    continue

                profile_chart = (
                    alt.Chart(profile_df)
                    .mark_bar(cornerRadiusEnd=5)
                    .encode(
                        y=alt.Y(f"{column}:N", sort="-x", title=None),
                        x=alt.X(
                            "Risk Rate (%):Q",
                            title="Predicted at-risk rate (%)",
                            scale=alt.Scale(domain=[0, 100])
                        ),
                        color=alt.Color(
                            f"{column}:N",
                            scale=alt.Scale(scheme="tableau10"),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip(f"{column}:N", title=column.replace("_", " ").title()),
                            alt.Tooltip("Records:Q", format=","),
                            alt.Tooltip("At_Risk:Q", title="At Risk", format=","),
                            alt.Tooltip("Risk Rate (%):Q", format=".1f")
                        ]
                    )
                    .properties(height=max(230, min(520, 34 * len(profile_df))), title=title)
                )
                st.altair_chart(profile_chart, use_container_width=True)

                highest = profile_df.iloc[0]
                lowest = profile_df.iloc[-1]
                st.caption(
                    f"Highest predicted risk rate: {highest[column]} "
                    f"({highest['Risk Rate (%)']:.1f}%). "
                    f"Lowest: {lowest[column]} ({lowest['Risk Rate (%)']:.1f}%)."
                )

            # -------------------- Module analysis --------------------
            if "code_module" in filtered.columns:
                st.markdown(
                    '<div class="quick-header">Module Risk Analysis</div>',
                    unsafe_allow_html=True
                )
                module_df = (
                    filtered.groupby("code_module", as_index=False)
                    .agg(
                        Records=("predicted_at_risk", "size"),
                        At_Risk=("predicted_at_risk", "sum")
                    )
                )
                module_df["Risk Rate (%)"] = (
                    module_df["At_Risk"] / module_df["Records"] * 100
                )
                module_df = module_df.sort_values("Risk Rate (%)", ascending=False)

                module_chart = (
                    alt.Chart(module_df)
                    .mark_bar(cornerRadiusEnd=5)
                    .encode(
                        y=alt.Y("code_module:N", sort="-x", title=None),
                        x=alt.X(
                            "Risk Rate (%):Q",
                            title="Predicted at-risk rate (%)",
                            scale=alt.Scale(domain=[0, 100])
                        ),
                        color=alt.Color(
                            "code_module:N",
                            scale=alt.Scale(scheme="tableau10"),
                            legend=None
                        ),
                        tooltip=[
                            "code_module:N",
                            alt.Tooltip("Records:Q", format=","),
                            alt.Tooltip("At_Risk:Q", title="At Risk", format=","),
                            alt.Tooltip("Risk Rate (%):Q", format=".1f")
                        ]
                    )
                    .properties(height=320, title="Risk by Module")
                )
                st.altair_chart(module_chart, use_container_width=True)

                if not module_df.empty:
                    hi = module_df.iloc[0]
                    lo = module_df.iloc[-1]
                    st.info(
                        f"Highest module risk rate: **{hi['code_module']}** "
                        f"({hi['Risk Rate (%)']:.1f}%). "
                        f"Lowest: **{lo['code_module']}** "
                        f"({lo['Risk Rate (%)']:.1f}%)."
                    )

            # -------------------- Highest-risk records --------------------
            st.markdown(
                '<div class="quick-header">Highest-Risk Records</div>',
                unsafe_allow_html=True
            )
            top_columns = [
                c for c in [
                    "id_student", "code_module", "code_presentation",
                    "risk_probability_pct", "predicted_risk",
                    "age_band", "gender", "highest_education", "imd_band",
                    "active_days_week6", "avg_score_week6",
                    "late_submissions_week6"
                ]
                if c in filtered.columns
            ]

            top_risk = filtered.copy()
            if "risk_probability_pct" in top_risk.columns:
                top_risk = top_risk.sort_values(
                    "risk_probability_pct", ascending=False
                )
            top_risk = top_risk[top_columns].head(25)

            st.dataframe(
                top_risk,
                use_container_width=True,
                hide_index=True,
                height=420
            )

            st.download_button(
                "↓ Download Highest-Risk Records",
                top_risk.to_csv(index=False).encode("utf-8"),
                "highest_risk_records.csv",
                "text/csv",
                key="analytics_top_risk_download"
            )

            # -------------------- Professional Analytics PDF Report --------------------
            st.markdown(
                '<div class="quick-header">Professional Analytics Report</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Download a business-style PDF beginning with an Executive Overview, "
                "followed by each major analysis, graph interpretation, management insight "
                "and an overall conclusion."
            )

            def build_analytics_pdf():
                try:
                    import matplotlib.pyplot as plt
                    from reportlab.lib import colors
                    from reportlab.lib.enums import TA_CENTER
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import mm
                    from reportlab.platypus import (
                        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                        Image as RLImage, PageBreak, KeepTogether
                    )
                except ImportError:
                    return None, (
                        "PDF reporting requires matplotlib and reportlab. "
                        "Install them with: python -m pip install matplotlib reportlab"
                    )

                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(
                    pdf_buffer,
                    pagesize=A4,
                    rightMargin=17 * mm,
                    leftMargin=17 * mm,
                    topMargin=17 * mm,
                    bottomMargin=17 * mm,
                    title="Student Academic Risk Analytics Report",
                    author="Student Insight"
                )

                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(
                    name="ReportTitle",
                    parent=styles["Title"],
                    fontSize=22,
                    leading=27,
                    alignment=TA_CENTER,
                    spaceAfter=14
                ))
                styles.add(ParagraphStyle(
                    name="SectionTitle",
                    parent=styles["Heading2"],
                    fontSize=15,
                    leading=19,
                    spaceBefore=10,
                    spaceAfter=8
                ))
                styles.add(ParagraphStyle(
                    name="Insight",
                    parent=styles["BodyText"],
                    fontSize=10,
                    leading=15,
                    spaceAfter=8
                ))
                styles.add(ParagraphStyle(
                    name="Small",
                    parent=styles["BodyText"],
                    fontSize=8.5,
                    leading=12
                ))

                story = []
                chart_buffers = []

                def safe_mean(data, col, group):
                    if col not in data.columns:
                        return np.nan
                    vals = pd.to_numeric(
                        data.loc[data["predicted_risk"] == group, col],
                        errors="coerce"
                    )
                    return vals.mean()

                def pct_change_phrase(at_value, not_value):
                    if pd.isna(at_value) or pd.isna(not_value):
                        return "A reliable comparison could not be calculated."
                    difference = at_value - not_value
                    if abs(difference) < 1e-9:
                        return "The two groups show approximately the same average."
                    direction = "higher" if difference > 0 else "lower"
                    return (
                        f"The At Risk group records an average that is "
                        f"{abs(difference):.2f} {direction} than the Not At Risk group."
                    )

                def add_chart(fig, width=175 * mm):
                    img = BytesIO()
                    fig.tight_layout()
                    fig.savefig(img, format="png", dpi=160, bbox_inches="tight")
                    plt.close(fig)
                    img.seek(0)
                    chart_buffers.append(img)
                    story.append(RLImage(img, width=width, height=width * 0.55))
                    story.append(Spacer(1, 4 * mm))

                def add_section(title, explanation, insight=None):
                    story.append(Paragraph(title, styles["SectionTitle"]))
                    story.append(Paragraph(explanation, styles["Insight"]))
                    if insight:
                        story.append(Paragraph(
                            f"<b>Business insight:</b> {insight}",
                            styles["Insight"]
                        ))

                # ---------- Cover ----------
                story.append(Spacer(1, 15 * mm))
                story.append(Paragraph(
                    "Student Academic Performance Predictor<br/>"
                    "& Early Intervention System",
                    styles["ReportTitle"]
                ))
                story.append(Paragraph(
                    "Professional Analytics Report",
                    ParagraphStyle(
                        "Subtitle", parent=styles["Heading2"],
                        alignment=TA_CENTER, fontSize=14, leading=18
                    )
                ))
                story.append(Spacer(1, 8 * mm))
                story.append(Paragraph(
                    "This report translates the system's prediction outputs into "
                    "management-ready insights for early academic support. Findings are "
                    "descriptive and predictive associations; they should not be interpreted "
                    "as proof that a characteristic causes academic failure.",
                    styles["Insight"]
                ))
                story.append(PageBreak())

                # ---------- Executive overview ----------
                add_section(
                    "1. Executive Overview",
                    (
                        f"The analysis covers <b>{total_records:,}</b> enrollment records "
                        f"representing <b>{unique_students:,}</b> unique students in the "
                        f"current filtered view. The model classified "
                        f"<b>{at_risk_count:,}</b> enrollments as At Risk, producing an "
                        f"overall predicted risk rate of <b>{risk_rate:.1f}%</b>. "
                        "The purpose of this report is to identify where risk is concentrated, "
                        "which early-warning signals are most visible, and where academic "
                        "support resources may need to be prioritised."
                    ),
                    (
                        "Management should use the predicted risk population as a prioritised "
                        "review queue rather than as a guaranteed statement of future academic "
                        "outcomes. Intervention decisions should combine these results with "
                        "academic judgement and other available student information."
                    )
                )

                # Prediction distribution graph
                fig, ax = plt.subplots(figsize=(8.5, 4.8))
                labels = distribution["Prediction"].tolist()
                values = distribution["Records"].tolist()
                ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
                ax.set_title("Prediction Distribution")
                add_chart(fig)
                story.append(Paragraph(
                    (
                        f"<b>Graph interpretation:</b> The chart shows the balance between "
                        f"students classified At Risk and Not At Risk. The current At Risk "
                        f"share is <b>{risk_rate:.1f}%</b>. This establishes the overall scale "
                        "of potential intervention demand before examining the factors and "
                        "student groups associated with that risk."
                    ),
                    styles["Insight"]
                ))

                # ---------- Early warning problems ----------
                if not problem_df.empty:
                    top_problem = problem_df.iloc[0]
                    bottom_problem = problem_df.iloc[-1]
                    add_section(
                        "2. Early-Warning Problems",
                        (
                            "This analysis examines observable Week-6 warning signals among "
                            "students predicted At Risk. Categories can overlap because one "
                            "student may display several warning signals at the same time."
                        ),
                        (
                            f"The most prevalent signal is <b>{top_problem['Problem']}</b>, "
                            f"observed among <b>{int(top_problem['At-Risk Students']):,}</b> "
                            f"unique at-risk students "
                            f"(<b>{top_problem['Share of At-Risk Students (%)']:.1f}%</b>). "
                            "This is the strongest operational area to examine when planning "
                            "early support, although prevalence does not establish causality."
                        )
                    )
                    pplot = problem_df.sort_values("At-Risk Students", ascending=True)
                    fig, ax = plt.subplots(figsize=(9, 5.5))
                    ax.barh(pplot["Problem"], pplot["At-Risk Students"])
                    ax.set_xlabel("Unique at-risk students")
                    ax.set_title("Most Common Early-Warning Problems")
                    add_chart(fig)
                    story.append(Paragraph(
                        (
                            f"<b>Graph interpretation:</b> The bars rank warning signals by the "
                            f"number of unique at-risk students affected. "
                            f"<b>{top_problem['Problem']}</b> is the most common signal, while "
                            f"<b>{bottom_problem['Problem']}</b> is the least common of the "
                            "signals measured. Because categories overlap, the bars should not "
                            "be added together to estimate the total number of at-risk students."
                        ),
                        styles["Insight"]
                    ))

                # ---------- Engagement & assessment ----------
                add_section(
                    "3. Engagement & Assessment Patterns",
                    (
                        "The following graphs compare average early engagement and assessment "
                        "measures for At Risk and Not At Risk students. Each measure is displayed "
                        "separately because the metrics use different scales."
                    )
                )
                for col, label in comparison_features:
                    if col not in filtered.columns:
                        continue
                    av = safe_mean(filtered, col, "At Risk")
                    nv = safe_mean(filtered, col, "Not At Risk")
                    if pd.isna(av) and pd.isna(nv):
                        continue
                    fig, ax = plt.subplots(figsize=(8.5, 4.5))
                    ax.bar(["At Risk", "Not At Risk"], [av, nv])
                    ax.set_ylabel(f"Average {label}")
                    ax.set_title(f"Average {label}: At Risk vs Not At Risk")
                    add_chart(fig)
                    story.append(Paragraph(
                        (
                            f"<b>Graph interpretation:</b> Average {label} is "
                            f"<b>{av:.2f}</b> for At Risk records and "
                            f"<b>{nv:.2f}</b> for Not At Risk records. "
                            f"{pct_change_phrase(av, nv)} "
                            "This comparison is an association in the current data and should "
                            "not be treated as proof of cause."
                        ),
                        styles["Insight"]
                    ))

                # ---------- Academic history ----------
                add_section(
                    "4. Academic History & Workload",
                    (
                        "Academic history and workload provide context about whether prior "
                        "study experience and current credit load differ between predicted "
                        "risk groups."
                    )
                )
                for col, label in academic_features:
                    if col not in filtered.columns:
                        continue
                    av = safe_mean(filtered, col, "At Risk")
                    nv = safe_mean(filtered, col, "Not At Risk")
                    if pd.isna(av) and pd.isna(nv):
                        continue
                    fig, ax = plt.subplots(figsize=(8.5, 4.5))
                    ax.bar(["At Risk", "Not At Risk"], [av, nv])
                    ax.set_ylabel(f"Average {label}")
                    ax.set_title(f"Average {label}: At Risk vs Not At Risk")
                    add_chart(fig)
                    story.append(Paragraph(
                        (
                            f"<b>Graph interpretation:</b> The At Risk group averages "
                            f"<b>{av:.2f}</b> for {label}, compared with "
                            f"<b>{nv:.2f}</b> for the Not At Risk group. "
                            f"{pct_change_phrase(av, nv)} "
                            "This helps management understand the academic context surrounding "
                            "predicted risk and where additional advising may be useful."
                        ),
                        styles["Insight"]
                    ))

                # ---------- Student profile patterns ----------
                add_section(
                    "5. Student Profile Risk Patterns",
                    (
                        "The next analyses show predicted risk rates across age, gender, "
                        "education, socioeconomic band, disability status and region where "
                        "those fields are available. These comparisons are included for "
                        "descriptive monitoring and fairness review, not as causal explanations."
                    )
                )
                for column, title in profile_specs:
                    if column not in filtered.columns:
                        continue
                    table = categorical_risk_table(filtered, column)
                    if table.empty:
                        continue
                    high = table.iloc[0]
                    low = table.iloc[-1]
                    plot_df = table.sort_values("Risk Rate (%)", ascending=True)
                    fig, ax = plt.subplots(figsize=(9, max(4.2, min(7, len(plot_df) * 0.42))))
                    ax.barh(plot_df[column].astype(str), plot_df["Risk Rate (%)"])
                    ax.set_xlabel("Predicted at-risk rate (%)")
                    ax.set_xlim(0, 100)
                    ax.set_title(title)
                    add_chart(fig)
                    story.append(Paragraph(
                        (
                            f"<b>Graph interpretation:</b> The highest predicted risk rate is "
                            f"observed for <b>{high[column]}</b> at "
                            f"<b>{high['Risk Rate (%)']:.1f}%</b>, while the lowest is "
                            f"<b>{low[column]}</b> at <b>{low['Risk Rate (%)']:.1f}%</b>. "
                            "Differences should be monitored carefully and interpreted alongside "
                            "group size and other academic information. They do not demonstrate "
                            "that the profile characteristic itself causes risk."
                        ),
                        styles["Insight"]
                    ))

                # ---------- Module ----------
                if "code_module" in filtered.columns and "module_df" in locals() and not module_df.empty:
                    high_mod = module_df.iloc[0]
                    low_mod = module_df.iloc[-1]
                    add_section(
                        "6. Module-Level Risk",
                        (
                            "Module-level analysis identifies where predicted academic risk is "
                            "most concentrated and can help academic managers target review and "
                            "support capacity."
                        ),
                        (
                            f"<b>{high_mod['code_module']}</b> has the highest predicted risk "
                            f"rate at <b>{high_mod['Risk Rate (%)']:.1f}%</b>, compared with "
                            f"<b>{low_mod['code_module']}</b> at "
                            f"<b>{low_mod['Risk Rate (%)']:.1f}%</b>."
                        )
                    )
                    mplot = module_df.sort_values("Risk Rate (%)", ascending=True)
                    fig, ax = plt.subplots(figsize=(8.5, 4.8))
                    ax.barh(mplot["code_module"].astype(str), mplot["Risk Rate (%)"])
                    ax.set_xlabel("Predicted at-risk rate (%)")
                    ax.set_xlim(0, 100)
                    ax.set_title("Risk by Module")
                    add_chart(fig)
                    story.append(Paragraph(
                        (
                            "<b>Graph interpretation:</b> The graph compares the proportion of "
                            "enrollment records predicted At Risk in each module. Modules with "
                            "higher rates may warrant closer review, but differences can also "
                            "reflect cohort composition, module design and other factors not "
                            "captured by this analysis."
                        ),
                        styles["Insight"]
                    ))

                # ---------- Highest risk records ----------
                add_section(
                    "7. Priority Student Review",
                    (
                        "The system ranks individual enrollment records by predicted risk "
                        "probability so academic teams can prioritise manual review and "
                        "intervention planning."
                    ),
                    (
                        "High probability should trigger review rather than automatic action. "
                        "The intervention queue should be used together with the student's "
                        "observed support needs and appropriate human judgement."
                    )
                )
                if not top_risk.empty:
                    display_cols = [
                        c for c in [
                            "id_student", "code_module", "code_presentation",
                            "risk_probability_pct", "predicted_risk"
                        ] if c in top_risk.columns
                    ]
                    table_data = [display_cols]
                    for _, row in top_risk[display_cols].head(10).iterrows():
                        vals = []
                        for c in display_cols:
                            value = row[c]
                            if c == "risk_probability_pct" and pd.notna(value):
                                vals.append(f"{float(value):.1f}%")
                            else:
                                vals.append(str(value))
                        table_data.append(vals)
                    tbl = Table(table_data, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D1DB")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    story.append(tbl)

                # ---------- Overall conclusion ----------
                story.append(PageBreak())
                add_section(
                    "8. Overall Conclusion",
                    (
                        f"The system identified <b>{at_risk_count:,}</b> At Risk enrollment "
                        f"records from <b>{total_records:,}</b> records analysed, equivalent "
                        f"to a predicted risk rate of <b>{risk_rate:.1f}%</b>. "
                        "The combined analytics show that academic risk should be considered "
                        "through several lenses: early engagement, assessment behaviour, "
                        "academic history, module context and student-profile monitoring. "
                        "No single graph should be used in isolation."
                    ),
                    (
                        "The recommended business response is to prioritise students with high "
                        "predicted probabilities, review the most prevalent early-warning "
                        "signals, allocate support to modules or groups showing elevated risk, "
                        "and continue monitoring outcomes after interventions. Profile variables "
                        "such as age, gender, disability, region and socioeconomic band should "
                        "be used cautiously for monitoring and fairness assessment rather than "
                        "as reasons for intervention. The model is a decision-support tool and "
                        "does not replace academic judgement."
                    )
                )
                story.append(Spacer(1, 4 * mm))
                story.append(Paragraph(
                    "<b>Interpretation note:</b> Predictions are probabilistic. The analyses "
                    "in this report describe associations in the available data and do not "
                    "establish causal relationships.",
                    styles["Small"]
                ))

                doc.build(story)
                pdf_buffer.seek(0)
                return pdf_buffer.getvalue(), None

            pdf_bytes, pdf_error = build_analytics_pdf()

            if pdf_error:
                st.warning(pdf_error)
            else:
                st.download_button(
                    "↓ Download Professional Analytics PDF Report",
                    pdf_bytes,
                    "student_academic_risk_analytics_report.pdf",
                    "application/pdf",
                    key="analytics_pdf_report_download"
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="page-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="page-subtitle">
Performance, deployment configuration and responsible-use
information for the academic risk model.
</div>
""",
        unsafe_allow_html=True
    )

    if not model_loaded:

        st.error(
            f"Model could not be loaded: {model_error}"
        )

    else:

        st.success(
            "Logistic Regression model ready"
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric(
                "Accuracy",
                "75.04%"
            )

        with m2:
            st.metric(
                "Precision",
                "74.14%"
            )

        with m3:
            st.metric(
                "Recall",
                "80.58%"
            )

        with m4:
            st.metric(
                "F1",
                "77.22%"
            )

        with m5:
            st.metric(
                "ROC-AUC",
                "84.77%"
            )

        st.markdown(
            '<div class="quick-header">Deployment Configuration</div>',
            unsafe_allow_html=True
        )

        config_df = pd.DataFrame(
            [
                [
                    "Model",
                    "Logistic Regression"
                ],
                [
                    "Observation window",
                    "First 6 weeks / Day 42"
                ],
                [
                    "Input features",
                    len(model_features)
                ],
                [
                    "Decision threshold",
                    f"{threshold:.4f}"
                ],
                [
                    "Prediction",
                    "At Risk / Not At Risk"
                ],
                [
                    "Training dataset",
                    "Open University Learning Analytics Dataset (OULAD)"
                ]
            ],
            columns=[
                "Setting",
                "Value"
            ]
        )

        st.dataframe(
            config_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="quick-header">Model Features</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Feature":
                        model_features
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="quick-header">Responsible Use</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "The deployed model was trained and evaluated on OULAD. "
            "The system accepts prepared analytical datasets or raw student data files that follow the expected university data structure. "
            "processed, but it does not guarantee that predictive "
            "performance transfers to another institution or population."
        )

        st.info(
            "Risk predictions should support human academic review "
            "and early intervention rather than act as automatic "
            "decisions about students."
        )