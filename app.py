 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import skops.io as sio

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PVR Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #ddd;
        padding: 12px;
        border-radius: 10px;
    }

    .small-note {
        font-size: 0.85rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 PVR Technologies</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Movie Success Predictor — Predict whether a movie is likely to be a HIT or FLOP'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "model/model.skops"

DATA_PATH = (
    "data/pvr_tales_hollywood_master_cleaned_engineered.csv"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    untrusted_types = sio.get_untrusted_types(
        file=MODEL_PATH
    )

    model = sio.load(
        MODEL_PATH,
        trusted=untrusted_types
    )

    return model


@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:

    model = load_model()
    df = load_data()

except Exception as e:

    st.error("Error loading model or dataset.")
    st.exception(e)
    st.stop()


# ============================================================
# MODEL COMPONENTS
# ============================================================

preprocessor = model.named_steps["preprocessor"]

rf_model = model.named_steps["model"]

feature_columns = list(model.feature_names_in_)


# ============================================================
# CREATE ENGINEERED FEATURES IF NECESSARY
# ============================================================

if "genre_count" not in df.columns:

    if "combined_genre" in df.columns:

        df["genre_count"] = (
            df["combined_genre"]
            .fillna("")
            .apply(
                lambda x: len(
                    [
                        g
                        for g in str(x).split(",")
                        if g.strip()
                    ]
                )
            )
        )

    else:

        df["genre_count"] = 0


if "number_of_production_companies" not in df.columns:

    if "production_company" in df.columns:

        df["number_of_production_companies"] = (
            df["production_company"]
            .fillna("")
            .apply(
                lambda x: len(
                    [
                        c
                        for c in str(x).split(",")
                        if c.strip()
                    ]
                )
            )
        )

    else:

        df["number_of_production_companies"] = 0


# ============================================================
# HISTORICAL FEATURES
# ============================================================

historical_features = [

    "director_movie_count_before_release",
    "director_hit_count_before_release",
    "director_flop_count_before_release",
    "director_hit_rate_before_release",
    "director_average_revenue_before_release",
    "director_average_rating_before_release",

    "actor_1_movie_count_before_release",
    "actor_1_hit_rate_before_release",
    "actor_1_average_revenue_before_release",

    "actor_2_movie_count_before_release",
    "actor_2_hit_rate_before_release",
    "actor_2_average_revenue_before_release",

    "actor_3_movie_count_before_release",
    "actor_3_hit_rate_before_release",
    "actor_3_average_revenue_before_release",

    "average_lead_actor_historical_success_rate",
    "number_of_known_stars",

    "franchise_movie_count_before_release",
    "franchise_previous_hit_rate",
    "previous_movie_revenue"
]

for col in historical_features:

    if col not in df.columns:

        df[col] = 0


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎬 Prediction",
        "📊 Model Performance",
        "🔍 Explainability",
        "📈 Data Drift"
    ]
)


# ============================================================
# TAB 1 - PREDICTION
# ============================================================

with tab1:

    st.header("🎬 Movie Success Prediction")

    st.info(
        "Enter the movie details below. "
        "Use the dropdowns and sliders to make the prediction easier."
    )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    st.subheader("🎥 Basic Movie Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        budget_usd = st.number_input(
            "Budget (USD)",
            min_value=0.0,
            max_value=1_000_000_000.0,
            value=10_000_000.0,
            step=1_000_000.0,
            help="Estimated production budget of the movie in US dollars."
        )

    with col2:

        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2100,
            value=2025,
            step=1,
            help="Year in which the movie is released."
        )

    with col3:

        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=30.0,
            max_value=300.0,
            value=120.0,
            step=5.0,
            help="Approximate running time of the movie."
        )


    # ========================================================
    # RELEASE INFORMATION
    # ========================================================

    st.subheader("📅 Release Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        release_month = st.selectbox(
            "Release Month",
            options=list(range(1, 13)),
            format_func=lambda x: [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ][x - 1],
            index=5,
            help="Select the month in which the movie will be released."
        )

    with col2:

        release_quarter = st.selectbox(
            "Release Quarter",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            index=1,
            help="Quarter corresponding to the release month."
        )

    with col3:

        release_day_of_week = st.selectbox(
            "Release Day",
            options=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ],
            index=4,
            help="Day of the week on which the movie is released."
        )

    with col4:

        number_of_production_companies = st.number_input(
            "Production Companies",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
            help="Number of production companies involved in the movie."
        )


    # ========================================================
    # GENRE
    # ========================================================

    st.subheader("🎭 Genre Information")

    genre_options = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Science Fiction",
        "Thriller",
        "War",
        "Western"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        primary_genre = st.selectbox(
            "Primary Genre",
            genre_options,
            index=6,
            help="Main genre of the movie."
        )

    with col2:

        secondary_genre = st.selectbox(
            "Secondary Genre",
            ["None"] + genre_options,
            index=4,
            help="Optional secondary genre."
        )

    with col3:

        genre_count = st.selectbox(
            "Number of Genres",
            options=[1, 2, 3, 4, 5],
            index=1,
            help="Total number of genres associated with the movie."
        )


    # ========================================================
    # LANGUAGE AND COUNTRY
    # ========================================================

    st.subheader("🌍 Language & Production")

    col1, col2 = st.columns(2)

    language_options = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "ja",
        "ko",
        "zh",
        "hi"
    ]

    country_options = [
        "United States",
        "United Kingdom",
        "Canada",
        "India",
        "France",
        "Germany",
        "Australia",
        "Japan",
        "South Korea",
        "China"
    ]

    with col1:

        original_language = st.selectbox(
            "Original Language",
            language_options,
            index=0,
            help="Original language of the movie."
        )

    with col2:

        production_country = st.selectbox(
            "Production Country",
            country_options,
            index=0,
            help="Main country associated with production."
        )


    # ========================================================
    # SEQUEL / FRANCHISE
    # ========================================================

    st.subheader("🔗 Movie Relationship")

    col1, col2 = st.columns(2)

    with col1:

        is_sequel_text = st.radio(
            "Is this a sequel?",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the movie continues a previous movie."
        )

        is_sequel = 1 if is_sequel_text == "Yes" else 0

    with col2:

        is_franchise_text = st.radio(
            "Is this part of a franchise?",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the movie belongs to an established franchise."
        )

        is_franchise = 1 if is_franchise_text == "Yes" else 0


    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    with st.expander(
        "⚙️ Advanced Historical Information",
        expanded=False
    ):

        st.info(
            "These features are required by the trained model. "
            "When historical information is unavailable, the "
            "current project uses 0 as the default value."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**🎬 Director History**")

            director_movie_count_before_release = st.number_input(
                "Director Previous Movies",
                min_value=0,
                value=0,
                help="Number of movies previously directed before this release."
            )

            director_hit_count_before_release = st.number_input(
                "Director Previous Hits",
                min_value=0,
                value=0,
                help="Number of previous successful movies by the director."
            )

            director_flop_count_before_release = st.number_input(
                "Director Previous Flops",
                min_value=0,
                value=0,
                help="Number of previous unsuccessful movies by the director."
            )

            director_hit_rate_before_release = st.slider(
                "Director Historical Hit Rate",
                0.0,
                1.0,
                0.0,
                0.01,
                help="Historical success rate of the director."
            )

            director_average_revenue_before_release = st.number_input(
                "Director Average Revenue",
                min_value=0.0,
                value=0.0,
                step=1_000_000.0
            )

            director_average_rating_before_release = st.slider(
                "Director Average Rating",
                0.0,
                10.0,
                0.0,
                0.1
            )

        with col2:

            st.markdown("**⭐ Actor History**")

            actor_1_movie_count_before_release = st.number_input(
                "Actor 1 Previous Movies",
                min_value=0,
                value=0
            )

            actor_1_hit_rate_before_release = st.slider(
                "Actor 1 Hit Rate",
                0.0,
                1.0,
                0.0,
                0.01
            )

            actor_1_average_revenue_before_release = st.number_input(
                "Actor 1 Average Revenue",
                min_value=0.0,
                value=0.0,
                step=1_000_000.0
            )

            actor_2_movie_count_before_release = st.number_input(
                "Actor 2 Previous Movies",
                min_value=0,
                value=0
            )

            actor_2_hit_rate_before_release = st.slider(
                "Actor 2 Hit Rate",
                0.0,
                1.0,
                0.0,
                0.01
            )

            actor_2_average_revenue_before_release = st.number_input(
                "Actor 2 Average Revenue",
                min_value=0.0,
                value=0.0,
                step=1_000_000.0
            )

            actor_3_movie_count_before_release = st.number_input(
                "Actor 3 Previous Movies",
                min_value=0,
                value=0
            )

            actor_3_hit_rate_before_release = st.slider(
                "Actor 3 Hit Rate",
                0.0,
                1.0,
                0.0,
                0.01
            )

            actor_3_average_revenue_before_release = st.number_input(
                "Actor 3 Average Revenue",
                min_value=0.0,
                value=0.0,
                step=1_000_000.0
            )

        st.markdown("**🏆 Overall & Franchise History**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            average_lead_actor_historical_success_rate = st.slider(
                "Lead Actor Success Rate",
                0.0,
                1.0,
                0.0,
                0.01
            )

        with col2:

            number_of_known_stars = st.number_input(
                "Known Stars",
                min_value=0,
                max_value=20,
                value=0
            )

        with col3:

            franchise_movie_count_before_release = st.number_input(
                "Previous Franchise Movies",
                min_value=0,
                value=0
            )

        with col4:

            franchise_previous_hit_rate = st.slider(
                "Franchise Previous Hit Rate",
                0.0,
                1.0,
                0.0,
                0.01
            )

        previous_movie_revenue = st.number_input(
            "Previous Movie Revenue",
            min_value=0.0,
            value=0.0,
            step=1_000_000.0
        )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        {
            "budget_usd": [budget_usd],
            "release_year": [release_year],
            "release_month": [release_month],
            "release_quarter": [release_quarter],

            "release_day_of_week": [
                release_day_of_week
            ],

            "runtime": [runtime],

            "original_language": [
                original_language
            ],

            "production_country": [
                production_country
            ],

            "primary_genre": [
                primary_genre
            ],

            "secondary_genre": [
                secondary_genre
            ],

            "genre_count": [genre_count],

            "is_sequel": [is_sequel],

            "is_franchise": [is_franchise],

            "number_of_production_companies": [
                number_of_production_companies
            ],

            "director_movie_count_before_release": [
                director_movie_count_before_release
            ],

            "director_hit_count_before_release": [
                director_hit_count_before_release
            ],

            "director_flop_count_before_release": [
                director_flop_count_before_release
            ],

            "director_hit_rate_before_release": [
                director_hit_rate_before_release
            ],

            "director_average_revenue_before_release": [
                director_average_revenue_before_release
            ],

            "director_average_rating_before_release": [
                director_average_rating_before_release
            ],

            "actor_1_movie_count_before_release": [
                actor_1_movie_count_before_release
            ],

            "actor_1_hit_rate_before_release": [
                actor_1_hit_rate_before_release
            ],

            "actor_1_average_revenue_before_release": [
                actor_1_average_revenue_before_release
            ],

            "actor_2_movie_count_before_release": [
                actor_2_movie_count_before_release
            ],

            "actor_2_hit_rate_before_release": [
                actor_2_hit_rate_before_release
            ],

            "actor_2_average_revenue_before_release": [
                actor_2_average_revenue_before_release
            ],

            "actor_3_movie_count_before_release": [
                actor_3_movie_count_before_release
            ],

            "actor_3_hit_rate_before_release": [
                actor_3_hit_rate_before_release
            ],

            "actor_3_average_revenue_before_release": [
                actor_3_average_revenue_before_release
            ],

            "average_lead_actor_historical_success_rate": [
                average_lead_actor_historical_success_rate
            ],

            "number_of_known_stars": [
                number_of_known_stars
            ],

            "franchise_movie_count_before_release": [
                franchise_movie_count_before_release
            ],

            "franchise_previous_hit_rate": [
                franchise_previous_hit_rate
            ],

            "previous_movie_revenue": [
                previous_movie_revenue
            ]
        }
    )

    # Force exact model feature order

    input_data = input_data[feature_columns]


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("---")

    if st.button(
        "🔮 Predict Movie Success",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = list(
                model.classes_
            )

            probability_dict = dict(
                zip(
                    classes,
                    probabilities
                )
            )

            hit_probability = probability_dict.get(
                "HIT",
                0
            )

            flop_probability = probability_dict.get(
                "FLOP",
                0
            )

            st.divider()

            # =================================================
            # RESULT
            # =================================================

            if prediction == "HIT":

                st.success(
                    "🎉 **Prediction: HIT**"
                )

                st.balloons()

            else:

                st.error(
                    "📉 **Prediction: FLOP**"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🎯 HIT Probability",
                    f"{hit_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "📉 FLOP Probability",
                    f"{flop_probability * 100:.2f}%"
                )

            probability_df = pd.DataFrame(
                {
                    "Outcome": [
                        "FLOP",
                        "HIT"
                    ],

                    "Probability": [
                        flop_probability,
                        hit_probability
                    ]
                }
            )

            st.subheader(
                "Prediction Probability"
            )

            st.bar_chart(
                probability_df.set_index(
                    "Outcome"
                )
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# TAB 2 - MODEL PERFORMANCE
# ============================================================

with tab2:

    st.header("📊 Model Performance")

    st.caption(
        "Evaluation of the trained Random Forest classifier "
        "using labeled HIT/FLOP records."
    )

    labeled_df = df[
        df["movie_success"].isin(
            ["HIT", "FLOP"]
        )
    ].copy()

    X = labeled_df[
        feature_columns
    ]

    y = labeled_df[
        "movie_success"
    ]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="HIT"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="HIT"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="HIT"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    st.divider()

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader(
        "Confusion Matrix"
    )

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    ax.set_title(
        "Movie Success Prediction - Confusion Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.subheader(
        "Model Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    with col2:

        st.metric(
            "Estimators",
            rf_model.n_estimators
        )

    with col3:

        st.metric(
            "Input Features",
            len(feature_columns)
        )

    with col4:

        st.metric(
            "Transformed Features",
            len(
                preprocessor.get_feature_names_out()
            )
        )


# ============================================================
# TAB 3 - SHAP
# ============================================================

with tab3:

    st.header(
        "🔍 Model Explainability"
    )

    st.write(
        "SHAP explains how individual features contribute "
        "to the model's predictions."
    )

    st.info(
        "The analysis explains the model's HIT-class predictions "
        "using the same preprocessing pipeline and Random Forest model."
    )

    labeled_df = df[
        df["movie_success"].isin(
            ["HIT", "FLOP"]
        )
    ].copy()

    @st.cache_resource
    def calculate_shap():

        X_shap = labeled_df[
            feature_columns
        ].copy()

        X_transformed = preprocessor.transform(
            X_shap
        )

        shap_feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        X_sample = X_transformed[:200]

        explainer = shap.TreeExplainer(
            rf_model
        )

        shap_values = explainer.shap_values(
            X_sample
        )

        if isinstance(
            shap_values,
            list
        ):

            hit_index = list(
                rf_model.classes_
            ).index("HIT")

            hit_values = shap_values[
                hit_index
            ]

        else:

            hit_index = list(
                rf_model.classes_
            ).index("HIT")

            if shap_values.ndim == 3:

                hit_values = shap_values[
                    :,
                    :,
                    hit_index
                ]

            else:

                hit_values = shap_values

        return (
            hit_values,
            X_sample,
            shap_feature_names
        )

    try:

        (
            hit_shap_values,
            X_shap_sample,
            shap_feature_names
        ) = calculate_shap()

        st.subheader(
            "Top Features Influencing HIT Prediction"
        )

        fig, ax = plt.subplots()

        shap.summary_plot(
            hit_shap_values,
            X_shap_sample,
            feature_names=shap_feature_names,
            plot_type="bar",
            show=False
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader(
            "SHAP Feature Impact Distribution"
        )

        fig, ax = plt.subplots()

        shap.summary_plot(
            hit_shap_values,
            X_shap_sample,
            feature_names=shap_feature_names,
            show=False
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "SHAP Samples",
                len(X_shap_sample)
            )

        with col2:

            st.metric(
                "Transformed Features",
                len(shap_feature_names)
            )

    except Exception as e:

        st.error(
            "SHAP analysis could not be generated."
        )

        st.exception(e)


# ============================================================
# TAB 4 - DATA DRIFT
# ============================================================

with tab4:

    st.header(
        "📈 Data Drift Monitoring"
    )

    st.write(
        "A simple statistical comparison between an earlier "
        "portion and a later portion of the dataset."
    )

    st.info(
        "This is a simple project-level monitoring check, "
        "not a formal production drift-monitoring system."
    )

    numeric_features = [

        "budget_usd",
        "release_year",
        "release_month",
        "runtime",
        "genre_count",
        "number_of_production_companies",
        "number_of_known_stars",
        "previous_movie_revenue"
    ]

    available_features = [
        col
        for col in numeric_features
        if col in df.columns
    ]

    drift_df = df[
        available_features
    ].copy()

    midpoint = len(
        drift_df
    ) // 2

    reference_data = drift_df.iloc[
        :midpoint
    ]

    current_data = drift_df.iloc[
        midpoint:
    ]

    drift_results = []

    for feature in available_features:

        reference_mean = (
            reference_data[feature]
            .mean()
        )

        current_mean = (
            current_data[feature]
            .mean()
        )

        if (
            reference_mean is not None
            and not pd.isna(reference_mean)
            and reference_mean != 0
        ):

            change_percent = (
                (
                    current_mean
                    - reference_mean
                )
                / abs(reference_mean)
            ) * 100

        else:

            change_percent = 0

        drift_results.append(
            {
                "Feature": feature,

                "Reference Mean":
                    reference_mean,

                "Current Mean":
                    current_mean,

                "Change (%)":
                    change_percent
            }
        )

    drift_results_df = pd.DataFrame(
        drift_results
    )

    st.subheader(
        "Feature Distribution Changes"
    )

    st.dataframe(
        drift_results_df.round(2),
        use_container_width=True
    )

    st.subheader(
        "Drift Flags"
    )

    drift_results_df[
        "Drift Status"
    ] = (
        drift_results_df[
            "Change (%)"
        ]
        .abs()
        .apply(
            lambda x:
                "⚠️ Potential Change"
                if x >= 20
                else "✅ Stable"
        )
    )

    st.dataframe(
        drift_results_df[
            [
                "Feature",
                "Change (%)",
                "Drift Status"
            ]
        ].round(2),
        use_container_width=True
    )

    st.caption(
        "A 20% mean change is used only as a simple "
        "project monitoring threshold."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🎬 PVR Movie Predictor"
    )

    st.caption(
        "Advanced Data Science Project"
    )

    st.divider()

    st.markdown(
        "**🤖 Model**"
    )

    st.write(
        "Random Forest Classifier"
    )

    st.markdown(
        "**🎯 Prediction Classes**"
    )

    st.write(
        "FLOP / HIT"
    )

    st.markdown(
        "**📥 Input Features**"
    )

    st.write(
        len(feature_columns)
    )

    st.markdown(
        "**⚙️ Transformed Features**"
    )

    st.write(
        len(
            preprocessor.get_feature_names_out()
        )
    )

    st.divider()

    st.markdown(
        "**Dashboard Sections**"
    )

    st.write(
        "🎬 Prediction"
    )

    st.write(
        "📊 Model Performance"
    )

    st.write(
        "🔍 SHAP Explainability"
    )

    st.write(
        "📈 Data Drift"
    )

    st.divider()

    st.caption(
        "PVR Technologies | Advanced Data Science Experiment 8"
    )
 
