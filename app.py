import streamlit as st
import pandas as pd
import numpy as np
import skops.io as sio

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PVR Technologies - Movie Success Predictor",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    trusted = sio.get_untrusted_types(
        file="model/model.skops"
    )

    return sio.load(
        "model/model.skops",
        trusted=trusted
    )

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/pvr_tales_hollywood_master_cleaned_engineered.csv"
    )


model = load_model()
df = load_data()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎬 PVR Technologies")
st.subheader("Movie Success Rate Predictor")
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Prediction",
    "📊 Model Performance",
    "🔍 Explainability",
    "📈 Data Drift"
])
st.markdown(
    """
    This application predicts whether a movie is likely to be
    classified as a **HIT** or **FLOP** using a tuned machine
    learning model based on historical movie data.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Movie Information")

budget = st.sidebar.number_input(
    "Budget (USD)",
    min_value=0.0,
    value=10000000.0,
    step=1000000.0
)

release_year = st.sidebar.number_input(
    "Release Year",
    min_value=1900,
    max_value=2100,
    value=2026
)

release_month = st.sidebar.number_input(
    "Release Month",
    min_value=1,
    max_value=12,
    value=6
)

release_quarter = st.sidebar.selectbox(
    "Release Quarter",
    [1, 2, 3, 4]
)

release_day = st.sidebar.selectbox(
    "Release Day of Week",
    list(range(7))
)

runtime = st.sidebar.number_input(
    "Runtime (minutes)",
    min_value=1,
    max_value=500,
    value=120
)

language = st.sidebar.text_input(
    "Original Language",
    value="en"
)

country = st.sidebar.text_input(
    "Production Country",
    value="United States"
)

primary_genre = st.sidebar.text_input(
    "Primary Genre",
    value="Drama"
)

secondary_genre = st.sidebar.text_input(
    "Secondary Genre",
    value="Unknown"
)

genre_count = st.sidebar.number_input(
    "Genre Count",
    min_value=1,
    max_value=10,
    value=1
)

is_sequel = st.sidebar.selectbox(
    "Is Sequel?",
    [0, 1]
)

is_franchise = st.sidebar.selectbox(
    "Is Franchise?",
    [0, 1]
)

production_companies = st.sidebar.number_input(
    "Number of Production Companies",
    min_value=0,
    max_value=50,
    value=1
)

# --------------------------------------------------
# PEOPLE / HISTORICAL FEATURES
# --------------------------------------------------

st.sidebar.subheader("Historical Information")

director_movie_count = st.sidebar.number_input(
    "Director Movies Before Release",
    min_value=0,
    value=0
)

director_hit_count = st.sidebar.number_input(
    "Director Hit Count Before Release",
    min_value=0,
    value=0
)

director_flop_count = st.sidebar.number_input(
    "Director Flop Count Before Release",
    min_value=0,
    value=0
)

director_hit_rate = st.sidebar.number_input(
    "Director Hit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

director_avg_revenue = st.sidebar.number_input(
    "Director Average Revenue",
    min_value=0.0,
    value=0.0
)

director_avg_rating = st.sidebar.number_input(
    "Director Average Rating",
    min_value=0.0,
    max_value=10.0,
    value=0.0
)

actor1_movie_count = st.sidebar.number_input(
    "Actor 1 Movies Before Release",
    min_value=0,
    value=0
)

actor1_hit_rate = st.sidebar.number_input(
    "Actor 1 Hit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

actor1_avg_revenue = st.sidebar.number_input(
    "Actor 1 Average Revenue",
    min_value=0.0,
    value=0.0
)

actor2_movie_count = st.sidebar.number_input(
    "Actor 2 Movies Before Release",
    min_value=0,
    value=0
)

actor2_hit_rate = st.sidebar.number_input(
    "Actor 2 Hit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

actor2_avg_revenue = st.sidebar.number_input(
    "Actor 2 Average Revenue",
    min_value=0.0,
    value=0.0
)

actor3_movie_count = st.sidebar.number_input(
    "Actor 3 Movies Before Release",
    min_value=0,
    value=0
)

actor3_hit_rate = st.sidebar.number_input(
    "Actor 3 Hit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

actor3_avg_revenue = st.sidebar.number_input(
    "Actor 3 Average Revenue",
    min_value=0.0,
    value=0.0
)

average_actor_success = st.sidebar.number_input(
    "Average Lead Actor Success Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

known_stars = st.sidebar.number_input(
    "Number of Known Stars",
    min_value=0,
    max_value=20,
    value=0
)

franchise_movie_count = st.sidebar.number_input(
    "Franchise Movies Before Release",
    min_value=0,
    value=0
)

franchise_hit_rate = st.sidebar.number_input(
    "Franchise Previous Hit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

previous_movie_revenue = st.sidebar.number_input(
    "Previous Movie Revenue",
    min_value=0.0,
    value=0.0
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Predict Movie Success",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "budget_usd": [budget],
        "release_year": [release_year],
        "release_month": [release_month],
        "release_quarter": [release_quarter],
        "release_day_of_week": [release_day],
        "runtime": [runtime],

        "original_language": [language],
        "production_country": [country],
        "primary_genre": [primary_genre],
        "secondary_genre": [secondary_genre],

        "genre_count": [genre_count],
        "is_sequel": [is_sequel],
        "is_franchise": [is_franchise],

        "number_of_production_companies": [
            production_companies
        ],

        "director_movie_count_before_release": [
            director_movie_count
        ],

        "director_hit_count_before_release": [
            director_hit_count
        ],

        "director_flop_count_before_release": [
            director_flop_count
        ],

        "director_hit_rate_before_release": [
            director_hit_rate
        ],

        "director_average_revenue_before_release": [
            director_avg_revenue
        ],

        "director_average_rating_before_release": [
            director_avg_rating
        ],

        "actor_1_movie_count_before_release": [
            actor1_movie_count
        ],

        "actor_1_hit_rate_before_release": [
            actor1_hit_rate
        ],

        "actor_1_average_revenue_before_release": [
            actor1_avg_revenue
        ],

        "actor_2_movie_count_before_release": [
            actor2_movie_count
        ],

        "actor_2_hit_rate_before_release": [
            actor2_hit_rate
        ],

        "actor_2_average_revenue_before_release": [
            actor2_avg_revenue
        ],

        "actor_3_movie_count_before_release": [
            actor3_movie_count
        ],

        "actor_3_hit_rate_before_release": [
            actor3_hit_rate
        ],

        "actor_3_average_revenue_before_release": [
            actor3_avg_revenue
        ],

        "average_lead_actor_historical_success_rate": [
            average_actor_success
        ],

        "number_of_known_stars": [
            known_stars
        ],

        "franchise_movie_count_before_release": [
            franchise_movie_count
        ],

        "franchise_previous_hit_rate": [
            franchise_hit_rate
        ],

        "previous_movie_revenue": [
            previous_movie_revenue
        ]
    })

    # Ensure correct feature order
    input_data = input_data[
        model.feature_names_in_
    ]

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_df = pd.DataFrame({
        "Class": classes,
        "Probability": probability
    })

    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.divider()

    st.header("Prediction Result")

    if prediction == "HIT":
        st.success(
            "🎬 Prediction: HIT"
        )
    else:
        st.error(
            "🎬 Prediction: FLOP"
        )

    predicted_probability = probability[
        list(classes).index(prediction)
    ]

    st.metric(
        "Prediction Probability",
        f"{predicted_probability * 100:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    st.dataframe(
        probability_df,
        use_container_width=True
    )