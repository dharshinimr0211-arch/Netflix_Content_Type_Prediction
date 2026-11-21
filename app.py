
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Netflix Content Type Prediction",
    page_icon="🎬",
    layout="wide"
)

# Colour theme
BLACK = "#000000"
BLUE = "#242f9b"
LAVENDER = "#d1d5f6"
WHITE = "#ffffff"

# Custom styling
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"],
[data-testid="stHeader"] {
    background-color: #000000;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #242f9b;
}
h1, h2, h3, h4, p, label {
    color: #ffffff;
}
h1 {
    font-size: 2.4rem;
    margin-top: 0.5rem;
}
div[data-testid="stMetric"] {
    background-color: #000000;
    border: 1px solid #242f9b;
    padding: 16px;
    border-radius: 8px;
}
div[data-testid="stMetricValue"] {
    color: #d1d5f6;
}
div[data-testid="stMetricLabel"] {
    color: #ffffff;
}
.stButton > button,
.stFormSubmitButton > button {
    background-color: #242f9b;
    color: #ffffff;
    border: 1px solid #d1d5f6;
    border-radius: 6px;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #d1d5f6;
    color: #000000;
}
hr {
    border-color: #242f9b;
}
</style>
""", unsafe_allow_html=True)

# Load dataset and model
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "netflix_content_type_model.pkl")

df = load_data()
model = load_model()

# Apply chart styling
def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=WHITE, size=12),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_xaxes(gridcolor="#333333")
    fig.update_yaxes(gridcolor="#333333")
    return fig

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Content Analysis", "Model Performance", "Predict Content"]
)

# Overview
if page == "Overview":

    st.title("🎬 Netflix Content Type Prediction")
    st.write("Explore Netflix content distribution, historical trends and machine learning predictions.")

    st.divider()

    total_titles = len(df)
    total_movies = (df["type"] == "Movie").sum()
    total_shows = (df["type"] == "TV Show").sum()
    total_countries = df["country"].dropna().str.split(", ").explode().nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Titles", f"{total_titles:,}")
    col2.metric("Movies", f"{total_movies:,}")
    col3.metric("TV Shows", f"{total_shows:,}")
    col4.metric("Countries", f"{total_countries:,}")

    st.divider()
    st.subheader("Content Type Distribution")

    type_counts = df["type"].value_counts().rename_axis("Content Type").reset_index(name="Count")

    fig = px.pie(
        type_counts,
        names="Content Type",
        values="Count",
        hole=0.5,
        color="Content Type",
        color_discrete_map={"Movie": BLUE, "TV Show": LAVENDER}
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Content Released Over the Years")

    year_counts = df.groupby(["release_year", "type"]).size().reset_index(name="Count")

    fig = px.line(
        year_counts,
        x="release_year",
        y="Count",
        color="type",
        color_discrete_map={"Movie": BLUE, "TV Show": LAVENDER},
        labels={
            "release_year": "Release Year",
            "Count": "Number of Titles",
            "type": "Content Type"
        }
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# Content analysis
elif page == "Content Analysis":

    st.title("Content Analysis")
    st.write("Explore Netflix content across ratings, countries and genres.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Movies vs TV Shows")

        type_counts = df["type"].value_counts().rename_axis("Type").reset_index(name="Count")

        fig = px.pie(
            type_counts,
            names="Type",
            values="Count",
            hole=0.5,
            color="Type",
            color_discrete_map={"Movie": BLUE, "TV Show": LAVENDER}
        )
        fig.update_traces(textinfo="percent+label")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Content Ratings")

        rating_counts = df["rating"].fillna("Not Given").value_counts().reset_index()
        rating_counts.columns = ["Rating", "Count"]

        fig = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            color_discrete_sequence=[LAVENDER]
        )
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Top 10 Countries")

    country_data = df.copy()
    country_data["country"] = country_data["country"].fillna("Not Given")

    country_counts = (
        country_data["country"]
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
        .reset_index()
    )
    country_counts.columns = ["Country", "Titles"]

    fig = px.bar(
        country_counts,
        x="Titles",
        y="Country",
        orientation="h",
        text="Titles",
        color_discrete_sequence=[BLUE]
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Popular Genres")

    genre_counts = (
        df["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
        .reset_index()
    )
    genre_counts.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_counts,
        x="Count",
        y="Genre",
        orientation="h",
        text="Count",
        color_discrete_sequence=[LAVENDER]
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# Model performance
elif page == "Model Performance":

    st.title("Model Performance")
    st.write("Evaluation of the tuned Decision Tree classification model.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", "71.16%")
    col2.metric("Macro F1-score", "0.69")
    col3.metric("TV Show Recall", "0.74")

    st.divider()
    st.subheader("Classification Report")

    report = pd.DataFrame({
        "Class": ["Movie", "TV Show", "Macro Average", "Weighted Average"],
        "Precision": [0.86, 0.52, 0.69, 0.76],
        "Recall": [0.70, 0.74, 0.72, 0.71],
        "F1-score": [0.77, 0.61, 0.69, 0.72]
    })

    st.dataframe(report, use_container_width=True, hide_index=True)

    st.subheader("Confusion Matrix")

    confusion = pd.DataFrame(
        [[858, 367], [140, 393]],
        index=["Actual Movie", "Actual TV Show"],
        columns=["Predicted Movie", "Predicted TV Show"]
    )

    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale=[BLACK, BLUE, LAVENDER],
        aspect="auto"
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model Details")

    st.write("Algorithm: Decision Tree Classifier")
    st.write("Input Features: Release Year and Country")
    st.write("Class Weight: Balanced")
    st.write("Hyperparameter Tuning: GridSearchCV")
    st.write("Evaluation Metric: Macro F1-score")

    st.info(
        "The tuned model improves TV Show detection compared "
        "with the original Decision Tree, although overall accuracy is lower."
    )

# Prediction
elif page == "Predict Content":

    st.title("Predict Content Type")
    st.write("Enter the release year and country to predict whether the content is a Movie or TV Show.")
    st.divider()

    with st.form("prediction_form"):

        release_year = st.number_input(
            "Release Year",
            min_value=int(df["release_year"].min()),
            max_value=int(df["release_year"].max()),
            value=2020,
            step=1
        )

        countries = sorted(df["country"].dropna().unique().tolist())

        country = st.selectbox("Select Country", countries)

        submitted = st.form_submit_button("Predict Content Type")

    if submitted:

        input_data = pd.DataFrame({
            "release_year": [release_year],
            "country": [country]
        })

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "Movie":
            st.success("Predicted Content Type: Movie")
        else:
            st.success("Predicted Content Type: TV Show")

        st.caption("Prediction generated using the trained Decision Tree model.")
