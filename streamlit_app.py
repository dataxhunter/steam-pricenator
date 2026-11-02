import ast
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Steam Pricenator", layout="wide")


@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None


def count_languages(val):
    if pd.isna(val) or not isinstance(val, str):
        return 0
    try:
        parsed_list = ast.literal_eval(val)
        if isinstance(parsed_list, list):
            return len(parsed_list)
        return 0
    except (ValueError, SyntaxError):
        return 0


@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean_steam_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    # Fix known fronkongames header bug where 'DiscountDLC count' is missing a comma
    first_newline = raw_bytes.find(b"\n")
    if first_newline != -1 and b"DiscountDLC count" in raw_bytes[:first_newline]:
        header = raw_bytes[:first_newline].replace(
            b"DiscountDLC count", b"Discount,DLC count"
        )
        raw_bytes = header + raw_bytes[first_newline:]

    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        encoding="utf-8",
        low_memory=False,
        on_bad_lines="skip",
    )

    # Dataset cleaning
    df["Full audio languages"] = df["Full audio languages"].apply(count_languages)
    df["Platforms count"] = df[["Windows", "Mac", "Linux"]].astype(int).sum(axis=1)
    df["Release date"] = pd.to_datetime(
        df["Release date"], format="mixed", errors="coerce"
    )
    df["Release year"] = df["Release date"].dt.year

    df = df[df["Release year"] >= 1997]
    df = df.dropna(subset=["Genres", "Categories"])
    df = df[df["Price"] > 0.0].copy()

    # Vectorized original price calculation (much faster than .apply)
    if "Discount" in df.columns:
        valid_discount = (df["Discount"] > 0) & (df["Discount"] < 100)
        df["Price"] = np.where(
            valid_discount,
            df["Price"] / (1 - df["Discount"] / 100),
            df["Price"],
        )
        df = df.drop(columns=["Discount"])

    # Show only prices between $1 and $500
    df = df[(df['Price'] >= 1.0) & (df['Price'] <= 500.0)]

    # Transform estimated owners to mean value
    valid_estimated_owners = df["Estimated owners"].str.split(" - ", expand=True)
    df["Estimated Owners"] = np.where(
        valid_estimated_owners.notnull().all(axis=1),
        valid_estimated_owners.astype(float).mean(axis=1),
        np.nan
    )

    df['weight'] = np.log1p(df['Estimated Owners'])

    return df


lottie_lift = load_lottieurl(
    "https://lottie.host/f6d45eb0-9842-4afd-acb0-14233f73966c/K028S2Ke2X.json"
)
if lottie_lift:
    st_lottie(lottie_lift, speed=1, height=200, key="initial")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 1, 0.2, 1, 0.1)
)

row0_1.title("Steam Pricenator")

with row0_2:
    st.space(1)

row0_2.subheader(
    "A Streamlit web app by [Data x Hunter](https://github.com/dataxhunter), follow for more content"
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    st.markdown(
        "Hey there! Welcome to the Steam Pricenator, an expert program that helps game publishers decide the ideal price for a new title on Steam. Based on features like genre, categories, tags and other features, the final project uses a basic algorithm named Decision Tree to predict the price."
    )
    st.markdown(
        "To get there, we explored the “Steam Games Dataset” maintained by *fronkongames* on Kaggle (MIT license, around 400 MB, updated weekly on Steam API). It gathers information on nearly 150,000 games across 39 columns, available as CSV and JSON."
    )
    st.markdown(
        "**Tip**: if you’re on mobile, switch to landscape mode for easier viewing of the graphs. Give it a go!"
    )
    st.markdown(
        "**Alert**: since the database csv is a little bit large and we can't afford the network costs, we will use a fixed .csv file you will need to download at *fronkongames* Kaggle profile: [Kaggle URL](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.csv)."
    )
    st.markdown(
        "**Info**: Both the general metrics and Steam Pricenator prediction model used a **weight** based calculation where games/apps with higher estimated owners influenced more on the result."
    )
    st.markdown(
        "Here is basic formula:"
    )
    st.latex(r'''
    weight = \ln(1 + x) \text{ where x  is the estimated owners count}
    ''')
    st.markdown(
        "**To begin please upload the .csv Kaggle dataset file from [fronkongames](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.csv)**"
    )

row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
with row2_1:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Stop execution here until the user uploads a CSV file
if uploaded_file is None:
    st.stop()

df = load_and_clean_steam_csv(uploaded_file)

with row2_1:
    st.write(f"### Cleaned Dataset Preview ({len(df):,} rows × {len(df.columns)} columns):")
    st.dataframe(df.head(100))

# Visualizations
line1_spacer1, line1_1, line1_spacer2 = st.columns((0.1, 3.2, 0.1))
with line1_1:
    st.header("General Metrics")

row3_space1, row3_1, row3_space2 = st.columns(
    (0.1, 1, 0.1)
)

with row3_1:
    price_by_year = df.groupby("Release year")["Price"].mean().reset_index()
    price_by_year = price_by_year[
        (price_by_year["Release year"] >= 1997)
        & (price_by_year["Release year"] <= 2026)
    ].copy()
    price_by_year["Release year"] = price_by_year["Release year"].astype(int)

    fig = px.bar(
        price_by_year,
        x="Release year",
        y="Price",
        title="Average Price of Paid Games by Release Year",
        labels={"Release year": "Release Year", "Price": "Average Price ($)"},
        color_discrete_sequence=["#9EE6CF"],
        text_auto=".2f",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis=dict(type="category"),
        yaxis=dict(gridcolor="lightgray", gridwidth=1, griddash="dash"),
        hovermode="x unified",
    )
    fig.update_traces(
        hovertemplate="<b>Year:</b> %{x}<br><b>Average Price:</b> $%{y:.2f}<extra></extra>"
    )

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    total_games = len(df)
    overall_avg_price = df["Price"].mean()
    peak_row = price_by_year.loc[price_by_year["Price"].idxmax()]

    st.markdown(
        f"After cleaning, we are analyzing a total of **{total_games:,} paid games** "
        f"with an overall average price of **\${overall_avg_price:.2f}**, with **{int(peak_row['Release year'])}** being the year"
        f"recording the highest average price (**\${peak_row['Price']:.2f}**)."
    )

row4_space1, row4_1, row4_space2 = st.columns(
    (0.1, 1, 0.1)
)

with row4_1:
    df_genres_exploded = df.copy()
    df_genres_exploded['Genres'] = df_genres_exploded['Genres'].str.split(',')
    df_genres_exploded = df_genres_exploded.explode('Genres')

    genre_avg_price = df_genres_exploded.groupby('Genres')['Price'].mean().reset_index()

    genre_avg_price = genre_avg_price.sort_values(by='Price', ascending=False)

    fig = px.bar(
        genre_avg_price,
        x="Price",
        y="Genres",
        title="Average Price of Paid Games by Genre",
        labels={"Genre": "Genre", "Average Price ($)": "Average Price ($)"},
        color_discrete_sequence=["#9EE6CF"],
        text_auto=".2f",
        orientation='h',
    )

    fig.update_traces(
        hovertemplate="<b>Average Price:</b> $%{x:.2f}<br><b>Genre:</b> %{y}<extra></extra>"
    )

    fig.update_xaxes(automargin=True)

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

row5_space1, row5_1, row5_space2 = st.columns(
    (0.1, 1, 0.1)
)

with row5_1:
    df_categories_exploded = df.copy()
    df_categories_exploded['Categories'] = df_categories_exploded['Categories'].str.split(',')
    df_categories_exploded = df_categories_exploded.explode('Categories')

    category_avg_price = df_categories_exploded.groupby('Categories')['Price'].mean().reset_index()

    category_avg_price = category_avg_price.sort_values(by='Price', ascending=False)

    fig = px.bar(
        category_avg_price,
        x="Price",
        y="Categories",
        title="Average Price of Paid Games by Category",
        labels={"Category": "Category", "Average Price ($)": "Average Price ($)"},
        color_discrete_sequence=["#9EE6CF"],
        text_auto=".2f",
        orientation='h',
        height=700
    )

    fig.update_traces(
        hovertemplate="<b>Average Price:</b> $%{x:.2f}<br><b>Category:</b> %{y}<extra></extra>"
    )

    fig.update_xaxes(automargin=True)

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)