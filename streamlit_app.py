import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Steam Pricenator", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_lift = load_lottieurl("https://lottie.host/f6d45eb0-9842-4afd-acb0-14233f73966c/K028S2Ke2X.json")
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
        "**Alert**: since the database csv is a little bit large and we can't afford the network costs, we will use a fixed .csv file you will need to download at *fronkongames* Kaggle profile: [Kaggle URL](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.json)."
    )
    set.markdown(
        "**Alert**: the raw games.csv file from *fronkongames* on Kaggle has a known header bug: the header row declares 39 columns because a comma is missing between Discount and DLC count (DiscountDLC count), while every data row has 40 values."
    )

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))
with row1_1:
    st.markdown(
        "**To begin please upload the .csv Kaggle dataset file from [fronkongames](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.json)**"
    )

row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
with row2_1:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

@st.cache_data
def read_uploaded_csv(uploaded_file):
    return pd.read_csv(uploaded_file)
    
if uploaded_file is not None:
    df = read_uploaded_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head(100))
