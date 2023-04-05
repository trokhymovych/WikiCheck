import urllib.request
import numpy as np
import pandas as pd
import requests
import streamlit as st
import glob
import altair as alt
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.altex import line_chart, get_stocks_data
from streamlit_lottie import st_lottie

# Header organisation
st.set_page_config(page_title="WikiCheck Analytics page", layout="wide", page_icon="🧿")
# Add some animation:
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_book = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_zex3p7vu.json")
row0_spacer1, row0_1, row0_spacer2, row0_2 = st.columns(
    (1, 3, 0.5, 3)
)
with row0_spacer1:
    st_lottie(lottie_book, speed=1, height=150, key="initial", )
row0_1.title(" WikiCheck analytics page ")

# Adding plots:
# Loading the data:
@st.cache_data
def get_logs():
    last_log_filename = np.sort(glob.glob("logs/*.csv"))[-1]
    data = pd.read_csv(last_log_filename)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["date"] = data['datetime'].dt.date
    return data

# reassigning rows
row0_spacer1, row0_1, row0_spacer2, row0_2 = st.columns(
    (1, 3, 0.5, 3)
)

# plot request per day:
with row0_1:
    data = get_logs()

    df_agg = data.groupby(["model_name", "date"])["request"].count().reset_index()
    df_agg["date"] = df_agg["date"].apply(str)
    df_agg.columns = ["model_name", "date", "number_of_requests"]

    a = alt.Chart(df_agg, title="Number of requests per model").mark_area(opacity=0.4).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_requests:Q", title="Number of requests", stack=None),
        color=alt.Color("model_name:N", title="Model name"),
        tooltip=["model_name", "date", "number_of_requests"]
    )
    b = alt.Chart(df_agg).mark_line(opacity=0.4, point=True).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_requests:Q", title="Number of requests"),
        color=alt.Color("model_name:N", title="Model name"),
        tooltip=["model_name", "date", "number_of_requests"]
    )
    st.altair_chart((a + b).configure_range(category={'scheme': 'tableau10'}), use_container_width=True)

with row0_2:
    data = get_logs()
    df_agg = data.groupby(["model_name", "date"])["ip"].nunique().reset_index()
    df_agg.columns = ["model_name", "date", "number_of_unique_users"]
    df_agg["date"] = df_agg["date"].apply(str)
    df_agg.columns = ["model_name", "date", "number_of_unique_users"]

    a = alt.Chart(df_agg, title="Number of unique users per model").mark_area(opacity=0.4).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_unique_users:Q", title="Number of unique users", stack=None),
        color=alt.Color("model_name:N", title="Model name"),
        tooltip=["model_name", "date", "number_of_unique_users"]
    )
    b = alt.Chart(df_agg).mark_line(opacity=0.4, point=True).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_unique_users:Q", title="Number of unique users", stack=None),
        color=alt.Color("model_name:N", title="Model name"),
        tooltip=["model_name", "date", "number_of_unique_users"]
    )
    st.altair_chart((a + b).configure_range(category={'scheme': 'tableau10'}), use_container_width=True)


# reassigning rows
row0_spacer1, row0_1, row0_2 = st.columns(
    (1, 2, 4.5)
)
# plot mean response time:
data = get_logs()
time_metrics_dict = data.groupby("model_name")["time_spend"].mean().to_dict()
for k, v in time_metrics_dict.items():
    with row0_1:
        st.metric(f"{k}   mean response time", round(v, 2))

with row0_2:
    st.write(get_logs().tail(100))
