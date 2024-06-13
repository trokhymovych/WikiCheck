import urllib.request
import numpy as np
import pandas as pd
import requests
import streamlit as st
import glob
import altair as alt
from pandas import json_normalize
from streamlit_lottie import st_lottie
import json


ALLOWED_PATH = [
    "/docs",
    "/get-fact-check-non-aggregated/", 
    "/get-fact-check-aggregated-base/", 
    "/get-nli-prediction/"
]

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
@st.cache_data(ttl=60)
def get_logs():
    data = []
    with open('logs/data_logs.log', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    data = pd.DataFrame(data)
    data["date"] = pd.to_datetime(data["time"]).dt.date
    data = data[data["path"].isin(ALLOWED_PATH)]
    return pd.DataFrame(data)

# reassigning rows
row0_spacer1, row0_1, row0_spacer2, row0_2 = st.columns(
    (1, 3, 0.5, 3)
)

# plot request per day:
with row0_1:
    data = get_logs()
    data = data[data["status_code"] == 200]

    df_agg = data.groupby(["path", "date"])["status_code"].count().reset_index()
    df_agg["date"] = df_agg["date"].apply(str)
    df_agg.columns = ["path", "date", "number_of_requests"]

    a = alt.Chart(df_agg, title="Number of successful requests per request_type").mark_area(opacity=0.4).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_requests:Q", title="Number of requests", stack=None),
        color=alt.Color("path:N", title="request type"),
        tooltip=["path", "date", "number_of_requests"]
    )
    b = alt.Chart(df_agg).mark_line(opacity=0.4, point=True).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_requests:Q", title="Number of requests"),
        color=alt.Color("path:N", title="request type"),
        tooltip=["path", "date", "number_of_requests"]
    )
    st.altair_chart((a + b).configure_range(category={'scheme': 'tableau10'}), use_container_width=True)

with row0_2:
    data = get_logs()

    df_agg = data.groupby(["status_code", "date"])["path"].count().reset_index()
    df_agg.columns = ["status_code", "date", "number_of_sc"]
    df_agg["date"] = df_agg["date"].apply(str)
    df_agg.columns = ["status_code", "date", "number_of_sc"]

    a = alt.Chart(df_agg, title="Number of status_code per date").mark_area(opacity=0.4).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_sc:Q", title="Number of status_code", stack=None),
        color=alt.Color("status_code:N", title="Model name"),
        tooltip=["status_code", "date", "number_of_sc"]
    )
    b = alt.Chart(df_agg).mark_line(opacity=0.4, point=True).encode(
        x=alt.X("monthdate(date):T", title="Date"),
        y=alt.Y("number_of_sc:Q", title="Number of status_code", stack=None),
        color=alt.Color("status_code:N", title="Model name"),
        tooltip=["status_code", "date", "number_of_sc"]
    )
    st.altair_chart((a + b).configure_range(category={'scheme': 'tableau10'}), use_container_width=True)


# reassigning rows
row0_spacer1, row0_1, row0_2 = st.columns(
    (1, 2, 4.5)
)
# plot mean response time:
data = get_logs()

time_metrics_dict = data.groupby("path")["process_time"].mean().to_dict()
for k, v in time_metrics_dict.items():
    with row0_1:
        st.metric(f"{k}   mean response time", round(v, 2))

with row0_2:
    st.write(data.tail(100))
