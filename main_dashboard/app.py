import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from data_functions import *
from paths import *

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 Analiza czasu pracy tapicerów")
st.sidebar.header("Filtry")

df_tapicernia_czasy = load_data("1Sb83J6VAINiYC5oGDVAxNM7NFiPq5j6r", "czas tapicernia")
df_tapicernia_czasy = update_data(df_tapicernia_czasy)

first_date = df_tapicernia_czasy['Start'].min()
last_date  = df_tapicernia_czasy['Stop'].max()

st.write(f"Analiza z okresu {first_date.date()} do {last_date.date()}")
st.dataframe(df_tapicernia_czasy)

tapicer_filtr = st.sidebar.selectbox('Wybierz tapicera:', sorted(df_tapicernia_czasy['Nazwisko'].unique()))

start_date = dt.date(2022, 12, 1)
end_date = dt.date(2025, 12, 31)

selected_dates = st.sidebar.slider(
    "Wybierz datę",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="YYYY-MM-DD"
)

st.write("Wybrana data:", selected_dates)



filtered_df = df_tapicernia_czasy[df_tapicernia_czasy['Nazwisko'] == tapicer_filtr]
df_filtered = df_tapicernia_czasy[(df_tapicernia_czasy["Start"] >= pd.to_datetime(selected_dates[0])) & 
                                  (df_tapicernia_czasy["Stop"] <= pd.to_datetime(selected_dates[1]))]
st.dataframe(filtered_df)
st.dataframe(df_filtered)

df_final = create_grouped_df(df_tapicernia_czasy)
st.write("przed tym powinna byc analiza")
st.dataframe(df_final)