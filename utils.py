import streamlit as st
import io
import pandas as pd

def check_password():
    correct_password = st.secrets["access"]["password"]

    def password_entered():
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Wpisz hasło:", type="password", on_change=password_entered, key="password")
        st.stop()   # Zatrzymaj dalsze działanie
    if not st.session_state.get("password_correct", False):
        st.text_input("Wpisz hasło:", type="password", on_change=password_entered, key="password")
        st.error("Niepoprawne hasło, spróbuj ponownie.")
        st.stop()   # Zatrzymaj dalsze działanie

def to_excel_first(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Dane')
    return output.getvalue()

def clean_pivot_df(df):
    df = df.dropna(how='all')       # usuń wiersze z samymi NaN
    df = df.dropna(axis=1, how='all')  # usuń kolumny z samymi NaN
    return df

def highlight_nan(val):
    if pd.isna(val):
        return 'background-color: lightgray'
    return ''
def highlight_above(val, number):
    if isinstance(val, (int, float)) and val > number:
        return 'background-color: yellow'
    return ''
def highlight_below(val, number):
    if isinstance(val, (int, float)) and val < number:
        return 'background-color: yellow'
    return ''
