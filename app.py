import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from data_functions import *
from paths import *
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from bs4 import BeautifulSoup




### OPISANIE NAZW
st.set_page_config(page_title="Analiza tapicerni", layout="wide")
st.title("📊 Analiza czasu pracy tapicerów")
st.sidebar.header("Filtry")

st.markdown("Proszę o chwilę cierpliwości 😊.  \n"
            "Po włączeniu strony dane mogą ładować się przez kilka minut ⏳.  \n")
loading_placeholder = st.empty()
dots = ["", ".", "..", "..."]
for dot in dots:
    loading_placeholder.text(f"Ładowanie{dot}")
    time.sleep(3)


### SPRAWDZENIE CZY DANE SA ZALADOWANE, BY PONOWNIE NIE PRZELICZAC STRONY PO UZYCIU PRZYCISKU
if 'bazowe_dane' not in st.session_state:
    st.session_state.bazowe_dane = load_data(path_raw_data, "czas tapicernia")
if 'obrobione_dane' not in st.session_state:
    st.session_state.obrobione_dane = update_data(st.session_state.bazowe_dane)
# st.write("Dane w session_state.bazowe_dane:", st.session_state.bazowe_dane)
# st.write("Dane w session_state.obrobione_dane:", st.session_state.obrobione_dane)
df_tapicernia_czasy = st.session_state.obrobione_dane
df_bazowe = st.session_state.bazowe_dane

### OPIS ANALIZY
st.markdown("""
### Opis analizy

Dane z systemu **Saturn** otrzymałem od **Damiana**.

Dane zostały posortowane dwustopniowo:
1. Numer tapicera  
2. Data rozpoczęcia tapicerowania

Następnie w **komisje** połączono bryły, dla których różnica czasu rozpoczęcia tapicerowania była mniejsza niż 2 minuty.

Do analizy zostały wzięte tylko te komisje, które spełniały wszystkie z poniższych warunków:
- Tapicerowanie rozpoczęło się i zakończyło tego samego dnia  
- Nie obejmowało modeli wycofanych (np. *Extreme*)  
- Nie obejmowało brył nietypowych (np. *SOFA NIETYPOWA*)  
- W tym samym czasie nie były tapicerowane różne komisje (z tolerancją do 3 minut)
""")



### ETAP I - OCZYSZCZENIE DANYCH
st.subheader("ETAP 1 - OCZYSZCZENIE DANYCH")
if st.button("Pokaz szczegółowo proces oczyszczania danych"):
    st.markdown("Tabela z **surowymi danymi** z systemu **Saturn**, które otrzymałem od **Damiana**.:")
    st.write(df_bazowe)
    analiza_tapicerzy(df_tapicernia_czasy)

    st.write("Tabela z **danymi po obróbce**.")
    st.write(df_tapicernia_czasy)
    
### ETAP II - GRUPOWANIE DANYCH
st.subheader("ETAP 2 - GRUPOWANIE DANYCH")

# WCZYTANIE ZGRUPOWANYCH DANYCH
df_final = create_grouped_df(df_tapicernia_czasy)
loading_placeholder.text("Sukces! Udało się pomyślnie załadować dane.")
# OKREŚLENIE Z KIEDY JEST ANALIZA
first_date = df_tapicernia_czasy['Start'].min()
last_date  = df_tapicernia_czasy['Stop'].max()
st.write(f"Zakres danych: {first_date.date()} do {last_date.date()}")
if st.button("Pokaz szczegółowo proces filtrowania danych"):
    df_final = create_grouped_df(df_tapicernia_czasy, czy_komentarz="tak")
    st.markdown("Tabela ze **zgrupowanymi i przefiltrowanymi** danymi:")
    st.write(df_final)



### USTAWIENIE FILTRÓW
# TAPICEROWIE
tapicer_filtr = st.sidebar.multiselect(
    'Tapicerzy:',
    options=sorted(df_tapicernia_czasy['Nazwisko'].unique()),
    default=['T01', 'T02', 'T10']
)

# DATY ANALIZY
start_date = dt.date(2022, 12, 1)
end_date = dt.date(2025, 12, 31)
selected_dates = st.sidebar.slider(
    "Zakres dat",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="YYYY-MM-DD"
)

# EFEKTYWNOŚĆ
min_efektywnosc = 100
max_efektywnosc = 200
selected_efektywnosc = st.sidebar.slider(
    "Zakres efektywności:",
    min_value=0,
    max_value=300,
    value=(min_efektywnosc, max_efektywnosc)
)




### ZASTOSTOWANIE FILTRÓW
st.subheader("ŚREDNIA EFEKTYWNOŚĆ (%)")
st.markdown("""
Po lewej stronie możesz skorzystać z dostępnych filtrów.
Po ich zmianie dane w tabeli poniżej zostaną automatycznie odświeżone
""")
mask_tapicerzy = df_final['nazwisko'].isin(tapicer_filtr)
total = df_final[mask_tapicerzy].shape[0]
#st.write("total ", total)
if total == 0:
    below_range = 0
    above_range = 0
else:
    below_range = df_final[(df_final["efektywnosc"]*100 < selected_efektywnosc[0]) & mask_tapicerzy].shape[0] / total
    above_range = df_final[(df_final["efektywnosc"]*100 > selected_efektywnosc[1]) & mask_tapicerzy].shape[0] / total

#st.write(f"Ponizej wybranej efektywnosci jest {round(below_range * 100, 1)}% komisji.")
#st.write(f"Powyzej wybranej efektywnosci jest {round(above_range * 100, 1)}% komisji.")


# ZASTOSOWANIE FILTRÓW
filtered_df_final = df_final[(df_final["efektywnosc"]*100>=selected_efektywnosc[0]) & (df_final["efektywnosc"]*100<= 
selected_efektywnosc[1])]
filtered_df_final['model'] = filtered_df_final['model'].apply(lambda x: sorted(x) if isinstance(x, (list, np.ndarray)) else x)
filtered_df_final["model"] = filtered_df_final["model"].apply(lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x)
filtered_df_final["nazwisko"] = filtered_df_final["nazwisko"].apply(lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x)
filtered_df_final = filtered_df_final[filtered_df_final["nazwisko"].isin(tapicer_filtr)]

# PIVOT ILOŚCI PO ZASTOSOWANIU FILTRÓW
pivot_count = filtered_df_final.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='count'
)
pivot_count[pivot_count < 10] = np.nan


# PIVOT Z ŚREDNIEJ PO ZASTOSOWANIU FILTRÓW
pivot_mean = filtered_df_final.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='mean'
)
pivot_mean[pd.isna(pivot_count)] = np.nan
pivot_mean = pivot_mean * 100
pivot_mean = pivot_mean.round(0)
pivot_count = pivot_count.dropna(how='all')
pivot_count = pivot_count.dropna(axis=1, how='all')
pivot_mean = pivot_mean.dropna(how='all')
pivot_mean = pivot_mean.dropna(axis=1, how='all')
# st.write("Pivot_count")
# st.write(pivot_count)
# st.write("Pivot_mean")
# st.write(pivot_mean)

# ZASTOSOWANIE FILTRU NA DF_FINAL, BY WYDOBYC WARTOSCI Z ZASTOSOWANYM PRZEDZIALEM EFEKTYWNOSCI Z FILTRU, BY POKAZAC ILE WARTOSCI LACZNIE JEST DLA WYBRANYCH PAR
valid_pairs = pivot_count[~pd.isna(pivot_count)].stack().index.tolist()
mask = df_final.apply(
    lambda row: (
        ', '.join(sorted(row['model'])) if isinstance(row['model'], (list, np.ndarray)) else row['model'],
        ', '.join(sorted(row['nazwisko'])) if isinstance(row['nazwisko'], (list, np.ndarray)) else row['nazwisko']
    ) in valid_pairs,
    axis=1
)
df_valid = df_final[mask]
df_valid['model'] = df_valid['model'].apply(lambda x: sorted(x) if isinstance(x, (list, np.ndarray)) else x)
df_valid["model"] = df_valid["model"].apply(lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x)
df_valid["nazwisko"] = df_valid["nazwisko"].apply(lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x)
pivot_df_valid_all = df_valid.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='count'
)
# st.write("Pivot_df_valid_all")
# st.dataframe(pivot_df_valid_all)

df_valid_ponizej_efektywnosc = df_valid[df_valid['efektywnosc']*100 < selected_efektywnosc[0]]
pivot_df_valid_ponizej_efektywnosc = df_valid_ponizej_efektywnosc.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='count'
)
# st.write("Pivot_df_valid_ponizej_efektywnosc")
# st.dataframe(pivot_df_valid_ponizej_efektywnosc)

df_valid_powyzej_efektywnosc = df_valid[df_valid['efektywnosc']*100 > selected_efektywnosc[1]]
pivot_df_valid_powyzej_efektywnosc = df_valid_powyzej_efektywnosc.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='count'
)
# st.write("Pivot_df_valid_powyzej_efektywnosc")
# st.dataframe(pivot_df_valid_powyzej_efektywnosc)


sum_ponizej_efektywnosc = pivot_df_valid_ponizej_efektywnosc.sum(axis=1, skipna=True)
sum_all_count = pivot_count.sum(axis=1, skipna=True)
pivot_mean['ponizej_efektywnosc'] = (sum_ponizej_efektywnosc / sum_all_count) * 100
pivot_mean['ponizej_efektywnosc'] = pivot_mean['ponizej_efektywnosc'].fillna(0).round(0)
# st.write("Pivot_mean with ponizej_efektywnosc")
# st.dataframe(pivot_mean)

sum_powyzej_efektywnosc = pivot_df_valid_powyzej_efektywnosc.sum(axis=1, skipna=True)
sum_all_count = pivot_count.sum(axis=1, skipna=True)
pivot_mean['powyzej_efektywnosc'] = (sum_powyzej_efektywnosc / sum_all_count) * 100
pivot_mean['powyzej_efektywnosc'] = pivot_mean['powyzej_efektywnosc'].fillna(0).round(0)
# st.write("Pivot_mean with powyzej_efektywnosc")
# st.dataframe(pivot_mean)



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
column_width = '120px'

pivot_mean = pivot_mean.rename(columns={
    'ponizej_efektywnosc': 'poniżej efektywność (%)',
    'powyzej_efektywnosc': 'powyżej efektywność (%)'
})
columns_less_more = ['poniżej efektywność (%)', 'powyżej efektywność (%)']
pivot_mean.columns.name = None
pivot_mean.index.name = None
tooltip_df = pivot_mean.copy()
for row in tooltip_df.index:
    for col in tooltip_df.columns:
        before = pivot_mean.at[row, col] if (row in pivot_mean.index and col in pivot_mean.columns) else None
        after = pivot_mean.at[row, col] if (row in pivot_mean.index and col in pivot_mean.columns) else None

        if pd.isna(after):
            tooltip = "Brak danych"
        else:
            tooltip = f"Przed filtrem: {before:.0f}\nPo filtrze: {after:.0f}" if not pd.isna(before) else f"Po filtrze: {after:.0f}"

        tooltip_df.at[row, col] = tooltip
styled_pivot_mean = pivot_mean.style \
    .applymap(lambda val: highlight_above(val, 20), subset=columns_less_more) \
    .applymap(lambda val: highlight_above(val, 155), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(lambda val: highlight_below(val, 120), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(highlight_nan) \
    .format("{:.0f}") \
    .set_tooltips(tooltip_df) \
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('white-space', 'normal'),
            ('word-wrap', 'break-word'),
            ('text-align', 'center'),
            ('width', '180px'),
            ('background-color', '#e6f2ff')  # Jasnoniebieskie tło dla nagłówków
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('width', '180px')
        ]},
        {'selector': 'td:first-child', 'props': [
            ('background-color', '#e6f2ff')  # Jasnoniebieskie tło dla pierwszej kolumny
        ]}
    ])
st.markdown(styled_pivot_mean.to_html(), unsafe_allow_html=True)


pivot_df_valid_all_median = df_valid.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='median'
)
pivot_df_valid_all_median = pivot_df_valid_all_median * 100
pivot_df_valid_all_median = pivot_df_valid_all_median.round(0)

styled_pivot_median = pivot_df_valid_all_median.style \
    .applymap(lambda val: highlight_above(val, 155), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(lambda val: highlight_below(val, 120), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(highlight_nan) \
    .format("{:.0f}") \
    .set_table_styles([
        {'selector': 'th', 'props': [('white-space', 'normal'), ('word-wrap', 'break-word'), ('text-align', 'center'), ('width', '180px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('width', '180px')]}
    ])
#st.markdown(styled_pivot_median.to_html(), unsafe_allow_html=True)

st.subheader("Analiza metodą najmniejszych kwadratów")
df_mnk = df_final

df_mnk["nazwisko"] = df_final["nazwisko"].apply(
    lambda x: ', '.join(sorted(x)) if isinstance(x, (list, np.ndarray)) else str(x)
)

df_mnk["model"] = df_final["model"].apply(
    lambda x: ' '.join(sorted(x)) if isinstance(x, (list, np.ndarray)) else str(x)
)

df_mnk["komisja_srednik"] = df_mnk["komisja"].apply(
    lambda x: '; '.join(sorted(x)) if isinstance(x, (list, np.ndarray)) else str(x)
)

tapicerzy = df_final["nazwisko"].unique()
modele = df_final["model"].unique()

tapicer = st.selectbox("Wybierz tapicera", sorted(tapicerzy))
model = st.selectbox("Wybierz model mebla", sorted(modele))

# Filtrowanie
df_mnk = df_final[(df_final["nazwisko"] == tapicer) & (df_final["model"] == model)].copy()
df_mnk = df_mnk[(df_mnk["efektywnosc"]*100>=selected_efektywnosc[0]) & (df_mnk["efektywnosc"]*100<= 
selected_efektywnosc[1])]

if df_mnk.empty:
    st.warning("Brak danych dla wybranego tapicera i modelu.")
    st.stop()



# Macierz brył (0/1)
bryly_unikalne = sorted(set(";".join(df_mnk["komisja_srednik"]).split(";")))
# st.write("bryly unikalne", bryly_unikalne)
bryly_unikalne = [b.strip() for b in bryly_unikalne]
bryly_unikalne = list(set([b.strip() for b in bryly_unikalne]))
# st.write("bryly unikalne po strip", bryly_unikalne)
for b in bryly_unikalne:
    df_mnk[b] = df_mnk["komisja"].apply(lambda x: x.count(b) if isinstance(x, list) else 0)

df_mnk_cleaned = df_mnk.copy()

def make_hashable(x):
    if isinstance(x, (list, np.ndarray)):
        return tuple(x)
    return x

for col in df_mnk_cleaned.columns:
    df_mnk_cleaned[col] = df_mnk_cleaned[col].apply(make_hashable)

# lista duplikatów
duplicates = [x for x in bryly_unikalne if bryly_unikalne.count(x) > 1]
duplicates = list(set(duplicates))


# Dane do regresji
X = df_mnk[bryly_unikalne]
y = df_mnk["czas_poprawiony"]

# Regresja MNK
model_ols = sm.OLS(y, X).fit()

# st.write(model_ols)

def ols_table_to_df(model_ols):
    html = model_ols.summary().tables[1].as_html()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(cols)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.rename(columns={df.columns[0]: "Zmienna"}, inplace=True)
    df = df.sort_values(by="Zmienna")
    return df

y_pred = model_ols.predict(X)

# Połącz wszystko w jeden DataFrame do podglądu
#df_check = X.copy()
#df_check['y_actual'] = y
#df_check['y_predicted'] = y_pred

# Wyświetl kilka pierwszych wierszy
#st.write(df_check.head(10))

# Wyniki
st.subheader("Wyniki regresji MNK")
df_ols = ols_table_to_df(model_ols)
st.dataframe(df_ols)

# R-kwadrat i podsumowanie
st.markdown(f"**R²:** {model_ols.rsquared:.3f}")
st.markdown(f"**Liczba komisji:** {len(df_mnk)}")

# Wykres: rzeczywisty vs. przewidziany
df_mnk["czas_przewidziany"] = model_ols.predict(X)

fig, ax = plt.subplots()
ax.scatter(df_mnk["czas_poprawiony"], df_mnk["czas_przewidziany"], alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Rzeczywisty czas komisji")
ax.set_ylabel("Przewidziany czas komisji")
ax.set_title("Dopasowanie modelu")
st.pyplot(fig)