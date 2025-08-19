import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io
import utils
from data_functions import *
# from paths import *
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from bs4 import BeautifulSoup


### WYGLAD STRONY
st.set_page_config(page_title="Analiza tapicerni", layout="wide")
st.title("📊 Analiza czasu pracy tapicerów")
st.sidebar.header("Filtry")

### WPISYWANIE HASŁA
utils.check_password()

### ŁADOWANIE
st.markdown("Proszę o chwilę cierpliwości 😊.  \n"
            "Po włączeniu strony dane mogą ładować się przez kilka minut ⏳.  \n")
loading_placeholder = st.empty()
dots = ["", ".", "..", "..."]
for dot in dots:
    loading_placeholder.text(f"Ładowanie{dot}")
    time.sleep(3)


### SPRAWDZENIE CZY DANE SA ZALADOWANE, BY PONOWNIE NIE PRZELICZAC STRONY PO UZYCIU PRZYCISKU
if 'bazowe_dane' not in st.session_state:
    st.session_state.bazowe_dane = load_data(st.secrets["paths"]["path_raw_data"], "czas tapicernia_new")
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
excel_data_obrobione = utils.to_excel_first(df_tapicernia_czasy)
st.download_button(
    label="Pobierz Excel - obrobione dane",
    data=excel_data_obrobione,
    file_name='dane_obrobione.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
    
### ETAP II - GRUPOWANIE DANYCH
st.subheader("ETAP 2 - GRUPOWANIE DANYCH")

# WCZYTANIE ZGRUPOWANYCH DANYCH
df_final = create_grouped_df(df_tapicernia_czasy)

excel_data = utils.to_excel_first(df_final)
st.download_button(
    label="Pobierz Excel - zgrupowane dane",
    data=excel_data,
    file_name='dane.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

loading_placeholder.text("Sukces! Udało się pomyślnie załadować dane.")
# OKREŚLENIE Z KIEDY JEST ANALIZA
first_date = df_tapicernia_czasy['Start'].min()
last_date  = df_tapicernia_czasy['Stop'].max()
st.write(f"Zakres danych: {first_date.date()} do {last_date.date()}")
if st.button("Pokaz szczegółowo proces filtrowania danych"):
    df_final = create_grouped_df(df_tapicernia_czasy, czy_komentarz="tak")
    st.markdown("Tabela ze **zgrupowanymi i przefiltrowanymi** danymi:")
    st.write(df_final)



### FILTRY
# TAPICERZY
tapicer_filtr = st.sidebar.multiselect(
    'Tapicerzy:',
    options=sorted(df_tapicernia_czasy['Nazwisko'].unique()),
    default=['T01', 'T02', 'T10']
)

# DATY ANALIZY
start_date = dt.date(2022, 12, 1)
end_date   = dt.date(2025, 12, 31)
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
⬅️ **Po lewej stronie** możesz skorzystać z dostępnych filtrów.  
Po ich zmianie dane w tabeli poniżej zostaną automatycznie odświeżone.
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

# PIVOT SUMY CZASU PO ZASTOSOWANIU FILTRÓW
pivot_time = filtered_df_final.pivot_table(
    index='model',
    columns='nazwisko',
    values='czas_poprawiony',
    aggfunc='sum'
)
pivot_time[pd.isna(pivot_count)] = np.nan


# PIVOT Z ŚREDNIEJ PO ZASTOSOWANIU FILTRÓW
pivot_mean = filtered_df_final.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='mean'
)
pivot_mean[pd.isna(pivot_count)] = np.nan

pivot_mean = (pivot_mean * 100).round(0)
pivot_count = utils.clean_pivot_df(pivot_count)
pivot_mean = utils.clean_pivot_df(pivot_mean)
pivot_time = utils.clean_pivot_df(pivot_time)


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
    .applymap(lambda val: utils.highlight_above(val, 20), subset=columns_less_more) \
    .applymap(lambda val: utils.highlight_above(val, 155), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(lambda val: utils.highlight_below(val, 120), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(utils.highlight_nan) \
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


# EKSPORTOWANIE PLIKU DO EXCEL
start_selected, end_selected = selected_dates
min_selected_efektywnosc, max_selected_efektywnosc = selected_efektywnosc
filtry = {
    "Data od": f"{start_selected} — {end_selected}",
    "Efektywność": f"{min_selected_efektywnosc}(%) — {max_selected_efektywnosc}(%)",
    "Tapicerzy": ", ".join(tapicer_filtr) if tapicer_filtr else "wszyscy"
}
def to_excel(df, filtry: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Efektywność')
        writer.sheets['Efektywność'] = worksheet

        # Wpisz filtry na górze
        row_offset = 0
        for i, (key, value) in enumerate(filtry.items()):
            worksheet.write(i, 0, f"{key}:")
            worksheet.write(i, 1, str(value))
        row_offset = len(filtry) + 2  # zostaw trochę odstępu

        # Zapisz tabelę z przesunięciem wiersza
        df.to_excel(writer, sheet_name='Efektywność', startrow=row_offset, index=True)

        # Automatyczne ustawienie szerokości
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
            worksheet.set_column(i + 1, i + 1, column_width)
        worksheet.set_column(0, 0, 30)

    output.seek(0)
    return output
excel_file_efektywnosc = to_excel(pivot_mean, filtry)
st.download_button(
    label="📥 Pobierz tabelę efektywności jako Excel",
    data=excel_file_efektywnosc,
    file_name="efektywnosc_tapicerow.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

excel_file_liczebnosc = to_excel(pivot_count, filtry)
st.download_button(
    label="📥 Pobierz tabelę liczebności jako Excel",
    data=excel_file_liczebnosc,
    file_name="efektywnosc_tapicerow.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

excel_file_czas = to_excel(pivot_time, filtry)
st.download_button(
    label="📥 Pobierz tabelę czasu jako Excel",
    data=excel_file_czas,
    file_name="efektywnosc_tapicerow.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)




# TABELA Z NISKA EFEKTYWNOSCIA
def make_hashable(x):
    if isinstance(x, (list, np.ndarray)):
        return tuple(x)
    return x
df_niska_efektywnosc = df_final[df_final['efektywnosc'] * 100 < selected_efektywnosc[0]]
df_niska_efektywnosc = df_niska_efektywnosc[df_niska_efektywnosc["nazwisko"].isin(tapicer_filtr)]
df_niska_efektywnosc["nazwisko"] = df_niska_efektywnosc["nazwisko"].apply(make_hashable)
df_niska_efektywnosc["komisja"] = df_niska_efektywnosc["komisja"].apply(make_hashable)
top_pary_efektywnosc = (
    df_niska_efektywnosc.groupby(["nazwisko", "komisja"])
    .size()
    .reset_index(name="liczba")
    .sort_values("liczba", ascending=False)
    .head(10)
)
st.markdown("### NISKA EFEKTYWNOŚĆ - Top 10 najczęstszych par tapicer–komisja")
st.dataframe(top_pary_efektywnosc, use_container_width=True)






st.subheader("MEDIANA EFEKTYWNOŚCI")
pivot_df_valid_all_median = df_valid.pivot_table(
    index='model',
    columns='nazwisko',
    values='efektywnosc',
    aggfunc='median'
)
pivot_df_valid_all_median = pivot_df_valid_all_median * 100
pivot_df_valid_all_median = pivot_df_valid_all_median.round(0)

styled_pivot_median = pivot_df_valid_all_median.style \
    .applymap(lambda val: utils.highlight_above(val, 20), subset=columns_less_more) \
    .applymap(lambda val: utils.highlight_above(val, 155), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(lambda val: utils.highlight_below(val, 120), subset=[col for col in pivot_mean.columns if col not in columns_less_more]) \
    .applymap(utils.highlight_nan) \
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
st.markdown(styled_pivot_median.to_html(), unsafe_allow_html=True)




### METODA NAJMNIEJSZYCH KWADRATOW
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

top_pary = (
    df_mnk.groupby(["nazwisko", "model"])
    .size()
    .reset_index(name="liczba")
    .sort_values("liczba", ascending=False)
    .head(10)
)
st.markdown("### 🔝 Top 10 najczęstszych par tapicer–model")
st.dataframe(top_pary, use_container_width=True)



tapicer = st.selectbox("Wybierz tapicera", sorted(tapicerzy))
model = st.selectbox("Wybierz model mebla", sorted(modele))

# Filtrowanie
df_mnk = df_final[(df_final["nazwisko"] == tapicer) & (df_final["model"] == model)].copy()
df_mnk = df_mnk[(df_mnk["efektywnosc"]*100>=selected_efektywnosc[0]) & (df_mnk["efektywnosc"]*100<= 
selected_efektywnosc[1])]

if df_mnk.empty:
    st.warning("Brak danych dla wybranego tapicera i modelu.")
    st.stop()



# Macierz brył
bryly_unikalne = sorted(set(";".join(df_mnk["komisja_srednik"]).split(";")))
bryly_unikalne = [b.strip() for b in bryly_unikalne]
bryly_unikalne = list(set([b.strip() for b in bryly_unikalne]))
for b in bryly_unikalne:
    df_mnk[b] = df_mnk["komisja"].apply(lambda x: x.count(b) if isinstance(x, list) else 0)

df_mnk_cleaned = df_mnk.copy()



for col in df_mnk_cleaned.columns:
    df_mnk_cleaned[col] = df_mnk_cleaned[col].apply(make_hashable)

# Dane do regresji
X = df_mnk[bryly_unikalne]
y = df_mnk["czas_poprawiony"]

# Regresja MNK
model_ols = sm.OLS(y, X).fit()

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


### ANALIZA METODA RIDGE
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Standaryzacja zmiennych
X_std = (X - X.mean()) / X.std()

# Regresja Ridge z walidacją krzyżową
model_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
model_ridge.fit(X_std, y)

# Szacowane czasy brył
czas_bryl_ridge = pd.Series(model_ridge.coef_, index=X.columns)
czas_bryl_df = czas_bryl_ridge.to_frame("Czas rzeczywisty")
czas_bryl_df.index.name = "Bryła"

st.subheader("📏 Szacowane czasy tapicerowania (Ridge)")
st.dataframe(czas_bryl_df)

# Przewidywane wartości
y_pred = model_ridge.predict(X_std)
df_mnk["czas_przewidziany"] = y_pred

# Błędy oszacowania
# Wyliczamy przybliżone standard errors (uwaga: tylko aproksymacja!)
resid = y - y_pred
sigma2 = np.var(resid, ddof=len(X.columns))
XtX_inv = np.linalg.inv(np.dot(X_std.T, X_std))
standard_errors = np.sqrt(np.diag(sigma2 * XtX_inv))
errors_df = pd.DataFrame({
    "Czas": model_ridge.coef_,
    "Błąd std": standard_errors
}, index=X.columns)

st.subheader("📉 Współczynniki Ridge i błędy standardowe")
st.dataframe(errors_df)

# Tabela porównawcza rzeczywisty vs przewidziany
df_porownanie = df_mnk[["czas_poprawiony", "czas_przewidziany"]].copy()
df_porownanie["Błąd bezwzględny"] = abs(df_porownanie["czas_poprawiony"] - df_porownanie["czas_przewidziany"])

st.subheader("🔍 Porównanie: rzeczywisty vs przewidziany czas komisji")
st.dataframe(df_porownanie.head(20))  # można ograniczyć do top N

# Wykres dopasowania
fig, ax = plt.subplots()
ax.scatter(df_mnk["czas_poprawiony"], df_mnk["czas_przewidziany"], alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Idealne dopasowanie")
ax.set_xlabel("Rzeczywisty czas")
ax.set_ylabel("Przewidziany czas")
ax.set_title("Dopasowanie modelu Ridge")
ax.legend()
st.pyplot(fig)

# Statystyki modelu
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.subheader("📊 Statystyki dopasowania Ridge")
st.markdown(f"- **MAE (średni błąd bezwzględny):** {mae:.2f}")
st.markdown(f"- **RMSE (pierwiastek błędu średniokwadratowego):** {rmse:.2f}")
st.markdown(f"- **R² (współczynnik determinacji):** {r2:.3f}")
st.markdown(f"- **Alpha wybrane przez CV:** {model_ridge.alpha_}")
