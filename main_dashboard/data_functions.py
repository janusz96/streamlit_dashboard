import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import re
from datetime import date
from datetime import time as datetime_time
from ..paths import *

czas_bez_pracy_minuty = 945
przerwy_dict = {
    (date(2023, 1, 5), date(2023, 1, 9)): ('po trzech kroli 2023', 3),
    (date(2023, 4, 7), date(2023, 4, 11)): ('po wielkanocy 2023', 3),
    (date(2023, 6, 7), date(2023, 6, 9)): ('po bozym ciele 2023', 1),
    (date(2023, 7, 4), date(2023, 7, 6)): ('po sobocie 04.07.2023', 1),
    (date(2023, 7, 15), date(2023, 7, 31)): ('po wakacjach 2023', 15),
    (date(2023, 8, 14), date(2023, 8, 16)): ('po 15 sierpnia 2023', 1),
    (date(2023, 3, 9), date(2024, 3, 11)): ('po sobocie 09.03.2024', 1),
    (date(2024, 3, 29), date(2024, 4, 2)): ('po wielkanocy 2024', 3),
    (date(2024, 4, 20), date(2024, 4, 22)): ('po sobocie 20.04.2024', 1),
    (date(2024, 4, 26), date(2024, 5, 6)): ('po majowce 2024', 9),
    (date(2024, 5, 11), date(2024, 5, 13)): ('po sobocie 11.05.2024', 1),
    (date(2024, 5, 25), date(2024, 5, 27)): ('po sobocie 25.05.2024', 1),
    ### CZY 5 maja (poniedzialek) byl niepracujacy?
    (date(2024, 8, 14), date(2024, 8, 19)): ('po 15 sierpnia 2024', 4),
    (date(2024, 10, 31), date(2024, 11, 4)): ('po 1 listopada 2024', 3),
    (date(2024, 12, 20), date(2025, 1, 7)): ('po swietach 2024', 17)
}

def load_data(path, sheet):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        return df
    except FileNotFoundError:
        st.error(f"Nie znaleziono pliku: {path}")
        return None

def update_data(df):
    df = remove_useless_columns(df)
    df = sort_data_by_name(df)
    df = add_model_bryla_column(df)
    df = add_material_info(df, path_komisja_skory)
    df = add_data_columns(df)
    df = add_when_finished(df)
    df = add_pricelist(df)
    df = add_breaks(df)
    df = modify_time(df)
    df = add_id_komisji(df)
    return df

def create_grouped_df(df):
    df = load_grouped_data(df)
    analiza_komisji(df)
    return df

def remove_useless_columns(df):
    # USUWANIE ZBĘDNYCH KOLUMN
    columns_to_remove = ["Grupy akord. kod", "Grupy akord. opis", "Imie",
                         "Wydzial", "Czynnosc", "Przerwa"]
    df_cleaned = df.drop(columns = columns_to_remove)
    return df_cleaned

def sort_data_by_name(df):
    df = df.sort_values(by=['Nazwisko', 'Start'])
    df = df.reset_index(drop=True)
    st.write("Sortowanie zakończone.")
    return df

def add_material_info(df, path):
    df_material = load_data(path_komisja_skory, 'komisje')
    df_poprawne_bryly = load_data(path_poprawne_bryly, 'model_bryla')

    def add_komisja(NS):
        if NS in df_material['Nr sys.'].values:
            return df_material[df_material['Nr sys.'] 
                        == NS]['Zebrana nazwa'].values[0]
        else:
            return ''
    df['zebrana_nazwa'] = df['NS'].apply(add_komisja)
    df['material'] = df['zebrana_nazwa'].apply(lambda x: 'SKÓRA' if 'ROMA' in x else 'TKANINA')
    df['model_bryla'] = df['model'] + ' ' + df['material'] + ' ' + df['bryla_zmodyfikowana']
    
    def modify_model_bryla(model_bryla):
        if model_bryla in df_poprawne_bryly['org_model_bryla'].values:
            return df_poprawne_bryly.loc[df_poprawne_bryly['org_model_bryla'] == model_bryla, 'poprawna_model_bryla'].iloc[0]
        return model_bryla
    df['model_bryla_zmodyfikowane'] = df['model_bryla'].apply(modify_model_bryla)

    return df

def add_data_columns(df):
    df_filtered = df.copy()
    
    df_filtered['rok']              = df['Start'].dt.year.astype(str)
    df_filtered['miesiąc']          = df['Start'].dt.month
    df_filtered['godzina_startu']   = df['Start'].dt.strftime('%H:%M:%S')
    df_filtered['godzina_stopu']    = df['Stop'].dt.strftime('%H:%M:%S')
    df_filtered['data_startu']      = df['Start'].dt.date
    df_filtered['data_stopu']       = df['Stop'].dt.date
    df_filtered['dzien_tyg_startu'] = df['Start'].dt.day_name()
    df_filtered['dzien_tyg_stopu']  = df['Stop'].dt.day_name()
    
    return df_filtered

def add_model_bryla_column(df):
    analizowane_modele = ['AMALFI', 'AVANT', 'CALYPSO', 'COCO', 'CUPIDO', 'DIVA A', 'DIVA B',
         'DUO II', 'ELIXIR', 'EXTREME I', 'EXTREME II', 'GOYA', 'GREY I', 'HUDSON', 'HORIZON A',
         'KELLY', 'LENOX', 'LOBBY', 'MAXWELL', 'MISTRAL', 'MYSTIC', 'ONYX', 'OVAL', 'OXYGEN',
         'RAY', 'REVERSO', 'RITZ', 'RONDO', 'SAMOA', 'SPECTRA', 'STONE', 'TOBAGO', 'TOPAZ', 'UNO',
         'WILLOW']
    modele_do_usuniecia = ['EXTREME I', 'EXTREME II', 'MYSTIC','RONDO']
    
    def extract_model(artykul_nazwa):
        for model in analizowane_modele:
            if model in artykul_nazwa:
                return model
        return 'brak_modelu'
    def add_bryla(artykul_nazwa):
        pattern = r'\b(?:' + '|'.join(map(re.escape, analizowane_modele)) + r')\b'
        return re.sub(pattern, '', artykul_nazwa).strip()
    def modify_bryla(bryla):
        if bryla.startswith('PD') or bryla.startswith('PO'):
            return "poduszka"
        if bryla and bryla[0] != '[' and bryla[-1] == ']':
            bryla = '[' + bryla[:-1]
        return bryla
    
    st.write(f"Modele wyłączone z analizy to {modele_do_usuniecia}")
    st.write(f"Liczba wierszy przed usunięciem modeli to {len(df)}")

    df['model'] = df['Artykul nazwa'].apply(extract_model)
    df = df[~df['model'].isin(modele_do_usuniecia)]

    st.write(f"Liczba wierszy po usunięciu modeli to {len(df)}")

    rozkład_modele = df.groupby('model').size().reset_index(name='Ilość')
    rozkład_modele = rozkład_modele.sort_values(by='Ilość', ascending=False)
    st.write("Rozkład ilościowy modeli:")
    st.dataframe(rozkład_modele)
    st.write("Najczęciej występujące artykul_nazwa w brak_modelu:")
    st.write (df[df['model'] == 'brak_modelu']['Artykul nazwa'].value_counts().head(10))

    df['bryla'] = df['Artykul nazwa'].apply(add_bryla)
    df['bryla_zmodyfikowana'] = df['bryla'].apply(modify_bryla)
    df.loc[df['bryla_zmodyfikowana'] == 'poduszka', 'model'] = 'poduszka'
    st.write("powinny sie zmienic modele")
    st.write(df)
    return df

def add_when_finished(df):
    def assign_completion_time(row):
        for condition, func in completion_conditions.items():
            if func(row):
                return condition
        for (start, end), (przerwa_opis, _) in przerwy_dict.items():
            if row['data_startu'] == start and row['data_stopu'] == end:
                return przerwa_opis
        return row.get('kiedy_ukonczono', 'NA')

    completion_conditions = {
        'mniej niż 3 minuty': lambda r: r['Czas'] < 3,
        'tego samego dnia': lambda r: r['data_startu'] == r['data_stopu'],
        'po weekendzie': lambda r: r['dzien_tyg_startu'] == 'Friday' and r['dzien_tyg_stopu'] == 'Monday' and (r['data_stopu'] - r['data_startu']).days == 3,
        'nastepnego dnia': lambda r: (r['data_stopu'] - r['data_startu']).days == 1
    }

    df['kiedy_ukonczono'] = df.apply(assign_completion_time, axis=1)

    rozkład_when_finished = df.groupby('kiedy_ukonczono').size().reset_index(name='Ilość')
    rozkład_when_finished = rozkład_when_finished.sort_values(by='Ilość', ascending=False)
    suma = rozkład_when_finished['Ilość'].sum()
    rozkład_when_finished['Udział (%)'] = ((rozkład_when_finished['Ilość'] / suma) * 100).round(0).astype(int)
    st.write("Rozkład ilościowy kiedy_ukonczono:")
    st.dataframe(rozkład_when_finished)

    st.write(f"Liczba wierszy przed usunięciem modeli to {len(df)}")
    df = df[df['kiedy_ukonczono'].isin(['mniej niż 3 minuty', 'tego samego dnia', 'po weekendzie', 'nastepnego dnia'])]

    st.write(f"Liczba wierszy po usunięciem modeli to {len(df)}")
    return df

def add_pricelist(df):
    df_cennik = load_data(path_cennik, "cennik_nowy")
    df_cennik = df_cennik [["model_material_bryla", "cennik"]]
    df_cennik = df_cennik.dropna (how="all")
    cennik_dict = df_cennik.set_index('model_material_bryla')['cennik'].to_dict()    
    df['czas_cennikowy'] = df['model_bryla_zmodyfikowane'].map(cennik_dict).fillna(0)
    df.loc[df['bryla_zmodyfikowana'] == 'poduszka', 'czas_cennikowy'] = 1

    st.write ("Wartości do których brak ceny w cenniku:")
    st.write(df[df['czas_cennikowy'] == 0]['Artykul nazwa'].unique())

    st.write ("Modele poduszek:")
    st.write(df[df['czas_cennikowy'] == 1]['Artykul nazwa'].unique())
    return df

def add_breaks(df):
    # PIERWSZA PRZERWA
    same_day_mask   = df['kiedy_ukonczono'] == 'tego samego dnia'
    next_day_values = ['nastepnego dnia', 'po weekendzie'] + [value[0] for value in przerwy_dict.values()]
    next_day_mask   = df['kiedy_ukonczono'].isin(next_day_values)
    df['pierwsza_przerwa'] = 0 
    df['pierwsza_przerwa_drugi_dzien'] = 0
    df.loc[same_day_mask, 'pierwsza_przerwa'] = (df.loc[same_day_mask,'Start'].dt.time < datetime_time(10, 0, 0)) & (df.loc[same_day_mask, 'Stop'].dt.time > datetime_time(10, 15, 0))
    df.loc[next_day_mask, 'pierwsza_przerwa'] = (df.loc[next_day_mask, 'Start'].dt.time < datetime_time(10, 0, 0))
    df.loc[next_day_mask, 'pierwsza_przerwa_drugi_dzien'] = (df.loc[next_day_mask, 'Stop'].dt.time > datetime_time(10, 15, 0))     

    # DRUGA PRZERWA
    df['druga_przerwa'] = 0 
    df['druga_przerwa_drugi_dzien'] = 0
    df.loc[same_day_mask, 'druga_przerwa'] = (df.loc[same_day_mask, 'Start'].dt.time < datetime_time(13, 0, 0)) & (df.loc[same_day_mask, 'Stop'].dt.time > datetime_time(13, 15, 0))
    df.loc[next_day_mask, 'druga_przerwa'] = (df.loc[next_day_mask, 'Start'].dt.time < datetime_time(13, 0, 0))   
    df.loc[next_day_mask, 'druga_przerwa_drugi_dzien'] = (df.loc[next_day_mask, 'Stop'].dt.time > datetime_time(13, 15, 0))

    columns_to_convert = ['pierwsza_przerwa', 'pierwsza_przerwa_drugi_dzien', 'druga_przerwa', 'druga_przerwa_drugi_dzien']
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    df['rozpoczeto_przed_7:15'] = (df['Start'].dt.time < datetime_time(7, 15, 0)).astype(int).map({1: 'tak', 0: 'nie'})
    df['zakonczono_po_15:30'] = (df['Stop'].dt.time > datetime_time(15, 30, 0)).astype(int).map({1: 'tak', 0: 'nie'})

    st.write(f"{round((df['rozpoczeto_przed_7:15'] == 'tak').sum()/len(df)*100, 2)}% zostało rozpoczetych przed 7:15")
    st.write(f"{round((df['zakonczono_po_15:30'] == 'tak').sum()/len(df)*100,2)}% zostało zakończonych po 15:30")

    return df

def modify_time(df):
    df['czasy_przerw'] = df[['pierwsza_przerwa', 'pierwsza_przerwa_drugi_dzien', 'druga_przerwa', 'druga_przerwa_drugi_dzien']].sum(axis=1) * 15
    def calculate_correct_time(row):
        czasy_przerw = row['czasy_przerw']
        start_date = row['data_startu']
        end_date = row['data_stopu']
    
        if (start_date, end_date) in przerwy_dict:
            dni_przerwy = przerwy_dict[(start_date, end_date)]
            return row['Czas'] - (czas_bez_pracy_minuty + dni_przerwy * 24 * 60) - czasy_przerw

        if row['kiedy_ukonczono'] == 'nastepnego dnia':
            return row['Czas'] - czas_bez_pracy_minuty - czasy_przerw
        elif row['kiedy_ukonczono'] == 'tego samego dnia':
            return row['Czas'] - czasy_przerw
        elif row['kiedy_ukonczono'] == 'po weekendzie':
            return row['Czas'] - czasy_przerw - czas_bez_pracy_minuty - 2 * 24 * 60
        else:
            return None  
        
    df['czas_poprawiony'] = df.apply(calculate_correct_time, axis=1)
    return df

def add_id_komisji(df):
    id_komisji = 1 
    df = df.reset_index(drop=True)
    df['id_komisji'] = 0
    df['ilosc_bryl_w_komisji'] = 0  
    df['time_diff'] = pd.Timedelta(0)

    time_diff = pd.Timedelta(0)
    time_diff_seconds = 0
    ilosc_bryl_w_komisji = 0

    for index, row in df.iterrows():
        if index > 0:
            time_diff = row['Start']- df.at[index - 1, 'Start']
            df.at[index, 'time_diff'] = time_diff  
            
            time_diff_seconds = time_diff.total_seconds()
            df.at[index, 'time_diff_seconds'] = time_diff_seconds

            if time_diff_seconds > 120 or row['Nazwisko'] != df.at[index - 1, 'Nazwisko']:
                ilosc_bryl_w_komisji = 0
                id_komisji += 1
            
        ilosc_bryl_w_komisji+=1

        df.at[index, 'id_komisji'] = id_komisji
        df.at[index, 'ilosc_bryl_w_komisji'] = ilosc_bryl_w_komisji

    max_values = df.groupby('id_komisji')['ilosc_bryl_w_komisji'].transform('max')
    df['ilosc_bryl_w_komisji'] = max_values

    df['id_komisji'] = df['id_komisji'].astype(int)
    agg_result = df.groupby('id_komisji')['model_bryla_zmodyfikowane'].apply(list)
    sorted_agg_result = agg_result.apply(lambda x: sorted(x, key=str))
    df['komisja'] = df['id_komisji'].map(sorted_agg_result)
    df['komisja'] = df['komisja'].apply(lambda x: '\n'.join(map(str, x)) if all(isinstance(item, str) for item in x) else '')

    return df

def load_grouped_data(df):
    unique_id_komisji = df['id_komisji'].unique()
    df_final = pd.DataFrame({'id_komisji': unique_id_komisji})
    grouped = df.groupby('id_komisji')

    ### SUMA CZASU CENNIKOWEGO
    sum_czas_cennikowy = grouped['czas_cennikowy'].sum().reset_index()
    df_final = pd.merge(df_final, sum_czas_cennikowy, on='id_komisji')

    ### ŚREDNI CZAS SKORYGOWANY
    mean_czas_poprawiony = grouped['czas_poprawiony'].mean().reset_index()
    df_final = pd.merge(df_final, mean_czas_poprawiony, on='id_komisji')

    ### LICZBA BRYŁ W RAMACH KOMISJI
    count_id_komisji = df['id_komisji'].value_counts().reset_index()
    count_id_komisji.columns = ['id_komisji', 'ilosc_bryl_w_komisji']
    df_final = pd.merge(df_final, count_id_komisji, on='id_komisji')

    ### NAZWISKO TAPICERA
    unique_nazwisko = grouped['Nazwisko'].unique().reset_index()
    unique_nazwisko.columns = ['id_komisji', 'nazwisko']
    df_final = pd.merge(df_final, unique_nazwisko, on='id_komisji')

    ### NAZWA WSZYSTKICH TAPICEROWANYCH BRYŁ
    all_komisja = grouped['model_bryla_zmodyfikowane'].agg(lambda x: sorted(list(x))).reset_index()
    all_komisja.columns = ['id_komisji', 'model_bryla_zmodyfikowane']
    df_final = pd.merge(df_final, all_komisja, on='id_komisji')
    df_final['komisja'] = df_final['model_bryla_zmodyfikowane']

    ### ILOŚĆ PODUSZEK W KOMISJI
    sum_poduszek = grouped['model_bryla_zmodyfikowane'].apply(lambda x: sum(1 for elem in x if 'poduszka' in elem)).reset_index()
    sum_poduszek.columns = ['id_komisji', 'suma_poduszek']
    df_final = pd.merge(df_final, sum_poduszek, on='id_komisji')

    ### NAZWA WSZYSTKICH TAPICEROWANYCH MODELI
    unique_model = grouped['model'].unique().reset_index()
    unique_model.columns = ['id_komisji', 'model']
    df_final = pd.merge(df_final, unique_model, on='id_komisji')

    ### KIEDY UKOŃCZONO
    unique_kiedy_ukonczono = grouped['kiedy_ukonczono'].unique().reset_index()
    unique_kiedy_ukonczono.columns = ['id_komisji', 'kiedy_ukonczono']
    df_final = pd.merge(df_final, unique_kiedy_ukonczono, on='id_komisji')

    ### MINIMALNY CZAS STARTU W RAMACH KOMISJI
    min_start = grouped['Start'].min().reset_index()
    min_start.columns = ['id_komisji', 'minimum_start']
    df_final = pd.merge(df_final, min_start, on='id_komisji')

    ### MAXYMALNY CZAS STOPU W RAMACH KOMISJI
    max_stop = grouped['Stop'].max().reset_index()
    max_stop.columns = ['id_komisji', 'maximum_stop']
    df_final = pd.merge(df_final, max_stop, on='id_komisji')

    ### ROZPOCZĘCIE PRZED 7:15
    przed_startem = grouped['rozpoczeto_przed_7:15'].first().reset_index()
    przed_startem.columns = ['id_komisji', 'Rozpoczeto_przed_7:15']
    df_final = pd.merge(df_final, przed_startem, on='id_komisji')

    ### EFEKTYWNOŚĆ
    df_final['efektywnosc'] = df_final ['czas_cennikowy'] / df_final['czas_poprawiony']
    bins_eff = [-float('inf'), 0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, float('inf')]
    labels_eff = ['<0%', '0-50%', '50-75%', '75-100%', '100-125%', 
          '125-150%', '150-200%', ">200%"]
    df_final['efektywnosc_przedzialy'] = pd.cut(df_final['efektywnosc'], 
                                                bins=bins_eff, labels=labels_eff, 
                                                right=False)

    ### ILOŚĆ PRZERW
    ilosc_przerw = (
        grouped['pierwsza_przerwa'].mean() +
        grouped['druga_przerwa'].mean() +
        grouped['pierwsza_przerwa_drugi_dzien'].mean() +
        grouped['druga_przerwa_drugi_dzien'].mean()
    ).reset_index() 
    ilosc_przerw.columns = ['id_komisji', 'ilosc_przerw']
    df_final = pd.merge(df_final, ilosc_przerw, on='id_komisji')
    return df_final

def analiza_tapicerzy(df):
    rozklad_tapicerzy = df.groupby('Nazwisko').size().reset_index(name='Ilość')
    st.write("Rozkład ilościowy tapicerów:")
    st.dataframe(rozklad_tapicerzy)


    pivot_df = df.pivot_table(index=['rok', 'miesiąc'], columns='Nazwisko', aggfunc='size', fill_value=0)

    st.write("Tabela z rokiem, miesiącem i nazwiskami:")
    st.dataframe(pivot_df)

    df['kiedy_ukonczono'] = df['Czas'].apply(lambda x: 'mniej niż 3 minuty' if x < 3 else '')

    grouped = df.groupby(['rok', 'miesiąc', 'Nazwisko']).agg(
        liczba_wierszy=('Czas', 'size'),
        liczba_mniej_niz_3minut=('kiedy_ukonczono', lambda x: (x == 'mniej niż 3 minuty').sum())
        ).reset_index()
    
    grouped['procent'] = grouped.apply(
        lambda row: round(100 - (row['liczba_mniej_niz_3minut'] / row['liczba_wierszy'] * 100))
        if row['liczba_wierszy'] > 20 else np.nan, axis=1
    )  

    pivot_df_2 = grouped.pivot_table(
        index=['rok', 'miesiąc'], columns='Nazwisko', values='procent', aggfunc='first'
    ).fillna(np.nan)

    pivot_df_2 = pivot_df_2.round().astype('Int64')

    def highlight_cells(val):
        if pd.isna(val):
            return 'background-color: lightgray'  # Szary kolor dla pustych komórek
        elif val >= 90:
            return 'background-color: lightgreen'  # Zielony kolor dla wartości > 90
        return 'background-color: lightcoral'  # Brak zmiany tła dla pozostałych komórek
    styled_pivot_df_2 = pivot_df_2.style.applymap(highlight_cells)

    st.write("Tabela z procentem ukończenia w czasie dłuższym niż 3 minuty:", styled_pivot_df_2)

def analiza_komisji(df):
    st.write(f"Ilość wszystkich komisji to: {len(df)}")
    rozklad_efektywnosc_przedzialy = df.groupby('efektywnosc_przedzialy').size().reset_index(name='Ilość')
    rozklad_efektywnosc_przedzialy = rozklad_efektywnosc_przedzialy.sort_values(by='Ilość', ascending=False)
    suma = rozklad_efektywnosc_przedzialy['Ilość'].sum()
    rozklad_efektywnosc_przedzialy['Udział (%)'] = ((rozklad_efektywnosc_przedzialy['Ilość'] / suma) * 100).round(0).astype(int)
    st.write("Rozkład ilościowy efektywnosc_przedzialy:")
    st.dataframe(rozklad_efektywnosc_przedzialy)

    st.write(f"Ilość wszystkich komisji to: {len(df)}")
    df['kiedy_ukonczono'] = df['kiedy_ukonczono'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
    rozklad_kiedy_ukonczono = df.groupby('kiedy_ukonczono').size().reset_index(name='Ilość')
    rozklad_kiedy_ukonczono = rozklad_kiedy_ukonczono.sort_values(by='Ilość', ascending=False)
    suma = rozklad_kiedy_ukonczono['Ilość'].sum()
    rozklad_kiedy_ukonczono['Udział (%)'] = ((rozklad_kiedy_ukonczono['Ilość'] / suma) * 100).round(0).astype(int)
    st.write("Rozkład ilościowy kiedy_ukonczono:")
    st.dataframe(rozklad_kiedy_ukonczono)

    df['kiedy_ukonczono'] = df['kiedy_ukonczono'].apply(lambda x: str(x) if isinstance(x, tuple) else x)
    df = df[df['kiedy_ukonczono']=="('tego samego dnia',)"]
    st.write(f"Ilość wszystkich komisji to: {len(df)}")
    rozklad_efektywnosc_przedzialy = df.groupby('efektywnosc_przedzialy').size().reset_index(name='Ilość')
    rozklad_efektywnosc_przedzialy = rozklad_efektywnosc_przedzialy.sort_values(by='Ilość', ascending=False)
    suma = rozklad_efektywnosc_przedzialy['Ilość'].sum()
    rozklad_efektywnosc_przedzialy['Udział (%)'] = ((rozklad_efektywnosc_przedzialy['Ilość'] / suma) * 100).round(0).astype(int)
    st.write("Rozkład ilościowy efektywnosc_przedzialy:")
    st.dataframe(rozklad_efektywnosc_przedzialy)

    df['model'] = df['model'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
    rozklad_modele = df.groupby('model').size().reset_index(name='Ilość')
    rozklad_modele = rozklad_modele.sort_values(by='Ilość', ascending=False)
    suma = rozklad_modele['Ilość'].sum()
    rozklad_modele['Udział (%)'] = ((rozklad_modele['Ilość'] / suma) * 100).round(0).astype(int)
    st.write("Rozkład ilościowy modeli:")
    st.dataframe(rozklad_modele)

    df['model']    = df['model'].apply(lambda x: str(x[0]) if isinstance(x, np.ndarray) or isinstance(x, list) else str(x))
    df['nazwisko'] = df['nazwisko'].apply(lambda x: str(x[0]) if isinstance(x, np.ndarray) or isinstance(x, list) else str(x))
    st.write(df['nazwisko'].unique())
    model_counts = df['model'].value_counts()

    # Filtruj tylko te, które występują minimum 10 razy
    popular_models = model_counts[model_counts >= 10].index
    tapicerowie_analiza = ['T01', 'T02']
    
    # Odfiltruj dane
    df = df[df['nazwisko'].isin(tapicerowie_analiza)]
    df_filtered = df[df['model'].isin(popular_models)]
    df_filtered = df[(df['efektywnosc'] >= 1) & (df['efektywnosc'] <= 2)]
    pivot_count = pd.pivot_table(
        df_filtered,
        index='model',
        columns='nazwisko',
        values='efektywnosc',
        aggfunc='count'
    )
    pivot_mean = pd.pivot_table(
        df_filtered,
        index='model',
        columns='nazwisko',
        values='efektywnosc',
        aggfunc='mean'
    )
    pivot_filtered = pivot_mean.where(pivot_count >= 10)
    pivot_filtered = pivot_filtered.dropna(how='all')
    pivot_filtered = pivot_filtered.applymap(lambda x: round(x * 100) if pd.notna(x) else x)
    # Oblicz procentowy udział wierszy z efektywnością poniżej 1 oraz powyżej 2 przed filtrowaniem
    below_1_percentage = df[df['efektywnosc'] < 1].groupby('model').size() / df.groupby('model').size() * 100
    above_2_percentage = df[df['efektywnosc'] > 2].groupby('model').size() / df.groupby('model').size() * 100

    # Dodaj te procenty jako kolumny do pivot_filtered
    pivot_filtered['% below 1'] = pivot_filtered.index.map(below_1_percentage)
    pivot_filtered['% above 2'] = pivot_filtered.index.map(above_2_percentage)
    st.write(pivot_filtered)


    # Tworzymy pivot dla mediany
    pivot_count = pd.pivot_table(
        df_filtered,
        index='model',
        columns='nazwisko',
        values='efektywnosc',
        aggfunc='count'
    )
    pivot_median = pd.pivot_table(
        df_filtered,
        index='model',
        columns='nazwisko',
        values='efektywnosc',
        aggfunc='median'  # Używamy mediany
    )
    pivot_filtered = pivot_median.where(pivot_count >= 10)
    pivot_filtered = pivot_filtered.dropna(how='all')
    pivot_filtered = pivot_filtered.applymap(lambda x: round(x * 100) if pd.notna(x) else x)
    st.write(pivot_filtered)



def clean_data(df):




    


    return df_cleaned


    nazwy = ['AMALFI',
         'AVANT',
         'CALYPSO',
         'COCO',
         'CUPIDO',
         'DIVA A',
         'DIVA B',
         'DUO II',
         'ELIXIR',
         'EXTREME I',
         'EXTREME II',
         'GOYA', 
         'GREY I',
         'HUDSON',
         'HORIZON A',
         'KELLY',
         'LENOX',
         'LOBBY',
         'MAXWELL', 
         'MISTRAL',
         'MYSTIC',
         'ONYX',
         'OVAL',
         'OXYGEN',
         'RAY', 
         'REVERSO',
         'RITZ',
         'RONDO',
         'SAMOA',
         'SPECTRA',
         'STONE',
         'TOBAGO',
         'TOPAZ',
         'UNO',
         'WILLOW']
    modele_do_usuniecia = ['EXTREME I',
        'EXTREME II',
        'MYSTIC',
        'RONDO']
    # Liczba minut w ciągu doby bez pracy
    # 15h 45 minut
    czas_bez_pracy_minuty = 945
    path_log_czasy = '/Users/janusz/Desktop/Olta/2024 - analiza tapicerni/czas_tapicernia.xls'
    path_poprawne_bryly = '/Users/janusz/Desktop/Olta/2024 - analiza tapicerni/poprawne_bryly.xlsx'
    path_komisja_skory = '/Users/janusz/Desktop/Olta/2024 - analiza tapicerni/GREY_RITZ_SPECTRA.xls'