from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import h2o
from h2o.automl import H2OAutoML

# Inicjalizacja H2O
h2o.init()

# Tytuł aplikacji
st.title("AutoML porównanie: PyCaret vs H2O AutoML")

# Krok 1: Ładowanie danych CSV
st.sidebar.header("1. Wczytaj dane CSV")
uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Załadowane dane:")
    st.write(df)

    # Krok 2: Generowanie raportu (profiling)
    st.sidebar.header("2. Profiling danych")
    if st.sidebar.button("Generuj raport"):
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)

    # Krok 3: Wybór kolumny target
    st.sidebar.header("3. Wybierz target")
    target_column = st.sidebar.selectbox("Wybierz kolumnę target", df.columns)

    if target_column:
        st.write(f"Wybrana kolumna target: {target_column}")

        # Krok 4: Porównanie PyCaret i H2O AutoML
        if st.sidebar.button("Uruchom PyCaret i H2O AutoML"):
            # PyCaret
            st.header("PyCaret - Automatyzacja ML")
            setup(data=df, target=target_column, silent=True, html=False)
            best_model = compare_models()
            st.write(f"Najlepszy model PyCaret: {best_model}")

            # H2O AutoML
            st.header("H2O AutoML - Automatyzacja ML")
            h2o_df = h2o.H2OFrame(df)
            aml = H2OAutoML(max_models=5, seed=1)
            aml.train(y=target_column, training_frame=h2o_df)

            # Wyświetlanie najlepszych modeli H2O AutoML
            lb = aml.leaderboard
            st.write("Tabela wyników H2O AutoML:")
            st.write(lb.head())

# Wyłączanie sesji H2O po zakończeniu
h2o.cluster().shutdown()
