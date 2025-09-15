import streamlit as st
import pandas as pd
import json
import io
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import time

from preprocess import *
from gpt import *

import nltk
#nltk.download('stopwords')  # nur beim ersten Mal erforderlich

from nltk.corpus import stopwords

# Stopwörter verschiedener Sprachen laden
english_stopwords = set(stopwords.words('english'))
german_stopwords = set(stopwords.words('german'))
spanish_stopwords = set(stopwords.words('spanish'))
french_stopwords = set(stopwords.words('french'))

# Alle Stopwörter zusammenführen
all_stopwords = english_stopwords.union(german_stopwords).union(spanish_stopwords).union(french_stopwords)


# Seitenkonfiguration
st.set_page_config(layout='wide')

# Initialisierung von session_state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'player_names' not in st.session_state:
    st.session_state.player_names = []
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'club_name' not in st.session_state:
    st.session_state.club_name = ""
if 'comment_label' not in st.session_state:
    st.session_state.comment_label = None
if 'current_player' not in st.session_state:
    st.session_state.current_player = None

def process_data(uploaded_file):
    # Anzeige des Ladekreises
    with st.spinner('Data is processed...'):
        # Spielernamen und Clubnamen aus dem Session State
        players = st.session_state.player_names
        club_name = st.session_state.club_name

        # Aufrufen der preprocess_json-Funktion
        df = preprocess_json(players, uploaded_file, club_name)
        print(df)
        if df.empty:
            st.warning("No comments found for the selected players.")
            return
        
        #Sentiment Analysis mit GPT 4o-mini
        df["sentiment_score"], df["sentiment"], df["explanation"], df["translation"] = zip(*df.apply(lambda row: analyze_sentiment(row["player"], row["comment"]), axis=1))
        print(df)
        

        # Speichern des DataFrames im Session State
        st.session_state.df = df
        st.session_state.data_processed = True

def main():
    if not st.session_state.data_processed:
        st.title("Sentiment Analysis for player scouting")

        # Tabs für die Auswahl der Funktionalität
        tab1, tab2 = st.tabs(["Process JSON", "Upload CSV"])

        with tab1:
            st.header("Process JSON-File")
            
            # Eingabefeld für Clubnamen
            st.text_input("Enter the club name:", key="club_name")

            # Eingabefeld für Spielernamen
            st.text_input(
                "Enter the player names (separated by commas):",
                value="",
                key="player_input_json"
            )

            # Dateiuploader für die JSON-Datei
            st.file_uploader(
                "Select a JSON file with comments:",
                type=['json'],
                key="uploaded_json"
            )

            # Start-Button mit Callback
            def start_analysis_json():
                club_name = st.session_state.club_name.strip()
                player_input = st.session_state.player_input_json.strip()
                uploaded_json = st.session_state.uploaded_json

                if uploaded_json is not None and player_input != "" and club_name != "":
                    # Spielernamen speichern
                    st.session_state.player_names = [name.strip() for name in player_input.split(",")]
                    st.session_state.club_name = club_name
                    
                    # Daten verarbeiten
                    start = time.time()
                    process_data(uploaded_json)
                    end= time.time()
                    processed_time = end - start
                    print(processed_time)
                else:
                    st.warning("Please enter club name, player name and upload a JSON file.")

            st.button("Start analysis", on_click=start_analysis_json, key="start_json")

        with tab2:
            st.header("Upload CSV-File")

            st.write("All available players from the CSV file are used.")
            
            # Dateiuploader für die CSV-Datei
            st.file_uploader(
                "Upload a CSV file to generate the dashboard directly:",
                type=['csv'],
                key="uploaded_csv"
            )

            # Start-Button mit Callback
            def start_analysis_csv():
                uploaded_csv = st.session_state.uploaded_csv

                if uploaded_csv is not None:
                    # CSV-Datei einlesen
                    try:
                        df = pd.read_csv(uploaded_csv)
                        required_columns = ['player', 'comment', 'sentiment_score', 'sentiment', 'explanation']
                        if not all(col in df.columns for col in required_columns):
                            st.error("The CSV file does not contain the required columns.")
                            return
                        # Speichern des DataFrames im Session State
                        st.session_state.df = df
                        # Spielernamen automatisch aus der CSV-Datei extrahieren
                        st.session_state.player_names = df['player'].unique().tolist()
                        st.session_state.data_processed = True
                    except Exception as e:
                        st.error(f"Error reading the CSV file: {e}")
                else:
                    st.warning("Please upload a CSV file.")

            st.button("Start analysis", on_click=start_analysis_csv, key="start_csv")
    else:
        # Dashboard anzeigen
        display_dashboard()


def display_dashboard():
    # Filtern des DataFrames nach ausgewählten Spielern
    df = st.session_state.df
    selected_players = st.session_state.player_names

    df_filtered = df[df['player'].isin(selected_players)]

    if df_filtered.empty:
        st.warning("No data available for the selected players.")
        return
    
    # Sidebar-Navigation
    st.sidebar.title("Navigation")

    # Option zum Zurückkehren zur Hauptansicht (immer sichtbar)
    if st.session_state.comment_label:
        st.sidebar.button("Back to overview", on_click=return_to_overview, key="back_to_overview")

    # Auswahl des Spielers für die Anzeige
    spielername = st.sidebar.selectbox('Select a player:', selected_players)
    st.session_state.current_player = spielername
    df_spieler = df_filtered[df_filtered['player'] == spielername]

    # Option zum Neustart der Analyse
    st.sidebar.button("Start new analysis", on_click=restart_analysis)

    # Download-Button für das gesamte DataFrame
    st.sidebar.header("Download")
    csv_data = st.session_state.df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download complete data",
        data=csv_data,
        file_name='data.csv',
        mime='text/csv'
    )

    # Wenn 'comment_label' gesetzt ist, zeige die Kommentare an
    if st.session_state.comment_label and st.session_state.current_player:
        comments_view(st.session_state.current_player, st.session_state.comment_label)
        return

    # Berechnungen
    durchschnitt_sentiment = df_spieler['sentiment_score'].mean()
    gesamtkommentare = len(df_spieler)
    sentiment_counts = df_spieler['sentiment'].value_counts()

    # Dashboard
    st.title(spielername)

    col1, col2, col3 = st.columns([2, 3, 2])

    # Linke Spalte
    with col1:
        st.header('Sentiment overview')
        # Gauge Diagramm
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=durchschnitt_sentiment,
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [-1, -0.05], 'color': "red"},
                    {'range': [-0.05, 0.05], 'color': "orange"},
                    {'range': [0.05, 1], 'color': "green"},
                ],
            },
            number={'font': {'size': 36}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.header('Distribution of sentiment scores')
        fig_hist = px.histogram(
            df_spieler,
            x='sentiment_score',
            nbins=20,
            color_discrete_sequence=['skyblue']
        )
        fig_hist.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Mittlere Spalte
    with col2:
        st.header('General information')
        st.markdown(f"**Number of comments:** {gesamtkommentare}")

        # Balkendiagramm der Sentiment Labels
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment Label', 'y': 'Count'},
            color=sentiment_counts.index,
            color_discrete_map={'NEGATIVE': 'red', 'NEUTRAL': 'orange', 'POSITIVE': 'green'}
        )
        # Achsen-Ticks auf die gewünschten Labels begrenzen
        fig_bar.update_xaxes(
            tickmode='array',
            tickvals=['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
            ticktext=['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        )
        fig_bar.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.header('Wordcloud')
        player_names = st.session_state.current_player.split(" ")
        first_name = player_names[0]
        last_name = player_names[1]
        print(f'{first_name} {last_name}')
        all_stopwords.update({first_name, last_name, first_name.lower(), last_name.lower()})
        text = " ".join(df_spieler['comment'].dropna().astype(str))
        wordcloud = WordCloud(
            background_color="white",
            width=600,
            height=300,
            max_words=50,
            stopwords=all_stopwords
        ).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_wc)

    # Rechte Spalte
    with col3:
        st.header('Sample comments')

        for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            st.subheader(f"{label.capitalize()} comment")
            beispiel_df = df_spieler[df_spieler['sentiment'] == label]
            if not beispiel_df.empty:

                if label == "POSITIVE":
                    st.write("What if City have palmer and Kdb on the mid, I think halland will get 70+ goals")
                elif label == "NEGATIVE":
                    st.write("You people need to stop blaming Jackson for everything! Do u think it was easy what he did? Do u know how many sitters Palmer himself missed on that match?")
                elif label == "NEUTRAL":
                    st.write("Mancity academy player Cole Palmer")
                else:
                    beispiel = beispiel_df['comment'].sample(1).iloc[0]
                    st.write(beispiel)
                # Button zum Anzeigen aller Kommentare dieses Labels
                st.button(
                    f"Show all {label.capitalize()} comments",
                    key=label,
                    on_click=set_comment_label,
                    args=(label,)
                )
            else:
                st.write(f"No {label} comments available.")

def set_comment_label(label):
    st.session_state.comment_label = label

def comments_view(player_name, label):
    st.title(f"All {label.capitalize()} comments for {player_name}")
    df = st.session_state.df
    df_player = df[(df['player'] == player_name) & (df['sentiment'] == label)]

    if df_player.empty:
        st.write(f"No {label} comments available.")
    else:
        for index, row in df_player.iterrows():
            if "translation" in df_player.columns:
                st.write(f"**Comment:** {row['comment']}")
                st.write(f"**Translation:** {row['translation']}")
                st.write(f"**Sentiment Score:** {row['sentiment_score']}")
                st.write(f"**Explanation:** {row['explanation']}")
                st.write("---")
            else:
                st.write(f"**Comment:** {row['comment']}")
                st.write(f"**Sentiment Score:** {row['sentiment_score']}")
                st.write(f"**Explanation:** {row['explanation']}")
                st.write("---")

    st.button("Back to overview", on_click=return_to_overview, key="back_to_overview_list")

def return_to_overview():
    st.session_state.comment_label = None

def restart_analysis():
    st.session_state.data_processed = False
    st.session_state.player_names = []
    st.session_state.df = pd.DataFrame()
    st.session_state.club_name = ""
    st.session_state.comment_label = None
    st.session_state.current_player = None


# Hauptprogramm
if __name__ == "__main__":
    main()
