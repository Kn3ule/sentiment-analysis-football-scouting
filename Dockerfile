# Basis-Image: schlanker Python-Container
FROM python:3.12-slim

# Setze Arbeitsverzeichnis im Container
WORKDIR /app

# Updates und Installation von Build-Tools und benötigten Dev-Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Kopiere Requirements und installiere diese
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Laden der NLTK-Stopwords
RUN python -m nltk.downloader stopwords

# Kopiere den gesamten Projektcode ins Arbeitsverzeichnis
COPY . .

# Optional: Falls Ihre Dateien aus src importiert werden:
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Setzen Sie Streamlit-Konfigurationen über ENV-Variablen oder eine separate Config-Datei.
# Beispiel: Den Headless-Modus aktivieren
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501


# Exponieren des Ports (Streamlit Standard: 8501)
EXPOSE 8501

# Startbefehl: Startet die Streamlit-App
CMD ["streamlit", "run", "src/app.py"]
