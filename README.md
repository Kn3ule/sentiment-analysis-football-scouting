# sentiment analysis football scouting

This repository provides the code for a masterthesis with the titel "Integrating Sentiment Analysis into Football Scouting: A Data Driven Approach".
It provides an application for the automated analysis of social media comments of football fans. The application can be run both locally and in a Docker environment. Below you will find a detailed explanation of how the project is structured, which data can be processed, and how to use the application.

---

## 1. Overview & Usage

### Purpose of the Application

This application performs **sentiment analysis** on comments from social media platforms (e.g., Instagram). It provides a **dashboard** for displaying the analyzed data. Concretely, you can:

- **Upload JSON files** containing comments (e.g., from a sports club’s Instagram posts) to analyse with GPT.
<img width="1726" alt="Bildschirmfoto 2024-12-31 um 15 09 14" src="https://github.com/user-attachments/assets/0db984eb-9de8-4d53-b4ab-940e498a8631" />

- **Perform sentiment analysis** to determine whether comments are positive, neutral, or negative
- **Upload CSV files** that have already been processed for direct display in the dashboard.
<img width="1726" alt="Bildschirmfoto 2024-12-31 um 15 11 26" src="https://github.com/user-attachments/assets/db8d10f9-881a-45f0-8e01-31990f688639" />

- **Visualize** data via statistics, histograms, word clouds, and sentiment-specific comment listings.
<img width="1726" alt="Bildschirmfoto 2024-12-31 um 15 11 14" src="https://github.com/user-attachments/assets/73e2b4fb-e11d-4742-a4d3-92b255bbe032" />


### JSON File Format

To process JSON files, the following format is expected (example based on Instagram data):

```json
[
  {
    "link": "https://www.instagram.com/p/xyz123/",
    "caption": "Example post description",
    "comments": [
      "Awesome match!",
      "What a goal!",
      "That lineup was terrible..."
    ]
  },
  ...
]
```

- **`link`**: A reference URL (optional).
- **`caption`**: The post description or caption (e.g., from Instagram).
- **`comments`**: A list of strings representing comments on this particular post.

### Additional Requirements

1. **.env File with OpenAI Key**  
   If you plan to use GPT-based models (e.g., GPT 4o-mini or others), you need to have your OpenAI key in a `.env` file. Create a `.env` file in the project directory (or in the appropriate location) with the following content:

   ```bash
   OPEN_API_KEY=sk-1234abcd...
   ```

   The application reads this key to call GPT functionalities. Make sure not to accidentally commit this `.env` file or make it publicly available.

2. **Cookies for Web Scraping**  
   If you plan to do web scraping (e.g., for Instagram), you must have a `cookies` folder containing a file named `insta-cookies.json`. This file holds **up-to-date session cookies** for Instagram and is used in the Jupyter notebooks (e.g., `Data_Collection/scraping.ipynb`).

   **Directory structure** example:

   ```
   cookies/
   └── insta-cookies.json
   ```

   Make sure your cookies remain valid; otherwise, the scraping process may fail.

### Quick Usage

1. **Local Usage**  
   - Clone the repository
   - Install the required Python packages (see [Local Deployment](#31-local-deployment))
   - Create the `.env` file (OpenAI Key) and, if necessary, provide cookies (for scraping)
   - Run `app.py` in the `src` directory (e.g., via `streamlit run app.py`)
   - Open the browser and navigate to the dashboard URL. From there, upload JSON/CSV files and start the analysis.

2. **Docker Usage**  
   - Build the Docker image (see [Docker Deployment](#32-docker-deployment))
   - Make sure to include the `.env` file (for the OpenAI key) when running the container
   - Run the container, then open the corresponding URL in your browser to use the dashboard.

---

## 2. Directory Structure

```
.
├── data
│   ├── instagram
│   │   └── club_instacomments.json        <-- JSON data from instagram for specific clubs
│       └── ...
│   ├── labeled
│   │   └── labeled_data.csv               <-- Labeled data for model evaluation
│       └── ...
│   └── processed
│       ├── club.csv                       <-- Example of processed CSV with sentiment analysis
│       └── ...
├── notebooks
│   ├── Data_Collection.ipynb              <-- Jupyter notebook for the example of potential web scraping
│   └── models.ipynb                       <-- Jupyter notebook for the model evaluation to perform sentiment analysis
├── src
│   ├── app.py                             <-- Main Streamlit dashboard
│   ├── preprocessing.py                   <-- Preprocessing, JSON processing, etc.
│   └── gpt.py                             <-- GPT-related functions/logic
├── cookies
│   └── insta-cookies.json                <-- Session cookies for Instagram scraping
├── requirements.txt
├── Dockerfile
└── README.md                              <-- This README file
```

- **`data/instagram/`**: Contains social media data (e.g., from club pages) in JSON format.  
- **`data/labeled/`**: Contains labeled data (CSV) for evaluating model accuracy.  
- **`data/processed/`**: Contains CSV files that were generated by the application after analysis (e.g., for individual players or clubs).  
- **`notebooks/`**: Various Jupyter notebooks:
  - **`Data_Collection.ipynb`**: Web scraping notebooks (collecting data from Instagram, etc.).
  - **`models.ipynb`**: Evaluation of different sentiment analysis models (Vader, XLM-RoBerta, GPT 4o-mini).
- **`src/`**: Houses the code for the **dashboard application**.
  - **`app.py`**: Entry point for the Streamlit dashboard.
  - **`preprocessing.py`**: Functions that handle data processing (e.g., `preprocess_json`).
  - **`gpt.py`**: Logic and functions for GPT models (or GPT APIs).
- **`cookies/insta-cookies.json`**: The file containing current Instagram session cookies needed for scraping.

---

## 3. Deployment

### 3.1 Local Deployment

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Kn3ule/Text-Mining-Sentiment-Analysis-for-player-scouting.git
   cd Text-Mining-Sentiment-Analysis-for-player-scouting
   ```

2. **Install Python Libraries**  
   - Ensure you have Python 3.12 installed.
   - Use a virtual environment to avoid conflicts, then install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Add .env File (OpenAI Key)**  
   Create a `.env` file in the project directory:
   ```bash
   OPEN_API_KEY=sk-1234abcd...
   ```
   *(You can skip this if GPT functionality is not required.)*

4. **Cookies**  
   For web scraping, create a `cookies` folder with a file named `insta-cookies.json` containing valid Instagram session cookies.

5. **Run the Application**  
   ```bash
   cd src
   streamlit run app.py
   ```
   - This opens a browser window (or provides a URL in the terminal).
   - Upload JSON/CSV files in the web interface and start the analysis.

---

### 3.2 Docker Deployment

1. **Build the Docker Image**  
   If a `Dockerfile` is present, go to the root directory and run:
   ```bash
   docker build -t sentiment-app .
   ```
   *(`sentiment-app` is a freely chosen name for the image.)*

2. **Use .env**  
   To ensure your `.env` (with OPEN_API_KEY, etc.) is recognized inside the container, specify `--env-file` when you run the container:
   ```bash
   docker run -p 8501:8501 --env-file .env sentiment-app
   ```

3. **Cookies for Scraping**  
   If you want to do scraping inside the container, you may mount the `cookies` folder as a volume:
   ```bash
   docker run -p 8501:8501      --env-file .env      -v /path/to/local/cookies:/app/cookies      sentiment-app
   ```

4. **Access the Dashboard**  
   - Open your browser at `http://localhost:8501`.
   - You can then upload JSON/CSV files and use the application just as locally.

---

## 4. Model Comparison: Evaluating Sentiment Analysis Models

To determine the most suitable sentiment analysis model for this project, three different models were evaluated:

1. **Vader**: A lexicon-based sentiment analysis tool.
2. **XLM-Roberta**: A transformer-based model for multilingual text classification.
3. **GPT 4o-mini**: A lightweight version of OpenAI’s GPT model.

The models were compared using key metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices**.

### Metrics and Results

Below are the evaluation results for each model on a dataset of 200 labeled comments.

### **1. Vader**

- **Accuracy**: 0.38  

**Classification Report**:
```
              precision    recall  f1-score   support

    POSITIVE       0.87      0.34      0.48       143
     NEUTRAL       0.18      0.64      0.28        28
    NEGATIVE       0.23      0.34      0.28        29

    accuracy                           0.38       200
   macro avg       0.43      0.44      0.35       200
weighted avg       0.68      0.38      0.43       200
```

**Confusion Matrix**:

![c6e77655-840a-4670-bfb7-5d6ad45cadd8](https://github.com/user-attachments/assets/93fbacca-1326-4621-bf72-8bc4d6d05c17)

**Analysis**:
- Vader performed poorly, especially for the **NEUTRAL** and **NEGATIVE** classes, where recall was below 0.35.
- The model heavily favored predicting the **POSITIVE** class, leading to imbalanced results.
- While its precision for **POSITIVE** comments is high, its low recall indicates it often failed to identify all positive comments correctly.

---

### **2. XLM-Roberta**

- **Accuracy**: 0.65  

**Classification Report**:
```
              precision    recall  f1-score   support

    POSITIVE       0.86      0.70      0.77       143
     NEUTRAL       0.29      0.43      0.35        28
    NEGATIVE       0.42      0.62      0.50        29

    accuracy                           0.65       200
   macro avg       0.52      0.58      0.54       200
weighted avg       0.72      0.65      0.67       200
```

**Confusion Matrix**:

![10e2bc4e-b972-4564-a8fd-dbcf09c20feb](https://github.com/user-attachments/assets/228d19e0-a599-4368-a270-ddbab30a4e30)

**Analysis**:
- XLM-Roberta significantly outperformed Vader, achieving an accuracy of 65%.
- The model demonstrated a better balance across all classes, particularly with higher recall for the **NEGATIVE** class.
- However, its performance for the **NEUTRAL** class remains weak, with low precision and recall.

---

### **3. GPT 4o-mini**

- **Accuracy**: 0.82  

**Classification Report**:
```
              precision    recall  f1-score   support

    POSITIVE       0.94      0.86      0.90       143
     NEUTRAL       0.49      0.64      0.55        28
    NEGATIVE       0.72      0.79      0.75        29

    accuracy                           0.82       200
   macro avg       0.71      0.77      0.74       200
weighted avg       0.84      0.82      0.83       200
```

**Confusion Matrix**:

![2651fc0f-a08f-47ba-8f3d-204942579627](https://github.com/user-attachments/assets/33603255-12e1-4008-a2e2-a6d82ec6db7b)

**Analysis**:
- GPT 4o-mini achieved the highest accuracy (82%) and demonstrated robust performance across all classes.
- It showed excellent precision and recall for the **POSITIVE** and **NEGATIVE** classes.
- While performance for the **NEUTRAL** class remains weaker, it is notably better than the other models.

---

## Key Observations:

1. **Overall Performance**:
   - GPT 4o-mini is the best-performing model, achieving the highest accuracy and balanced performance across all classes.
   - XLM-Roberta offers a middle ground but struggles with the **NEUTRAL** class.
   - Vader is unsuitable for this task, as its lexicon-based approach heavily biases predictions toward the **POSITIVE** class.

2. **Confusion Matrices**:
   - The confusion matrices reveal that GPT 4o-mini is significantly better at minimizing false negatives and false positives compared to the other models.
   - XLM-Roberta shows potential but requires further fine-tuning.

3. **Practical Implications**:
   - For the sentiment analysis task in this application, **GPT 4o-mini** is used due to its high accuracy and consistency.
