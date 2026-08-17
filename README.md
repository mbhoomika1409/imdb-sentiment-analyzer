# IMDB Sentiment Analyzer

## Project Overview

The **IMDB Sentiment Analyzer** is a machine learning web application that analyzes movie reviews and predicts whether the review expresses a **positive** or **negative** sentiment.

The project uses **Natural Language Processing (NLP)** to preprocess the review text and **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features. A **Logistic Regression** machine learning model is then used to classify the sentiment.

The trained model is integrated with a **FastAPI** backend and a simple HTML interface, allowing users to enter a movie review and receive the predicted sentiment.

---

## Features

* Analyze movie reviews for sentiment
* Predict **Positive** or **Negative** sentiment
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression classification
* FastAPI backend
* HTML frontend using Jinja2 templates
* Trained model saved using Pickle

---

## Technologies Used

* **Python**
* **FastAPI**
* **Machine Learning**
* **Natural Language Processing (NLP)**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **TF-IDF**
* **Logistic Regression**
* **Jinja2**
* **HTML**
* **Uvicorn**

---

## Project Structure

```text
imdb-sentiment-analyzer/
│
├── .gitignore
├── app.py
├── sentiment_analysis.py
├── README.md
├── IMDB Dataset.csv
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
│
└── templates/
    └── index.html
```

---

## How the Project Works

The system follows these steps:

```text
User enters movie review
        ↓
Text preprocessing
        ↓
TF-IDF Vectorization
        ↓
Trained Logistic Regression Model
        ↓
Sentiment Prediction
        ↓
Positive / Negative
```

### 1. Data Collection

The project uses the **IMDB movie review dataset**, which contains movie reviews along with their sentiment labels.

### 2. Text Preprocessing

The review text is cleaned before being passed to the machine learning model.

Typical preprocessing includes:

* Converting text to lowercase
* Removing unnecessary characters
* Removing unwanted spaces
* Preparing text for vectorization

### 3. TF-IDF Vectorization

TF-IDF converts text into numerical values that can be understood by the machine learning model.

It gives higher importance to words that are useful for distinguishing between different reviews.

The trained vectorizer is stored in:

```text
tfidf_vectorizer.pkl
```

### 4. Logistic Regression

A Logistic Regression model is trained using the TF-IDF features.

The trained model is stored in:

```text
sentiment_model.pkl
```

The model predicts:

```text
1 → Positive
0 → Negative
```

### 5. FastAPI

FastAPI is used to create the backend API.

The application receives a movie review, processes it, sends it through the TF-IDF vectorizer and trained model, and returns the predicted sentiment.

---

## API Endpoint

### Prediction Endpoint

```text
POST /predict
```

The endpoint accepts a movie review and returns the predicted sentiment.

Example input:

```json
{
    "text": "This movie was amazing and I really enjoyed it."
}
```

Example output:

```json
{
    "review": "This movie was amazing and I really enjoyed it.",
    "sentiment": "positive"
}
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mbhoomika1409/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy jinja2
```

---

## Running the Application

Start the FastAPI application using:

```bash
uvicorn app:app --reload
```

The application will start locally.

Open the URL shown in the terminal, usually:

```text
http://127.0.0.1:8000
```

---

## Example

### Input

```text
The movie was fantastic. The story was interesting and the acting was excellent.
```

### Output

```text
Sentiment: Positive
```

Another example:

### Input

```text
The movie was boring and disappointing.
```

### Output

```text
Sentiment: Negative
```

---

## Machine Learning Model

### Algorithm

**Logistic Regression**

Logistic Regression is a supervised machine learning classification algorithm used to predict the class of an input.

In this project, it is used for binary classification:

```text
Positive
Negative
```

### Feature Extraction

**TF-IDF**

TF-IDF converts text into numerical feature vectors that represent the importance of words in the dataset.

---

## Backend

The backend is developed using **FastAPI**.

FastAPI handles:

* Receiving user requests
* Processing review text
* Calling the trained ML model
* Returning sentiment predictions
* Serving the HTML frontend

---

## Files Description

| File                    | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `app.py`                | FastAPI application and prediction API              |
| `sentiment_analysis.py` | Machine learning model training and text processing |
| `index.html`            | Web interface for entering reviews                  |
| `IMDB Dataset.csv`      | IMDB movie review dataset                           |
| `sentiment_model.pkl`   | Trained Logistic Regression model                   |
| `tfidf_vectorizer.pkl`  | Trained TF-IDF vectorizer                           |
| `.gitignore`            | Specifies files that should not be tracked by Git   |
| `README.md`             | Project documentation                               |

---

## Learning Outcomes

Through this project, I learned:

* Basics of Natural Language Processing
* Text preprocessing
* TF-IDF vectorization
* Logistic Regression classification
* Model training and saving
* Loading trained ML models
* Building APIs using FastAPI
* Connecting a machine learning model with a web application
* Using HTML with a Python backend
* Git and GitHub project management

---

## Future Improvements

* Add more sentiment categories
* Improve text preprocessing
* Try advanced NLP models
* Add model performance metrics
* Deploy the application online
* Add confidence scores for predictions
* Improve the frontend UI
* Experiment with deep learning and transformer-based models

---

## Author

**Bhoomika M T**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: `https://github.com/mbhoomika1409`

---

## Project Status

**Completed ✅**

This project was developed as part of my machine learning and AI learning journey.
