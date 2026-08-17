from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords

app = FastAPI()

# Load saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

class Review(BaseModel):
    text: str

from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
def predict(review: Review):
    cleaned = clean_text(review.text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"review": review.text, "sentiment": sentiment}

from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})