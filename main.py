from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import joblib
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

nb_model = joblib.load("naive_bayes_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Analyzer Webapp!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if 'text' not in df.columns:
        return JSONResponse(content={"error": "'text' column not found in the uploaded CSV."}, status_code=400)

    X = tfidf_vectorizer.transform(df['text'])
    predictions = nb_model.predict(X)
    df['prediction'] = predictions

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return JSONResponse(content={"file": output.getvalue()})

@app.post("/plot/")
async def plot(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if 'prediction' not in df.columns or 'actual' not in df.columns:
        return JSONResponse(content={"error": "'prediction' or 'actual' column not found in the uploaded CSV."}, status_code=400)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual'], df['prediction'], color='green')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
