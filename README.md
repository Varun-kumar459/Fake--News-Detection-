# AI Fake News Detection System

Paste a news article or social-media post and a machine-learning model predicts **Fake** or **Real**.

**Tech:** Python · Pandas · Scikit-learn · NLP (NLTK) · Flask

## Project structure

```
fake-news-detector/
├── app.py                  # Flask web app + JSON API
├── train.py                # trains, evaluates, saves the model
├── predict_cli.py          # command-line predictions
├── generate_sample_data.py # builds a small demo dataset
├── config.py               # paths and settings
├── requirements.txt
├── src/
│   ├── data_loader.py      # Pandas: load + clean dataset
│   ├── preprocess.py       # NLP: lowercase, stopwords, stemming
│   ├── features.py         # style features (caps, !, ?, length)
│   ├── model.py            # TF-IDF + Logistic Regression pipeline
│   └── predictor.py        # load model, predict, explain words
├── templates/index.html    # web page
├── static/style.css
├── data/                   # datasets go here
└── models/                 # trained model is saved here
```

## How it works

1. **Load data** (Pandas): merges `Fake.csv` (label 1) and `True.csv` (label 0), or uses the sample CSV.
2. **Clean text** (NLP): lowercase, remove URLs/mentions/punctuation, drop stopwords, stem words.
3. **Features:** TF-IDF unigrams + bigrams, plus style signals (exclamation marks, ALL-CAPS words, text length).
4. **Model:** Logistic Regression in a scikit-learn `Pipeline`, evaluated with accuracy, F1 and a confusion matrix.
5. **Serve:** Flask loads the saved pipeline and returns a verdict, confidence and the words that drove it.

## Setup and run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Get data (pick one)
python generate_sample_data.py   # quick demo data
#   OR download the Kaggle "Fake and Real News Dataset" (ISOT)
#   and place Fake.csv and True.csv inside data/

# 2. Train
python train.py

# 3. Run the web app
python app.py                    # open http://127.0.0.1:5000
```

CLI: `python predict_cli.py "Breaking!!! Doctors hate this miracle cure"`

API:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The central bank published a report showing a 2.3 percent rise in growth."}'
```

## Important notes

- The bundled sample data is synthetic and very easy, so it scores near 100%. Use the real ISOT dataset for meaningful results.
- ISOT articles have source-specific patterns (e.g. "WASHINGTON (Reuters) -"), so the model may partly learn publisher style rather than truth. Test on other sources before trusting it.
- The model detects writing style, not facts. It cannot verify claims.

## Ideas to extend

- Compare models (Naive Bayes, Passive Aggressive, Random Forest, SVM) with cross-validation.
- Add a fact-check lookup API or source-credibility feature.
- Replace TF-IDF with transformer embeddings (BERT).
- Store checked texts in SQLite and show history.
