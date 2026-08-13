# Customer Review Intelligence System

A small end-to-end project I built to dig into thousands of Flipkart product reviews and pull out something a business team could actually use — overall sentiment, the biggest recurring complaints, and a one-line summary of what's going on.

I work in AI data annotation day-to-day, so I see raw labeled text constantly. This project was my attempt to go the other direction — take unlabeled, messy review text and turn it into something structured and readable, end to end, without relying on any paid API.

## What it does

- Reads raw Flipkart product review data (~190k rows)
- Cleans the text (strips links, punctuation, lowercases everything)
- Runs sentiment analysis on every review — Positive / Negative / Neutral
- Tags each review against three complaint buckets using keyword matching: delivery delays, packaging issues, customer support problems
- Generates a plain-English summary of the findings
- Displays everything in an interactive Streamlit dashboard — pie chart for sentiment split, bar chart for complaint volume, and a raw-review explorer you can filter by sentiment

## Why I built it this way

Most tutorials for a project like this just call an OpenAI endpoint and call it a day. I didn't want to depend on a paid key, so everything here runs locally:

- **Sentiment** — NLTK's VADER. It's a lexicon-based scorer, so no training or GPU needed, and it holds up reasonably well on short, informal review text.
- **Complaints** — a keyword-dictionary approach. Not glamorous, but it's fast, fully explainable, and easy to extend if a new complaint category shows up in the data.
- **Summary** — the dashboard uses a template that fills in the actual computed numbers (so it's always accurate, never hallucinated). In the notebook I also experimented with `distilbart-cnn-12-6` from Hugging Face to generate a more natural-sounding abstractive summary from a sample of reviews — that part's slower, so it stayed in the notebook rather than the live dashboard.

## Tech stack

`pandas` · `nltk` (VADER) · `scikit-learn` · `transformers` (DistilBART, experimental) · `Streamlit` · `Plotly`

## Project structure

```
├── analysis.ipynb          # data cleaning, sentiment, complaint tagging, summary experiments
├── dashboard.py             # Streamlit app
├── data/
│   ├── flipkart_product.csv     # raw source data
│   └── processed_reviews.csv    # cleaned + labeled output from the notebook
└── requirements.txt
```

## Running it locally

```bash
git clone <this-repo>
cd Customer_review_intelligence_system
pip install -r requirements.txt
```

1. Open `analysis.ipynb` and run it top to bottom — this cleans the raw data, runs sentiment + complaint tagging, and writes `data/processed_reviews.csv`.
2. Then launch the dashboard:
```bash
streamlit run dashboard.py
```

## What I'd add next

- Swap the keyword-based complaint tagger for a proper topic model (LDA or BERTopic) so it can surface complaint categories I haven't manually defined
- Add a date column to the source data so the dashboard can show sentiment trending over time, not just a snapshot
- Try a fine-tuned sentiment model instead of VADER and compare accuracy on a hand-labeled sample

## Dataset

Flipkart product reviews dataset (public, sourced from Kaggle).
