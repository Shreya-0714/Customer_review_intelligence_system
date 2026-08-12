import streamlit as st
import pandas as pd
import plotly.express as px
import ast

st.set_page_config(page_title="Customer Review Intelligence", layout="wide")

st.title("📊 Customer Review Intelligence Dashboard")
st.caption("Flipkart Product Reviews — Sentiment, Complaints & AI Summary")

# Load processed data
df = pd.read_csv('data/processed_reviews.csv')

# complaints column was saved as a string like "['Delivery delays']" — convert back to a list
df['complaints'] = df['complaints'].apply(ast.literal_eval)

# ---- Sentiment Section ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Breakdown")
    sentiment_counts = df['sentiment'].value_counts()
    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Main Complaints")
    complaint_flat = df['complaints'].explode().dropna().value_counts()
    fig2 = px.bar(
        x=complaint_flat.values,
        y=complaint_flat.index,
        orientation='h',
        labels={'x': 'Number of Mentions', 'y': ''}
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---- AI Summary Section ----
st.subheader("🤖 AI Summary")

sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
top_complaint = complaint_flat.idxmax()
positive_pct = sentiment_pct.get('Positive', 0)

summary = (
    f"{positive_pct:.0f}% of customers had a positive experience overall, "
    f"but satisfaction is impacted mainly by {top_complaint.lower()}, "
    f"which was the most frequently mentioned issue "
    f"({complaint_flat.max():,} mentions)."
)
st.info(summary)

# ---- Raw Data Explorer ----
with st.expander("🔍 Explore raw reviews"):
    selected_sentiment = st.selectbox("Filter by sentiment", ["All"] + list(df['sentiment'].unique()))
    display_df = df if selected_sentiment == "All" else df[df['sentiment'] == selected_sentiment]
    st.dataframe(display_df[['ProductName', 'Rate', 'review_text', 'sentiment']].head(200))