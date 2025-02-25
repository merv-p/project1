import streamlit as st
import pandas as pd

# Attempt to import transformers.pipeline; if missing, show an error.
try:
    from transformers import pipeline
except ImportError as e:
    st.error("The transformers package is not installed. Please run 'pip install transformers'.")
    raise e

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("AI Text Analytics: Multi-Topic Categorization & Sentiment Detection")

# Upload the main file (CSV or Excel) with comments/text.
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# Upload a candidate topics file (TXT, CSV, or XLSX).
candidate_topics_file = st.file_uploader("Upload candidate topics file (TXT, CSV, or XLSX)", type=["txt", "csv", "xlsx"])

# Build candidate topics list from file if provided.
candidate_list = []
if candidate_topics_file:
    try:
        filename = candidate_topics_file.name.lower()
        if filename.endswith('.txt'):
            content = candidate_topics_file.getvalue().decode("utf-8")
            candidate_list = [line.strip() for line in content.splitlines() if line.strip()]
        elif filename.endswith('.csv'):
            df_candidates = pd.read_csv(candidate_topics_file)
            candidate_list = df_candidates.iloc[:, 0].dropna().astype(str).tolist()
        elif filename.endswith('.xlsx'):
            df_candidates = pd.read_excel(candidate_topics_file)
            candidate_list = df_candidates.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        st.error(f"Error reading candidate topics file: {e}")

# If no candidate topics file is provided, allow manual input.
if not candidate_list:
    candidate_topics_input = st.text_area(
        "Or enter a comma-separated list of topics (required)", 
        "Finance, Health, Technology, Sports, Entertainment"
    )
    candidate_list = [topic.strip() for topic in candidate_topics_input.split(",") if topic.strip()]

if not candidate_list:
    st.error("No candidate topics provided. Please upload a candidate topics file or enter topics manually.")

if uploaded_file is not None:
    try:
        # Load the main file.
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Select the text column for analysis.
        text_column = st.selectbox("Select the column containing text", df.columns)
        
        # Let the user choose a threshold for topic detection.
        threshold = st.slider("Score Threshold for Topic Detection", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        
        if st.button("Analyze Comments"):
            with st.spinner("Analyzing..."):
                # Use the TensorFlow backend to avoid torch-related errors.
                classifier = pipeline("zero-shot-classification", 
                                      model="facebook/bart-large-mnli", 
                                      framework="tf")
                analyzer = SentimentIntensityAnalyzer()
                
                topics_detected = []
                sentiments = []
                
                # Process each row in the selected text column.
                for text in df[text_column]:
                    if isinstance(text, str) and text.strip():
                        # Use zero-shot classification (multi_label enabled) to score candidate topics.
                        result = classifier(text, candidate_list, multi_label=True)
                        selected_topics = [
                            label for label, score in zip(result["labels"], result["scores"]) 
                            if score >= threshold
                        ]
                        topics_detected.append(", ".join(selected_topics))
                        
                        # Perform sentiment analysis using VADER.
                        vs = analyzer.polarity_scores(text)
                        compound = vs["compound"]
                        if compound >= 0.05:
                            sentiment = "Positive"
                        elif compound <= -0.05:
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"
                        sentiments.append(sentiment)
                    else:
                        topics


