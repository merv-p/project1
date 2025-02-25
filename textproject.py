import streamlit as st
import pandas as pd

# Import the transformers pipeline
try:
    from transformers import pipeline
except Exception as e:
    st.error(
        "Error importing the transformers package. "
        "Please ensure you have installed a compatible version (e.g., run 'pip install --upgrade transformers tensorflow')."
    )
    raise e

# Revised import for VADER: use the all-lowercase package name.
try:
    from vadersentiment.vadersentiment import SentimentIntensityAnalyzer
except Exception as e:
    st.error(
        "Error importing vadersentiment. Please install it using 'pip install vadersentiment'."
    )
    raise e

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
        threshold = st.slider(
            "Score Threshold for Topic Detection", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        
        if st.button("Analyze Comments"):
            with st.spinner("Analyzing..."):
                # Use the TensorFlow backend to avoid torch-related issues.
                classifier = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli", 
                    framework="tf"
                )
                analyzer = SentimentIntensityAnalyzer()
                
                topics_detected = []
                sentiments = []
                
                # Process each row in the selected text column.
                for text in df[text_column]:
                    if isinstance(text, str) and text.strip():
                        result = classifier(text, candidate_list, multi_label=True)
                        selected_topics = [
                            label for label, score in zip(result["labels"], result["scores"]) 
                            if score >= threshold
                        ]
                        topics_detected.append(", ".join(selected_topics))
                        
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
                        topics_detected.append("")
                        sentiments.append("")
                
                df["Detected Topics"] = topics_detected
                df["Sentiment"] = sentiments
                
                st.subheader("Analysis Results")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV", 
                    data=csv,
                    file_name="ai_topic_sentiment_results.csv", 
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error processing file: {e}")
