# Text Insights: Sentiment Analysis & Summarization

Welcome to **Text Insights**! This application allows you to analyze the sentiment of a given text and get a concise summary of it. It leverages pre-trained models for sentiment analysis and summarization.

## Features:
- **Sentiment Analysis:** Classify text as Positive, Neutral, or Negative using CardiffNLP's `twitter-roberta-base-sentiment` model.
- **Text Summarization:** Generate a summary of your input text using Facebook's BART model.
- **Real-time Feedback:** Get a confidence score for sentiment analysis and a concise summary for longer text.

## Technologies Used:
- **Streamlit:** Framework for building interactive web apps.
- **HuggingFace Transformers:** Pre-trained models for sentiment analysis (`cardiffnlp/twitter-roberta-base-sentiment`) and text summarization (`facebook/bart-large-cnn`).
- **PyTorch:** Machine learning library for running the models, optimized for GPU if available.
  
## try this app here:
[https://text-insights-sentiment-analysis-summarization.streamlit.app/]
