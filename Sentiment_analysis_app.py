# Import necessary libraries
import streamlit as st
from transformers import pipeline
import concurrent.futures
import torch
import random
import time

# Set page layout to wide by default
st.set_page_config(layout="wide")

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the smaller models for faster performance (use GPU if available)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Initialize models
sentiment_analysis = load_sentiment_model()
summarization_model = load_summarization_model()

# Mapping model output labels to human-readable labels
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Custom caching function for better performance
def get_sentiment(text):
    return sentiment_analysis(text)

def get_summary(text):
    return summarization_model(text)

# List of random statistics facts
stats_facts = [
    # Statistical Facts
    "The amount of data generated globally every day is estimated to exceed 2.5 quintillion bytes.",
    "Over 90% of the data in the world today has been generated in just the last two years.",
    "More than 2.5 quintillion bytes of data are created each day, but less than 1% of this data is analyzed.",
    "The number of machine learning jobs has increased by over 200% in the last four years.",
    "By 2025, the global public cloud market is expected to reach $800 billion.",
    "70% of the global population is expected to shop online by 2040.",
    "40% of businesses globally have experienced some form of a data breach.",
    "The AI market is forecast to grow to $190 billion by 2025.",
    "Over 3.8 billion people globally use smartphones, which accounts for more than 45% of the world’s population.",
    "By 2030, it is estimated that over 75 billion IoT devices will be in use.",
    "Cybercrime costs are expected to reach $10.5 trillion annually by 2025.",
    "77% of enterprises have at least one AI or machine learning project underway.",
    "The global healthcare analytics market is expected to reach $60 billion by 2024.",
    "Over 55% of businesses are expected to integrate blockchain technology in some form by 2025.",
    "The quantum computing market is expected to reach $65 billion by 2030.",
    "The number of 5G connections worldwide is projected to reach 1.7 billion by 2025.",
    "Digital payments are projected to reach $12 trillion in 2025.",
    "The AR market is expected to surpass $100 billion by 2024.",
    "There are over 4.7 billion social media users worldwide, which is more than half of the global population.",
    "The global RPA market is expected to grow to $13.4 billion by 2029.",
    
    # Technology Facts
    "The first 'computer bug' was a real moth found inside a Harvard Mark II computer in 1947.",
    "The first email ever sent was by Ray Tomlinson in 1971 and it was 'QWERTYUIOP.'",
    "The first website, created by Tim Berners-Lee, went live on August 6, 1991.",
    "The first smartphone was the IBM Simon, released in 1994, combining mobile and PDA functions.",
    "The Apple App Store launched in 2008, and by 2020, it had over 2 million apps.",
    "Dropbox, launched in 2007, had 200 million users by 2017, demonstrating the rise of cloud storage services.",
    "As of 2024, Japan's Fugaku holds the title of the world’s fastest supercomputer.",
    "The first video game, 'Tennis for Two,' was created in 1958 on an oscilloscope.",
    "Over 250 million blockchain transactions were processed in 2023, highlighting its growing"]


# Streamlit app title and description (Centered)
st.markdown("<h1 style='text-align: center;'>Text Insights: Sentiment & Summary</h1>", unsafe_allow_html=True)

# Brief app introduction and model info
st.markdown("""
    <p style="text-align: center; font-size: 16px;">
        Welcome to <b>Text Insights</b>! Analyze the sentiment and get summaries of your text with ease.
    </p>
    
    <p style="text-align: center; font-size: 14px;">
        <b>Sentiment Analysis:</b> This app uses <b>CardiffNLP's Twitter RoBERTa</b> model to classify text as Positive, Neutral, or Negative, along with a confidence score.
    </p>
    
    <p style="text-align: center; font-size: 14px;">
        <b>Text Summarization:</b> <b>BART</b> by Facebook AI is used to generate concise summaries of your text, retaining key information.
    </p>
""", unsafe_allow_html=True)

# Custom CSS for the button width and footer
st.markdown("""
    <style>
        .wide-button {
            display: block;
            width: 50%;
            margin: 0 auto;
        }
        .footer-text {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;
            padding: 5px 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Create the form for text input and submission
with st.form(key='sentiment-form'):
    user_input = st.text_area('Enter your text below:')
    submit = st.form_submit_button('Analyze Text' , use_container_width=True)

# When the submit button is pressed
if submit:
    st.markdown("---")  # Creates a horizontal divider

    # Check for word count
    word_count = len(user_input.split())
    if word_count > 100:
        warning_placeholder = st.empty()  # Create a placeholder for the warning message
        warning_placeholder.warning("Model might get slower than expected due to large input text (over 100 words).")
        time.sleep(10)  # Wait for 10 seconds before removing the warning
        warning_placeholder.empty()  # Clear the warning after 10 seconds


    # Validate user input
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        # Placeholder for displaying statistics facts while models are running
        fact_placeholder = st.empty()

        # Run sentiment analysis and summarization in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sentiment_future = executor.submit(get_sentiment, user_input)
            summary_future = executor.submit(get_summary, user_input)

            # While the models are running, display random facts
            for _ in range(10):  # Display 10 facts, adjust as needed
                random_fact = random.choice(stats_facts)
                fact_placeholder.markdown(f"Fun Facts:  {random_fact}", unsafe_allow_html=True)
                time.sleep(7)  # Wait for 7 seconds before displaying the next fact

            # Adding timeout for future results
            try:
                sentiment_result = sentiment_future.result(timeout=30)
                summary_result = summary_future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                st.error("The analysis took too long. Please try again later.")
                fact_placeholder.empty()
                st.stop()  # Stop the Streamlit app execution

        # Hide the facts by clearing the placeholder
        fact_placeholder.empty()

        # Sentiment Analysis Results
        sentiment = label_map[sentiment_result[0]["label"]]  # Convert model label to readable text
        sentiment_score = sentiment_result[0]["score"] * 100  # Convert the score to percentage

        # Summarization Results
        if len(user_input.split()) > 50:  # Ensure input is large enough for summarization
            summarized_text = summary_result[0]['summary_text']
        else:
            summarized_text = "The input text is too short for summarization. Please enter a longer text."

        # Create two columns layout for displaying the results
        col1, col2 = st.columns(2)

        # Display Sentiment Results in the first column inside a code block
        with col1:
            st.write("**Sentiment Analysis**")
            sentiment_code = f"Sentiment: {sentiment}\nConfidence Score: {sentiment_score:.2f}%"
            st.code(sentiment_code)

        # Display Summary in the second column with custom styling (matte black background & rounded border)
        with col2:
            st.write("**Summary**")
            summary_html = f"""
            <div style="padding: 20px; border: 2px solid #262730; background-color: #1A1C24; color: white; border-radius: 10px;">
                {summarized_text}
            </div>
            """
            st.markdown(summary_html, unsafe_allow_html=True)  # Use markdown with custom HTML for styling

# Footer Text at bottom-right
st.markdown('<div class="footer-text" style=" border: 2px solid #262730; background-color: #1A1C24; color: white; border-radius: 10px;">Created by Omkar Sutar</div>', unsafe_allow_html=True)
