import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Range: -1 (negative) to +1 (positive)

# Load news data
news_df = pd.read_csv("../../data/news_data/stock_news.csv", parse_dates=['Date'])

# Apply sentiment analysis
news_df['Sentiment'] = news_df['Headline'].apply(get_sentiment)

# Save processed sentiment data
news_df.to_csv("../../outputs/processed_data/sentiment_scores.csv", index=False)

print("✅ Sentiment scores computed and saved.")