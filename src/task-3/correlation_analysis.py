import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Tesla stock data
tesla_df = pd.read_csv("../../data/yfinance_data/TSLA_historical_data.csv",
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits'],
                      skiprows=1,
                      parse_dates=['Date'])
tesla_df.set_index('Date', inplace=True)
tesla_df.sort_index(inplace=True)

# Calculate daily returns
tesla_df['Daily_Return'] = tesla_df['Close'].pct_change()
tesla_df.dropna(inplace=True)

# Load sentiment scores
news_df = pd.read_csv("../../outputs/processed_data/sentiment_scores.csv", parse_dates=['Date'])
daily_sentiment = news_df.groupby('Date')['Sentiment'].mean().reset_index()
daily_sentiment.set_index('Date', inplace=True)

# Merge with Tesla returns
combined_df = pd.merge(tesla_df[['Close', 'Daily_Return']],
                       daily_sentiment,
                       left_index=True,
                       right_index=True,
                       how='inner')

# Compute correlation
correlation = combined_df['Daily_Return'].corr(combined_df['Sentiment'])
print(f"\n📊 Pearson Correlation: {correlation:.4f}")

# Plot scatter plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=combined_df, x='Sentiment', y='Daily_Return')
plt.title("Tesla Daily Return vs News Sentiment")
plt.xlabel("News Sentiment Score")
plt.ylabel("Daily Return (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../../outputs/visualizations/return_vs_sentiment.png")
plt.show()

# Optional: Rolling correlation
combined_df['Rolling_Corr'] = combined_df['Daily_Return'].rolling(window=5).corr(combined_df['Sentiment'])

plt.figure(figsize=(12,6))
plt.plot(combined_df.index, combined_df['Rolling_Corr'], label='5-day Rolling Correlation')
plt.axhline(0, color='black', linestyle='--')
plt.title("Rolling Correlation: Sentiment vs Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../outputs/visualizations/rolling_correlation.png")
plt.show()

print("📈 Visualizations saved to outputs/visualizations/")