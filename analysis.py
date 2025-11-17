import os
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load all CSVs from folder
def load_dataset(folder):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

df = load_dataset("dataset_1000")

# Convert epoch to datetime
df['date'] = pd.to_datetime(df['epoch'], unit='s')

### 1. Sentiment Analysis ###
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = sia.polarity_scores(str(text))
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['clean_text'].apply(get_vader_sentiment)

# BERT-based Sentiment Classifier
sentiment_pipeline = pipeline("sentiment-analysis")
df['bert_sentiment'] = df['clean_text'].apply(lambda x: sentiment_pipeline(str(x))[0]['label'])

### 2. Hashtag & Co-occurrence Analysis ###
from itertools import combinations
from collections import Counter

# Extract hashtags
df['hashtags'] = df['hashtags'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])

# Create co-occurrence network
tag_pairs = [tuple(sorted(pair)) for tags in df['hashtags'] if len(tags) > 1 for pair in combinations(tags, 2)]
co_occurrence = Counter(tag_pairs)

# Create network graph
G = nx.Graph()
for (tag1, tag2), weight in co_occurrence.items():
    G.add_edge(tag1, tag2, weight=weight)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)
plt.title("Hashtag Co-occurrence Network")
plt.show()

### 3. Engagement Metrics Analysis ###
engagement_cols = ['replyCount', 'retweetCount', 'likeCount', 'quoteCount']
for col in engagement_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['total_engagement'] = df[engagement_cols].sum(axis=1)

# Engagement distribution by sentiment
plt.figure(figsize=(10, 5))
sns.boxplot(x='vader_sentiment', y='total_engagement', data=df)
plt.yscale('log')
plt.title("Engagement Distribution by Sentiment")
plt.show()

### 4. Ideological Group Analysis ###
# Extract mentioned users and retweeted users
df['mentionedUsers'] = df['mentionedUsers'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
mentioned_users = [user['screen_name'] for users in df['mentionedUsers'] for user in users]
user_counts = pd.Series(mentioned_users).value_counts()

# Top influencers
print("Top 10 mentioned users:")
print(user_counts.head(10))

### 5. Temporal Sentiment & Engagement Analysis ###
plt.figure(figsize=(12, 6))
df.groupby(df['date'].dt.date)['vader_sentiment'].value_counts().unstack().plot(kind='line', marker='o')
plt.title("Sentiment Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend(title="Sentiment")
plt.show()

# Engagement trends over time
plt.figure(figsize=(12, 6))
df.groupby(df['date'].dt.date)['total_engagement'].sum().plot(kind='line', marker='o')
plt.title("Total Engagement Over Time")
plt.xlabel("Date")
plt.ylabel("Total Engagement")
plt.show()

print("Analysis Complete!")
