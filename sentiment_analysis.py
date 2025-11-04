# Level 3 - Task 3

# Import necessary libraries
import pandas as pd
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('3) Sentiment dataset.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# There is no missing data in this dataset

# Print the column names
print("\nColumn Names in the Dataset:")
print(df.columns)

# Text Preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Join back to sentence
    return ' '.join(stemmed_words)

# Apply preprocessing to the 'text' column
df['cleaned_text'] = df['Text'].apply(preprocess_text)
print("\nPreprocessed Text Preview:")
print(df[['Text', 'cleaned_text']].head())

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
    
# Apply Sentiment analysis
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Show results
print("\nSentiment Analysis Results:")
print(df[['Text', 'sentiment']].head())

# Save the results to a new CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
print("\nSentiment analysis results saved to 'sentiment_analysis_results.csv'.")


# Visualization

# Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment') 
plt.ylabel('Number of Texts')

# Save the sentiment distribution plot
plt.savefig('sentiment_distribution.png')
print("Sentiment distribution plot saved as 'sentiment_distribution.png'.")

plt.show()

# Word Cloud for each Sentiments
for sentiment in ['Positive', 'Negative', 'Neutral']:
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    
    # Save the word cloud image
    plt.savefig(f'wordcloud_{sentiment.lower()}.png')
    print(f'Word cloud for {sentiment} sentiment saved as wordcloud_{sentiment.lower()}.png')
    
    plt.show()
    
# The code above performs sentiment analysis on a dataset of text entries. It preprocesses the text, classifies the sentiment using TextBlob, and visualizes the results with bar charts and word clouds.# The results are saved in a new CSV file and word cloud images are generated for each sentiment category.

