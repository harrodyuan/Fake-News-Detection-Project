import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Loading data...")
df = pd.read_csv("data/fake_news_train.csv")
df['clean_text'] = df['text'].apply(clean_text)

print("Running LDA...")
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(df['clean_text'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf)

feature_names = vectorizer.get_feature_names_out()
with open("topics.txt", "w") as f:
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        f.write(f"Topic {topic_idx}: {', '.join(top_features)}\n")
print("Topics saved to topics.txt")

