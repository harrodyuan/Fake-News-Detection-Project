import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Ensure nltk resources are downloaded
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

def run_analysis():
    print("Loading data...")
    df = pd.read_csv("data/fake_news_train.csv")
    
    # Preprocessing
    print("Preprocessing...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # EDA & Unsupervised Learning
    print("Running Unsupervised Learning (LDA)...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(df['clean_text'])
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)
    
    # Visualize Topics (Word Clouds)
    feature_names = vectorizer.get_feature_names_out()
    plt.figure(figsize=(20, 5))
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = [topic[i] for i in top_features_ind]
        
        plt.subplot(1, 5, topic_idx + 1)
        plt.barh(top_features, weights)
        plt.title(f"Topic {topic_idx}")
        plt.gca().invert_yaxis()
        
        # Print topics to file
        with open("topics.txt", "a") as f:
            f.write(f"Topic {topic_idx}: {', '.join(top_features)}\n")
            
    plt.tight_layout()
    plt.savefig('topic_distribution.png')
    plt.close()
    
    # Supervised Learning: Baseline
    print("Running Supervised Learning (Baseline)...")
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    tfidf_sup = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_sup.fit_transform(X_train)
    X_test_tfidf = tfidf_sup.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    
    y_pred_baseline = clf.predict(X_test_tfidf)
    print("Baseline Results:")
    print(classification_report(y_test, y_pred_baseline))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_baseline)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.savefig('baseline_cm.png')
    plt.close()
    
    # Supervised Learning: BERT
    print("Running Supervised Learning (DistilBERT)...")
    # Use a smaller subset for speed if needed, but 30k is okay-ish. 
    # Let's use 2000 samples for training for demonstration/speed purposes in this environment, 
    # but acknowledge it in the report. Or maybe 5000.
    # The user wants me to "finish the project", so I should try to make it good.
    # I'll try 2000 for training to ensure it completes in reasonable time.
    
    train_texts = X_train.tolist()[:2000] 
    train_labels = y_train.tolist()[:2000]
    val_texts = X_test.tolist()[:500]
    val_labels = y_test.tolist()[:500]
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    class FakeNewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    val_dataset = FakeNewsDataset(val_encodings, val_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1, # Keep it low for speed
        per_device_train_batch_size=8, # Small batch for memory
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps"
    )
    
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    # Evaluate
    print("Evaluating BERT...")
    predictions = trainer.predict(val_dataset)
    y_pred_bert = np.argmax(predictions.predictions, axis=-1)
    
    print("BERT Results:")
    print(classification_report(val_labels, y_pred_bert))
    
    # Save BERT CM
    cm_bert = confusion_matrix(val_labels, y_pred_bert)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens')
    plt.title('BERT Confusion Matrix')
    plt.savefig('bert_cm.png')
    plt.close()

if __name__ == "__main__":
    run_analysis()

