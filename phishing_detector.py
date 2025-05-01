import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Feature extraction function
def extract_features(url):
    features = {
        'url_length': len(url),
        'has_https': int('https' in url),
        'has_at_symbol': int('@' in url),
        'has_dash': int('-' in url),
        'has_double_slash': int(url.count('//') > 1),
        'has_ip': int(bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url))),
        'count_dots': url.count('.'),
        'has_login': int('login' in url.lower()),
        'has_secure': int('secure' in url.lower()),
        'has_bank': int('bank' in url.lower())
    }
    return pd.Series(features)

# Load dataset
df = pd.read_csv('url.csv')
df = df.dropna()
df_features = df['url'].apply(extract_features)
df_final = pd.concat([df_features, df['label'].map({'legitimate': 0, 'phishing': 1})], axis=1)

X = df_final.drop('label', axis=1)
y = df_final['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'phishing_model.pkl')
