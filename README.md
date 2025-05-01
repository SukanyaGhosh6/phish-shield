#  Phish-Shield  
*A Machine Learning Approach to Phishing URL Detection*  
by [SukanyaGhosh6](https://github.com/SukanyaGhosh6)

---

##  About the Project

**Phish-Shield** is a machine learning-powered phishing detection system designed to analyze and classify URLs as either **legitimate** or **phishing**. This project was born from my passion for combining **cybersecurity awareness** with **hands-on machine learning skills**, developed entirely in **Python** using **VS Code** on a Windows machine.

Phishing attacks are among the most prevalent forms of social engineering today, often using fake URLs to trick users into revealing sensitive data. This project demonstrates how **simple, interpretable features extracted from URLs** — like the presence of symbols, length, or keyword patterns — can be leveraged to train an intelligent system capable of flagging suspicious links.

---

##  Objectives

- Analyze and extract features from raw URLs
- Classify URLs using supervised machine learning models
- Build a complete project executable in VS Code with zero additional setup
- Promote awareness of how machine learning can be applied to real-world cybersecurity threats

---

##  Technologies Used

- **Python 3.12.4**
- **pandas**, **scikit-learn**, **numpy**, **matplotlib** *(only standard libraries or pip-installable)*
- **VS Code** (Windows environment)
- No external data scraping or API integration — just CSV input and clean ML pipeline.

---

##  Dataset

The dataset used contains labeled URLs: `phishing` or `legitimate`.  
You can use [this dataset from Kaggle](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) or create your own CSV with the following structure:

| url                            | label        |
|-------------------------------|--------------|
| http://example.com/login      | phishing     |
| https://google.com            | legitimate   |

---

## 🔍 Feature Engineering

Each URL is converted into a numerical feature vector based on patterns like:

- Length of the URL
- Use of HTTPS
- Count of `@`, `-`, `.` and `//`
- Presence of IP address
- Suspicious keywords (`login`, `secure`, `bank`, etc.)

---

##  Machine Learning Models Used

- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree

Each model was trained and evaluated for **accuracy**, **precision**, **recall**, and **F1-score**, helping us understand tradeoffs in security systems.

---

##  Sample Code Workflow

```python
# Load and preprocess dataset
df = pd.read_csv('urls.csv')
df['url_length'] = df['url'].apply(lambda x: len(x))
df['has_https'] = df['url'].apply(lambda x: int('https' in x))
...

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Train and evaluate model
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Performance
print(classification_report(y_test, predictions))
```

---

##  Results

- Accuracy ranged from **85% to 96%**, depending on the model.
- Random Forest and SVM yielded the best performance.
- Even lightweight models (like Naive Bayes) were decent with proper features.

---

##  What I Learned

- How machine learning can support cybersecurity
- Importance of feature extraction in text-based data like URLs
- How to build, validate, and improve classification models
- The practical risks posed by phishing attacks and how we can automate detection

---

##  How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/SukanyaGhosh6/phish-shield.git
   cd phish-shield
   ```

2. Install the requirements (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python phishing_detector.py
   ```

4. Replace `urls.csv` with your own dataset if needed.

---

##  References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Phishing Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- Research: *Phishing Detection Using Machine Learning Techniques* – IEEE Papers

---

##  Future Work

- Deploy as a Flask API or web app
- Use deep learning (LSTM) for sequence-based detection
- Add browser extension integration for real-time protection

---

If you found this project interesting, feel free to fork, clone, or contribute!  
**Star this repo if you're passionate about defending the web!**

