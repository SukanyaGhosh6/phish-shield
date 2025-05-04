# Phish-Shield: A Machine Learning Approach to Phishing URL Detection

**Author**: [Sukanya Ghosh](https://github.com/SukanyaGhosh6)  
**License**: MIT  
**Environment**: Python 3.12.4, Windows OS  
**IDE**: Visual Studio Code  
**Repository**: [Phish-Shield GitHub](https://github.com/SukanyaGhosh6/phish-shield)

---

##  Project Overview

**Phish-Shield** is a machine learning-based detection tool focused on identifying phishing URLs. This project leverages pattern-based feature engineering combined with classical supervised learning algorithms to build an interpretable and effective classification model. Its goal is twofold: (1) demonstrate the application of ML in cybersecurity and (2) provide a simple, extensible tool for phishing URL analysis.

Phishing attacks remain one of the most common cybersecurity threats, targeting users through deceptive URLs designed to mimic legitimate web addresses. By analyzing structural characteristics of URLs, this project showcases how machine learning can offer a proactive defense mechanism.

---

##  Objectives

- Extract interpretable features from raw URLs
- Classify URLs into phishing or legitimate categories
- Compare multiple ML models based on performance metrics
- Raise awareness of phishing tactics and ML-based countermeasures
- Provide a plug-and-play Python implementation executable with minimal setup

---

##  Technologies and Tools

- **Programming Language**: Python 3.12.4  
- **Libraries**:
  - [`pandas`](https://pandas.pydata.org/)
  - [`numpy`](https://numpy.org/)
  - [`scikit-learn`](https://scikit-learn.org/stable/)
  - [`matplotlib`](https://matplotlib.org/)
- **Environment**: Visual Studio Code on Windows
- **No web scraping** or third-party APIs — relies on CSV datasets only

---

##  Dataset

The model uses a CSV dataset consisting of URLs labeled as either `phishing` or `legitimate`. A sample row structure is as follows:

```

url,label
[http://www.crestonwood.com/router.php,phishing](http://www.crestonwood.com/router.php,phishing)
[https://google.com,legitimate](https://google.com,legitimate)

````

You can use an existing dataset from [Kaggle’s Phishing Website Dataset](https://www.kaggle.com/datasets) or supply a custom one in the same format.

---

##  Feature Engineering

Each URL is transformed into a numerical vector using the following features:

- Length of the URL
- Use of HTTPS protocol
- Frequency of suspicious symbols: `@`, `-`, `.`, `//`
- Presence of an IP address
- Inclusion of common phishing keywords such as `login`, `secure`, `account`, `bank`, etc.

These features are selected based on research into phishing detection patterns ([IEEE Reference](https://ieeexplore.ieee.org/document/8755885)).

---

##  Machine Learning Models

The following supervised models are implemented and benchmarked:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree Classifier**

All models are evaluated based on:

- Accuracy
- Precision
- Recall
- F1 Score

###  Sample Workflow

```python
# Load and preprocess data
df = pd.read_csv('urls.csv')
df['url_length'] = df['url'].apply(len)
df['has_https'] = df['url'].apply(lambda x: int('https' in x))
# Additional features here...

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction and evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
````

---

##  Results

* **Accuracy** ranged from **85% to 96%**, depending on the model.
* **Random Forest** and **SVM** provided the highest accuracy.
* Even lightweight classifiers like **Naive Bayes** performed reasonably well with the engineered features.

---

##  What Was Learned

* Application of machine learning in cybersecurity
* Importance of feature extraction from non-standard textual data
* Comparative analysis of classification models
* Real-world relevance of phishing threats and detection mechanisms

---

##  How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SukanyaGhosh6/phish-shield.git
   cd phish-shield
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main detection script:**

   ```bash
   python phishing_detector.py
   ```

4. **To use your own dataset**, replace `urls.csv` with a custom file following the same structure.

---

##  Future Enhancements

* Deploy as a **Flask-based web application**
* Integrate a **browser extension** for real-time URL scanning
* Explore **deep learning architectures** (e.g., LSTM for sequence modeling)
* Add visualization dashboard for feature insights and predictions

---

##  References

* Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
* Kaggle Phishing Dataset: [https://www.kaggle.com/](https://www.kaggle.com/)
* IEEE Research on Phishing Detection: [https://ieeexplore.ieee.org/document/8755885](https://ieeexplore.ieee.org/document/8755885)
* MIT License Details: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

---

##  Contribute

If you find this project useful, feel free to **star**, **fork**, or contribute. Contributions are welcome via pull requests. Together, we can make the web a safer place.

---

*“Cybersecurity is much more than an IT topic — it’s a public concern.”*

