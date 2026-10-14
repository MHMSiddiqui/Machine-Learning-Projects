# 🐦 Twitter Sentiment Analysis Using Python

A machine learning project focused on classifying tweets as **positive or negative** using Python and Scikit-learn. The project works with the Sentiment140 dataset and applies text preprocessing, TF-IDF feature extraction, supervised classification, and model evaluation to understand sentiment expressed in tweets.

## ✨ Key Features

- **Dataset Exploration**: Loads the Sentiment140 dataset, selects the tweet text and sentiment label columns, and inspects the dataset structure.
- **Sentiment Label Preparation**: Maps negative tweets (`0`) and positive tweets (`4`) into binary labels (`0` and `1`). The workflow excludes neutral labels (`2`) if they are present in the source data.
- **Text Preprocessing**: Converts tweet text to lowercase before feature extraction.
- **TF-IDF Feature Extraction**: Transforms text into numerical features using `TfidfVectorizer`, with up to 5,000 features and unigram/bigram terms.
- **Model Training**: Trains Bernoulli Naive Bayes, Linear Support Vector Classification (LinearSVC), and Logistic Regression models.
- **Model Evaluation**: Compares model performance using accuracy, precision, recall, and F1-score on a held-out test set.
- **Sample Predictions**: Tests the trained models on example tweets, such as positive, negative, and mixed-opinion text.

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Data Handling**: Pandas
- **Machine Learning**: Scikit-learn
- **Text Feature Extraction**: TF-IDF (`TfidfVectorizer`)
- **Models**: Bernoulli Naive Bayes, LinearSVC, Logistic Regression
- **Environment**: Jupyter Notebook
- **Data Source**: Kaggle — Sentiment140 dataset

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or a compatible Python version
- Jupyter Notebook
- Basic understanding of Python and machine learning
- The Sentiment140 CSV dataset (approximately 239 MB)

### Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/MHMSiddiqui/Twitter-Sentiment-Analysis.git
   cd Twitter-Sentiment-Analysis
   ```

   Replace the repository URL and folder name if your GitHub repository uses a different name.

2. **Install dependencies**

   ```bash
   pip install pandas scikit-learn jupyter
   ```

3. **Download the dataset**

   Download the **Sentiment140 dataset with 1.6 million tweets** from Kaggle:

   **Dataset link:** https://www.kaggle.com/datasets/kazanova/sentiment140

   Extract the archive and place `training.1600000.processed.noemoticon.csv` in the project directory, or update the notebook's file path to its actual location. The dataset is large, so review the dataset's terms before redistributing it or committing it to a public repository.

4. **Load the dataset**

   The notebook originally contains a local Windows file path. Update that path, or use a relative path such as:

   ```python
   import pandas as pd

   df = pd.read_csv(
       "training.1600000.processed.noemoticon.csv",
       encoding="latin-1",
       header=None
   )
   df = df[[0, 5]].copy()
   df.columns = ["polarity", "text"]
   ```

5. **Run the notebook**

   ```bash
   jupyter notebook Twitter_Sentiment_Analysis.ipynb
   ```

   Run the cells in order to prepare the labels, preprocess the text, extract TF-IDF features, train the models, and review the evaluation results.

---

## 📖 Usage Guide

The project follows five main analytical stages.

### 1. Dataset Exploration

- Load the CSV without a header and use the `latin-1` encoding.
- Select the sentiment label and tweet text columns.
- Inspect sample rows, data dimensions, and label distribution.
- Work with the 1.6 million records available in the provided dataset.

### 2. Sentiment Preparation and Text Preprocessing

- Map negative sentiment (`0`) to class `0` and positive sentiment (`4`) to class `1`.
- Exclude neutral class (`2`) records if they occur in the loaded data.
- Convert tweet text to lowercase.
- Split the data into training and test sets using an 80/20 split and `random_state=42`.

### 3. Text Feature Extraction

- Use `TfidfVectorizer` to convert tweet text into numeric features.
- Limit the vocabulary to 5,000 features.
- Include both unigram and bigram features with `ngram_range=(1, 2)`.
- Fit the vectorizer on the training data and apply the same transformation to the test data.

### 4. Model Training and Evaluation

- Train a Bernoulli Naive Bayes classifier.
- Train a Linear Support Vector Classifier (LinearSVC).
- Train a Logistic Regression classifier.
- Compare accuracy and classification reports, including precision, recall, and F1-score.

### 5. Sample Sentiment Prediction

- Pass example tweets through the fitted TF-IDF vectorizer.
- Generate predictions using all three classifiers.
- Compare how each model labels positive, negative, or mixed-opinion example text.

## 📊 Model Results

The following test accuracies are recorded in the notebook after an 80/20 train-test split, with 1,280,000 training records and 320,000 test records.

| Model | Test Accuracy |
| --- | ---: |
| Bernoulli Naive Bayes | 76.65% |
| LinearSVC | 79.53% |
| Logistic Regression | 79.54% |

**Best recorded result:** Logistic Regression achieved **79.54% test accuracy**, narrowly outperforming LinearSVC in this run. These are the results saved in the notebook; retraining may produce different results depending on the environment and data preparation.

## ❓ Questions This Project Explores

- Can tweet text be used to classify sentiment as positive or negative?
- How can text be converted into numerical features for machine learning?
- How do Bernoulli Naive Bayes, LinearSVC, and Logistic Regression compare on the same test set?
- What precision, recall, and F1-score do the models achieve for each sentiment class?
- How do the trained models classify new example tweets?

## 📂 Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── Twitter_Sentiment_Analysis.ipynb
├── training.1600000.processed.noemoticon.csv  # Download separately from Kaggle
└── README.md
```

The CSV is large (approximately 239 MB). Consider excluding it from Git and adding it to `.gitignore`; provide the Kaggle link instead of committing the dataset directly to the repository.

## ⚙️ Key Python and Machine Learning Concepts Used

| Concept / Tool | Application |
| --- | --- |
| `pandas.read_csv()` | Load the raw CSV dataset |
| DataFrame column selection | Keep the sentiment label and tweet text |
| Label mapping | Convert source labels into binary classes |
| `train_test_split()` | Create separate training and test sets |
| `TfidfVectorizer` | Convert text into unigram and bigram TF-IDF features |
| `BernoulliNB` | Train a Naive Bayes classifier |
| `LinearSVC` | Train a linear support vector classifier |
| `LogisticRegression` | Train a logistic regression classifier |
| `accuracy_score()` | Calculate overall test accuracy |
| `classification_report()` | Review precision, recall, and F1-score by class |

## 📈 Skills Demonstrated

- Python and Pandas for working with large text datasets
- Text preprocessing and binary sentiment classification
- TF-IDF vectorization with unigram and bigram features
- Training and comparing supervised learning models
- Model evaluation using accuracy, precision, recall, and F1-score
- Interpreting classification reports and test-set performance

## 🔍 Potential Applications

The techniques demonstrated in this project can be adapted for:

- **Social Media Monitoring**: Classifying sentiment expressed in public posts.
- **Customer Feedback Analysis**: Reviewing positive and negative text feedback.
- **Brand Perception Research**: Summarizing sentiment patterns across a collection of text data.
- **Text Classification Workflows**: Serving as a starting point for other supervised NLP classification tasks.

*Note: This notebook performs binary sentiment classification and uses a relatively simple text-cleaning step that converts text to lowercase. It does not establish sarcasm, context, sentiment intensity, or reliable performance on every topic or modern social-media dataset.*

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push your branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## 📄 License

No project license file is specified in this repository by this README. Add a `LICENSE` file if you choose to license the project code. The dataset has separate terms; review the Kaggle dataset page before redistributing it.

