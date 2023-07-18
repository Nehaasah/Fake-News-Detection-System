# Fake-News-Detection-System

## Overview
The Rise of fake news in today's digital landscape has created an urgent need for powerful and effective methods to detect and combat its spread.
This project presents a sophisticated solution for fake news detection using a combination of ** NLTK (Natural Language Toolkit), scikit-learn, and Logistic Regression **. The system aims to accurately classify news articles as REAL or FAKE.

## Features

- **Text Preprocessing**: NLTK's powerful text preprocessing capabilities like removal of stopwords and Porter stemming are applied. These techniques streamline the text data, reducing noise and improving the accuracy of subsequent analyses.

- **TF-IDF Vectorization**: Scikit-learn's TfidfVectorizer(Term Frequency-Inverse Document Frequency) algorithm is used to convert the preprocessed text data into meaningful numerical representations. This technique assigns higher weights to words that are more relevant to the article while attenuating the impact of common words.[vectorizer = TfidfVectorizer()]

- **Logistic Regression Model**: Logistic regression offers interpretability, scalability, and excellent performance in binary classification.

- **Model Evaluation and Accuracy Metrics**: Scikit-learn's accuracy_score metric is used to rigorously evaluate the trained logistic regression model. This metric provides a comprehensive assessment of the model's performance, allowing for reliable estimation of accuracy.

## Steps Followed to Get the desired output.



1. **Importing Dependencies**: Importing necessary libraries, including numpy, pandas, NLTK, and scikit-learn.

2. **Preparing and Preprocessing the Dataset**: The dataset contains labeled news articles indicating their veracity (real or fake). Utilized pandas to load the dataset and apply preprocessing techniques that inludes removing unwanted characters, converting text to lowercase, removing stopwords, and performing Porter stemming.

3. **Feature Extraction**: Extracting relevant features from the preprocessed text data using scikit-learn's TfidfVectorizer. This transformation converts the text into numerical feature vectors, capturing the essential semantic information required for effective fake news detection.

4. **Spliting the Dataset**: Dividing the dataset into separate training and testing sets. This ensures a comprehensive evaluation of the model's performance.

5. **Train the Logistic Regression Model**: Scikit-learn's LogisticRegression class is instantiated and trained using the training data.

6. **Evaluating the Model**: Evaluating the performance of the model by predicting the labels for the test data. Calculated the accuracy score (using scikit-learn's accuracy_score metric) that came equal to "97.9 % " which provies model's effectiveness in classifying real and fake news articles as extremely good.
   ## Finally A Predictive System is Ready to be used.

