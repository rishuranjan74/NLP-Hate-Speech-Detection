# Hate Speech Detection using BERT and LSTM

This project is a comprehensive exploration of detecting hate speech in social media text using various Natural Language Processing (NLP) techniques. The goal is to build and compare different models, from classic baselines to state-of-the-art Transformers, to accurately classify text as hateful or non-hateful.

# Project Overview:
This project implements a binary classification task to identify hate speech. I started with data cleaning and exploratory data analysis to understand the dataset's characteristics, such as comment length and language distribution. Based on this analysis, I decided to focus on monolingual (English) models. I then built, trained, and evaluated three distinct models to compare their performance.

# Dataset:
The project uses the Hate Speech Detection Curated Dataset from Kaggle. The dataset is balanced, containing a significant number of both hateful and non-hateful comments, making it suitable for training a robust classifier. The data was loaded directly into the Google Colab environment using the Kaggle API for reproducibility.

# Methodology:
The project follows a structured approach, comparing three models of increasing complexity:

* Baseline Model: TF-IDF + Logistic Regression
Text was preprocessed by removing stopwords and punctuation, followed by stemming.
The cleaned text was vectorized using TfidfVectorizer.
A LogisticRegression model was trained on these features to establish a baseline performance metric.

* Classic Deep Learning Model: Bidirectional LSTM
A Recurrent Neural Network (RNN) with a bidirectional LSTM layer was built using PyTorch.
This model learns sequential patterns in the text from an embedding layer.

* State-of-the-Art Model: Fine-Tuned BERT
A pre-trained BERT model (bert-base-uncased) was fine-tuned on the hate speech dataset.
This approach leverages transfer learning to achieve high performance with less training data compared to training from scratch.

# Results:

The performance of each model was evaluated using Accuracy and the F1-Score. The results clearly demonstrate the performance gains from using more advanced architectures.

Model1 - TF-IDF + Logistic Regression:
                0.82 (Accuracy), 0.81 (F1-Score_Weighted)
Model2 - Bidirectional LSTM (RNN):
                0.86 (Accuracy), 0.85 (F1-Score_Weighted)
Model3 - Fine-Tuned BERT:
                0.90 (Accuracy), 0.89 (F1-Score_Weighted)

# Conclusion: 

The fine-tuned BERT model significantly outperformed the other models, highlighting the effectiveness of Transformer architectures on complex NLP tasks.
Technologies Used:
1.Python
2.Pandas & NumPy
3.Scikit-learn
4.NLTK
5.PyTorch
6.Hugging Face Transformers
7.Google Colab
8.Kaggle API
