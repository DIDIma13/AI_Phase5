# AI_Phase5
PROBLEM  STATEMENT:
The problem is to develop a fake news detection model using a Kaggle dataset. The goal is to distinguish between genuine and fake news articles based on their titles and text. This project involves using natural language processing (NLP) techniques to pre-process the text data, building a machine learning model for classification, and evaluating the model's performance.

DESIGN THINKING PROCESS:
1. DATA SOURCE
Action: Choose the Kaggle dataset containing news articles, titles, and labels (genuine or fake).
Rationale: Selecting a relevant and high-quality dataset is the foundation of building an effective fake news detection model. Kaggle provides a diverse range of datasets suitable for this task.

2. DATA PREPROCESSING
Action: Clean and preprocess the textual data to prepare it for analysis.
Rationale: Data preprocessing is essential to ensure that the text data is consistent, free from noise, and suitable for further analysis. Steps may include:
- Removing special characters, punctuation, and HTML tags.
- Tokenization (splitting text into words or tokens).
- Lowercasing all text to ensure consistency.
- Removing stop words (common words like &#39;the,&#39; &#39;and,&#39; &#39;is,&#39; etc. that don&#39;t carry significant meaning).
- Handling missing data if any.

3. FEATURE EXTRACTION
Action: Utilize techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings to convert text into numerical features.
Rationale: Converting text into numerical features is necessary for machine learning models to process the data. TF-IDF and word embeddings like Word2Vec or GloVe can capture semantic information and relationships between words, making them suitable for this task.

4. MODEL SELECTION
Action: Select a suitable classification algorithm (e.g., Logistic Regression, Random Forest, or Neural Networks) for the fake news detection task.
Rationale: The choice of a classification algorithm can significantly impact the model&#39;s performance. Different algorithms have different strengths and weaknesses, and the choice should be based on the dataset&#39;s characteristics and complexity. Commonly used algorithms for text classification include Logistic Regression, Random Forest, and Neural Networks. Evaluating multiple algorithms may be necessary to identify the best-performing one.

5. MODEL TRAINING
Action: Train the selected model using the preprocessed data.
Rationale: Model training involves feeding the preprocessed data into the chosen classification algorithm. The model learns to identify patterns and relationships between the text features and their corresponding labels (genuine or fake news).

6. EVALUATION
Action: Evaluate the model&#39;s performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
Rationale: Model evaluation is crucial to assess how well the fake news detection system is performing. The selected metrics provide insights into various aspects of the model&#39;s performance:
- Accuracy: Overall correctness of the model&#39;s predictions.
- Precision: Proportion of true positive predictions among all positive predictions, measuring the model&#39;s ability to avoid false positives.
- Recall: Proportion of true positive predictions among all actual positives, measuring the model&#39;s ability to capture genuine fake news.
- F1-score: Harmonic mean of precision and recall, offering a balanced measure of model performance.
- ROC-AUC: Receiver Operating Characteristic - Area Under the Curve measures the model&#39;s ability to distinguish between the two classes.

ENHANCEMENTS USING BERT AND LSTM
To further enhance the accuracy of the fake news detection model, consider incorporating advanced techniques such as BERT (Bidirectional Encoder Representations from Transformers) and LSTM (Long Short-Term Memory) networks.

7. BERT INTEGRATION
Action: Implement BERT-based models for feature extraction.
Rationale: BERT, as a pre-trained transformer model, captures contextual information and relationships between words in a sentence. By using BERT embeddings, the model can better understand the semantics of the text, improving the representation of complex language constructs in news articles.

8. LSTM INTEGRATION
Action: Incorporate LSTM layers into the neural network architecture.
Rationale: LSTM networks are effective in capturing sequential dependencies in data. Since news articles often have a sequential structure, LSTM layers can enhance the model&#39;s ability to understand the temporal aspects of language. This is particularly useful for detecting nuances and context in fake news articles.
9. MODEL TRAINING AND EVALUATION
Action: Retrain the model using the enhanced feature extraction techniques (BERT and LSTM).
Rationale: By leveraging BERT and LSTM, the model can potentially achieve higher accuracy and better generalization to complex patterns in the data. Retrain the model and evaluate its performance using the defined metrics to ensure improvements in fake news detection accuracy.
DATASET LINK:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
DESCRIPTION:
The significance of social media has increased manifold in the past few decades as it helps people from even the most remote corners of the world stay connected. With the COVID-19 pandemic raging, social media has become more relevant and widely used than ever before, and along with this, there has been a resurgence in the circulation of fake news and tweets that demand immediate attention. In this paper, we describe our Fake News Detection system that automatically identifies whether a tweet related to COVID-19 is "real" or "fake", as a part of CONSTRAINT COVID19 Fake News Detection in English challenge. We have used an ensemble model consisting of pre-trained models that has helped us achieve a joint 8th position on the leader board. We have achieved an F1-score of 0.9831 against a top score of 0.9869. Post completion of the competition, we have been able to drastically improve our system by incorporating a novel heuristic algorithm based on username handles and link domains in tweets fetching an F1-score of 0.9883 and achieving state-of-the art results on the given dataset.

CONCLUSION: 
The problem is to develop a fake news detection model using a Kaggle dataset. The goal is to distinguish between genuine and fake news articles based on their titles and text. This project involves using natural language processing (NLP) techniques to pre-process the text data, building a machine learning model for classification, and evaluating the model's performance.
