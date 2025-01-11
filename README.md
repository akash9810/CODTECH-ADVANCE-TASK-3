# CODTECH-ADVANCE-TASK-3

Name:AKASH PV 
Company:CODTECH IT SOLUTIONS 
ID:CTO8DHK 
Domain:Data Science 
Duration:December12th,2024 to January 12th,2025 
Mentor:Neela Santhosh

PROJECT OVER VIEW PROJECT:MOVIE SENTIMENT ANALYSIS Project Description: Sentiment Analysis using LSTM Objective: The primary goal of this project is to build a machine learning model using deep learning techniques to classify user reviews as either positive (1) or negative (0). Sentiment analysis is an essential task in natural language processing (NLP) that helps businesses and researchers understand the emotions and opinions expressed in textual data. Dataset: The dataset consists of user reviews and their associated sentiment labels: Features: The review column contains text data (user reviews). Target: The sentiment column indicates the sentiment label: positive: Represented as 1. negative: Represented as 0. Steps and Implementation: Data Preprocessing:

Replace sentiment labels (positive → 1, negative → 0) for numerical processing.
Split the dataset into training and testing subsets (80% training, 20% testing). Text Tokenization and Padding:
Tokenize the textual data using Tokenizer from TensorFlow/Keras, converting text into sequences of integers.
Pad the tokenized sequences to ensure uniform input lengths, making them suitable for the LSTM model. The maximum length of each sequence is set to **200 Model Architecture: The sentiment analysis model is built using the following layers: Embedding Layer: Converts words into dense vector representations of fixed size (output_dim=128). LSTM Layer: A Long Short-Term Memory (LSTM) layer with 128 units, which captures temporal relationships in the input data while preventing vanishing gradients. Dense Output Layer: A fully connected layer with a single neuron and a sigmoid activation function to predict probabilities for binary classification. Compilation and Training:
The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.
The model is trained on the training dataset for 10 epochs with a batch size of 32. Validation is performed using the testing dataset. Model Evaluation:
The model is evaluated on the testing dataset to compute the test loss and accuracy.
Accuracy and loss trends during training and validation are visualized using Matplotlib. 


![WhatsApp Image 2025-01-11 at 22 40 45_eef0b332](https://github.com/user-attachments/assets/54c1c076-f12d-47bc-a8a4-0a1ec5ea2525)
![WhatsApp Image 2025-01-11 at 22 41 11_bcd02457](https://github.com/user-attachments/assets/608cc6e1-91d0-40b2-8954-fe4bbaa1e87a)

![WhatsApp Image 2025-01-11 at 22 40 59_fce9860d](https://github.com/user-attachments/assets/2f36a36b-4230-4b73-800a-1e5a38d58e2b)



