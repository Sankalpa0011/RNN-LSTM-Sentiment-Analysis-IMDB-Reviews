---

# RNN LSTM Sentiment Analysis on IMDB Reviews

---

This repository contains a project that performs sentiment analysis on IMDB reviews using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The goal is to classify movie reviews as positive or negative based on their text content.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Sentiment analysis is a key application of natural language processing (NLP) that involves determining the sentiment expressed in a piece of text. This project utilizes RNN and LSTM models to analyze the sentiment of movie reviews from the IMDB dataset. The implementation is done using Python and popular deep learning libraries such as TensorFlow and Keras.

## Installation

To run this project, you need to have Python installed on your system. Additionally, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

If you are using Google Colab, you can also install the necessary packages directly in the notebook:

```python
!pip install kaggle
```

## Dataset

The dataset used in this project is the IMDB movie reviews dataset. You can download the dataset using the Kaggle API. Ensure that you have the Kaggle API configured on your system. The dataset will be automatically downloaded and loaded into the project.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

1. **Import Libraries:**

   Import the necessary libraries required for data processing, model building, and evaluation.

   ```python
   import os
   import json

   import pandas as pd
   import numpy as np
   import tensorflow as tf

   import matplotlib.pyplot as plt
   import seaborn as sns

   from sklearn.model_selection import train_test_split
   from keras.models import Sequential
   from keras.layers import Dense, Dropout, Embedding, LSTM
   from keras.preprocessing.text import Tokenizer
   from keras.preprocessing.sequence import pad_sequences
   from keras.utils import to_categorical
   ```

2. **Data Collection:**

   Use the Kaggle API to download the IMDB reviews dataset.

3. **Data Preprocessing:**

   - Tokenize the text data
   - Pad the sequences
   - Split the data into training and testing sets

4. **Model Building:**

   Build an RNN LSTM model using Keras.

   ```python
   model = Sequential()
   model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
   model.add(LSTM(units=128, return_sequences=True))
   model.add(Dropout(0.2))
   model.add(LSTM(units=128))
   model.add(Dropout(0.2))
   model.add(Dense(units=1, activation='sigmoid'))
   ```

5. **Model Training:**

   Train the model using the training data.

   ```python
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
   ```

6. **Model Evaluation:**

   Evaluate the model on the test data and visualize the results.

   ```python
   loss, accuracy = model.evaluate(X_test, y_test)
   print(f'Test Accuracy: {accuracy}')
   ```

## Model Architecture

The model consists of the following layers:

- Embedding Layer
- LSTM Layers
- Dropout Layers
- Dense Layer

## Results

The results of the model, including the accuracy and loss on the test data, will be displayed and visualized using plots.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

---
