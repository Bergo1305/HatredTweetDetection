from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
import os
import csv
import pickle
import numpy as np
from typing import Text
from itertools import tee
from datetime import datetime

import tensorflow as tf
print(tf.__version__)
from imblearn.under_sampling import TomekLinks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment.config import AUGMENTED_TRAIN_FILE_PATH, CURRENT_DIR
from sklearn.metrics import precision_recall_fscore_support


def get_current_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def construct_count_vectorized_data(
        dataset_path: Text = AUGMENTED_TRAIN_FILE_PATH
):

    with open(dataset_path, "r") as _file:

        csv_reader = csv.reader(_file)

        _its = tee(csv_reader)

        labels, tweets = (
            [
                int(row[1]) for row in _its[0] if row[1] != "label"
            ],
            [
                row[2] for row in _its[1] if row[1] != "label"
            ]
        )

    X_train, X_test, Y_train, Y_test = train_test_split(tweets, labels, test_size=0.2)

    vectorized = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.75,
        min_df=5,
        max_features=5000
    )

    train_data_features = vectorized.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorized.transform(X_test)
    test_data_features = test_data_features.toarray()

    model_root_path = f"{CURRENT_DIR}/training/models/LSTM"
    pickle.dump(vectorized, open(f"{model_root_path}/vocabulary.sav", 'wb'))

    return train_data_features, np.array(Y_train), test_data_features, np.array(Y_test)


def train() -> None:

    model_root_path = f"{CURRENT_DIR}/training/models/LSTM"

    if not os.path.isdir(model_root_path):
        os.mkdir(model_root_path)

    logs_file = open(f"{model_root_path}/model.logs", "w")

    logs_file.write(f"Training started in {get_current_time()} \n")

    X_train, Y_train, X_test, Y_test = construct_count_vectorized_data()

    model = Sequential()
    model.add(Embedding(5000, 128, input_length=5000))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=32,
              epochs=1,
              validation_data=[X_test, Y_test])

    logs_file.write(f"Training finished in {get_current_time()} \n")

    model.save(f"{model_root_path}/model.h5")

    logs_file.write(f"******************************************* \n")

    logs_file.write(f"Testing started in {get_current_time()} \n")
    predicted = model.predict(X_test)
    accuracy = np.mean(predicted == Y_test)
    logs_file.write(f"Testing finished in {get_current_time()} \n")

    pickle.dump(model, open(f"{model_root_path}/logistic_regrsion.sav", 'wb'))

    logs_file.write(f"******************************************* \n")
    logs_file.write(f"Accuracy: {accuracy} \n")
    score_svm = precision_recall_fscore_support(Y_test, predicted, average='weighted')
    logs_file.write(f"Precision: {score_svm[0]} \n")
    logs_file.write(f"Recall: {score_svm[1]} \n")
    logs_file.write(f"FScore: {score_svm[2]} \n")
    logs_file.write(f"Support: {score_svm[3]} \n")

    logs_file.close()


if __name__ == "__main__":
    train()
