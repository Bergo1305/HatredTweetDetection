import pickle
from typing import Text
from sentiment.preprocessing.Augmentation import Augmentation


def evaluate_from_input(
        input_tweet: Text,
        model_path: Text,
        vocabulary_path: Text
):
    with open(f"{model_path}", 'rb') as f:
        model = pickle.load(f)

    with open(f"{vocabulary_path}", 'rb') as f:
        vocabulary = pickle.load(f)

    augmentation = Augmentation()
    clear_tweet = augmentation.augment(input_tweet)

    input_array_transformed = vocabulary.transform([clear_tweet]).toarray()

    prediction = model.predict(input_array_transformed)

    return prediction

