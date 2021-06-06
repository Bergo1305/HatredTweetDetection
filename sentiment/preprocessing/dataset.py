import csv
import nltk
import pandas as pd

import concurrent.futures
from typing import Text, Union
from sentiment.preprocessing.Augmentation import Augmentation
from sentiment.config import TRAIN_FILE_PATH, TEST_FILE_PATH, CURRENT_DIR, logger

nltk.download('wordnet')


class DataFrame(object):

    def __init__(self, obj: pd.DataFrame):

        self.augmentation = Augmentation()
        self._obj = obj
        self._augment_obj = self.augment()

    def augment(self):

        with concurrent.futures.ProcessPoolExecutor() as exc:

            results = exc.map(self.augmentation.augment, [row["tweet"] for _, row in list(self._obj.iterrows())])

        return results

    def text(self):
        return self._augment_obj

    def labels(self):

        try:
            return [
                row['label'] for _, row in list(self._obj.iterrows())
            ]
        except Exception as _exc:
            logger.exception(_exc)
            return None


class Dataset(object):

    def __init__(self, train_obj, test_obj):

        self._train_obj = DataFrame(train_obj)
        # self._test_obj = DataFrame(test_obj)

    @property
    def train(self):
        return {
            "text": [data_point for data_point in self._train_obj.text()],
            "label": [str(label) for label in self._train_obj.labels()]
        }

    # @property
    # def test(self):
    #     return {
    #         "text": [data_point for data_point in self._test_obj.text()]
    #     }


def generate(
        train_dataset_path: Union[None, Text] = None,
        test_dataset_path: Union[None, Text] = None,
        save_path: bool = True
):

    if train_dataset_path is None:
        train_dataset_reader = pd.read_csv(TRAIN_FILE_PATH, sep=",")

    else:
        train_dataset_reader = pd.read_csv(train_dataset_path, sep=",")

    if test_dataset_path is None:
        test_dataset_reader = pd.read_csv(TEST_FILE_PATH, sep=",")
    else:
        test_dataset_reader = pd.read_csv(test_dataset_path, sep=",")

    dataset = Dataset(train_dataset_reader, test_dataset_reader)

    train_obj = dataset.train

    if save_path:

        with open(f"{CURRENT_DIR}/data/aug_train.csv", "w") as _file:

            csv_writer = csv.writer(_file)
            csv_writer.writerow(["_id", "label", "tweet"])

            for _idx, tweet in enumerate(train_obj['text']):

                csv_writer.writerow([_idx + 1, train_obj['label'][_idx], tweet])


if __name__ == "__main__":

    with open(f"{CURRENT_DIR}/data/train.csv", "r") as _file:

        csv_reader = csv.reader(_file)

        x = 1

        for row in csv_reader:
            if x - 1 == 16:

                a = Augmentation()
                s = a.augment(row[2])
                print(s)
                break

            x += 1


