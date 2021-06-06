import os
import logging


def create_logger(name, level=logging.DEBUG):

    logg = logging.getLogger(name)
    logg.setLevel(level)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(
        logging.Formatter(
            '[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s'
        )
    )

    logg.addHandler(stdout_handler)

    return logg


logger = create_logger("SENTIMENT-ANALYSIS_IMDB")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_PATH = f"{CURRENT_DIR}/data"

TRAIN_FILE_PATH = f"{DATASET_PATH}/train.csv"

TEST_FILE_PATH = f"{DATASET_PATH}/test.csv"

AUGMENTED_TRAIN_FILE_PATH = f"{DATASET_PATH}/aug_train.csv"

