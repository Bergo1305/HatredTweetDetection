import csv
import matplotlib.pyplot as plt
from typing import Union, Text
from collections import Counter
from sentiment.config import TRAIN_FILE_PATH, CURRENT_DIR


def percentage(part, whole):
    return 100 * float(part)/float(whole)


def labels_chart(
    train_path: Union[None, Text] = None
):

    if train_path is None:
        train_path = TRAIN_FILE_PATH

    positive_count: int = 0
    negative_count: int = 0

    with open(train_path, "r") as _file:
        csv_reader = csv.reader(_file)

        for row in csv_reader:

            if row[1] == "label":
                continue

            label = int(row[1])

            if label == 0:
                positive_count += 1

            elif label == 1:
                negative_count += 1

    labels = [
        f"Positive [ {positive_count} ]",
        f"Negative [ {negative_count} ]"
    ]

    plt.pie(
        [positive_count, negative_count],
        colors=['blue', 'red'],
        startangle=90
    )

    plt.style.use("default")
    plt.legend(labels)
    plt.title(f"Labels [positive negative tweets]")
    plt.axis('equal')
    plt.savefig(f"{CURRENT_DIR}/visualizer/images/labels.png")
    plt.show()


def count_longest(train_path: None = None):
    if train_path is None:
        train_path = TRAIN_FILE_PATH

    maximal_count: int = 100

    with open(train_path, "r") as _file:
        csv_reader = csv.reader(_file)

        for row in csv_reader:

            if row[1] == "label":
                continue

            no_words = len(row[2].split(" "))

            if no_words < maximal_count:
                maximal_count = no_words

    print(maximal_count)


def words_statistics():

    words_counter = {}

    with open(TRAIN_FILE_PATH, "r") as _file:

        csv_reader = csv.reader(_file)

        for row in csv_reader:

            text = row[2]

            for word in text.split(" "):

                if word:
                    if words_counter.get(word) is None:
                        words_counter[word] = 1

                    else:
                        words_counter[word] += 1

    n_largest = sorted(words_counter.items(), key=lambda pair: -pair[1] )[:10]
    n_smallest = sorted(words_counter.items(), key=lambda pair: pair[1])[:10]

    print(n_largest)
    print(n_smallest)

    labels = [x[0] for x in n_smallest]
    left = range(len(n_smallest))
    height = [x[1] for x in n_smallest]

    # plotting a bar chart
    plt.bar(left, height, tick_label=labels,
            width=0.8, color=['tab:blue'])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('Least common phrases!')

    plt.savefig(f"{CURRENT_DIR}/visualizer/images/least_common.png")


if __name__ == "__main__":
    words_statistics()