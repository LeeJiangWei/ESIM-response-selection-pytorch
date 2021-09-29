import os
import json
import pickle
from tqdm import tqdm

from esim.data import Preprocessor


def transform_UDC_testset(test_file, label_file):
    """
    Transform UDC test set with labels so that it has the same structure as
    training data and dev data.

    Args:
        test_file: The path to the test data file.
        label_file: The path to the test label file.
    """
    true_responses = []
    with open(label_file, encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")

            true_responses.append({
                "example-id": int(line[0]),
                "candidate-id": line[1],
                "utterance": line[2]
            })

    with open(test_file, encoding="utf8") as f:
        examples = json.load(f)
        for index, example in enumerate(tqdm(examples)):
            assert example["example-id"] == true_responses[index]["example-id"]

            example["options-for-correct-answers"] = [{
                "candidate-id": true_responses[index]["candidate-id"],
                "utterance": true_responses[index]["utterance"]
            }]

        with open("../dataset/udc/transformed_ubuntu_test.json", "w", encoding="utf8") as o:
            json.dump(examples, o, indent=4)


def preprocess_UDC_data(input_dir,
                        target_dir,
                        lowercase=False,
                        ignore_punctuation=False,
                        num_words=None,
                        stopwords=[],
                        labeldict={}):
    """
    Preprocess the data from the UDC corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        input_dir: The path to the directory containing the dialogue corpus.
            word vectors that must be used to build the embedding matrix.
        target_dir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the contexts
            and responses in the input data. Defaults to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = "ubuntu_train_subtask_1.json"
    dev_file = "ubuntu_dev_subtask_1.json"
    test_file = "transformed_ubuntu_test.json"

    # -------------------- Train data preprocessing -------------------- #
    print(20 * "=", " Preprocessing UDC train data set ", 20 * "=")
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict)

    print("* Reading data...")
    data = preprocessor.read_udc(os.path.join(input_dir, train_file))

    preprocessor.build_worddict(data)
    with open(os.path.join(target_dir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    transformed_data = preprocessor.transform_to_indices(data)

    print("* Saving result...")
    with open(os.path.join(target_dir, "udc_train_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20 * "=", " Preprocessing UDC validation data set ", 20 * "=")
    print("* Reading data...")
    data = preprocessor.read_udc(os.path.join(input_dir, dev_file), sample_size=0)

    transformed_data = preprocessor.transform_to_indices(data)

    print("* Saving result...")
    with open(os.path.join(target_dir, "udc_dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20 * "=", " Preprocessing UDC test data set ", 20 * "=")
    print("* Reading data...")
    data = preprocessor.read_udc(os.path.join(input_dir, test_file), sample_size=0)

    transformed_data = preprocessor.transform_to_indices(data)

    print("* Saving result...")
    with open(os.path.join(target_dir, "udc_test_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20 * "=", " Preprocessing embeddings ", 20 * "=")
    print("* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_pretrained_spacy_embedding_matrix()
    with open(os.path.join(target_dir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == '__main__':
    # transform_UDC_testset("../dataset/udc/ubuntu_test_subtask_1.json",
    #                       "../dataset/udc/ubuntu_responses_subtask_1.tsv")

    data_dir = "../dataset/udc/"
    out_dir = "../tmp"

    preprocess_UDC_data(input_dir=data_dir,
                        target_dir=out_dir)
