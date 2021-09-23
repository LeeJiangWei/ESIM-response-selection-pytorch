import os
import pickle
from esim.data import Preprocessor


def preprocess_UDC_data(input_dir,
                        embeddings_file,
                        target_dir,
                        lowercase=False,
                        ignore_punctuation=False,
                        num_words=None,
                        stopwords=[],
                        labeldict={}):
    """
    Preprocess the data from the SNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        input_dir: The path to the directory containing the dialogue corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        target_dir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the contexts
            and responses in the input data. Defautls to False.
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
    test_file = "ubuntu_dev_subtask_1.json"

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
    data = preprocessor.read_udc(os.path.join(input_dir, dev_file))

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
    data_dir = "dataset/udc/"
    out_dir = "./tmp"
    preprocess_UDC_data(input_dir=data_dir,
                        embeddings_file=".",
                        target_dir=out_dir)
