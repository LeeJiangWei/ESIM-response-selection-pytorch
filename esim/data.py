"""
Preprocessor and dataset definition for dialogue.
"""

import re
import json
import string
from collections import Counter

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Preprocessor(object):
    """
    Preprocessor class for UDC datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their contexts, responses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 ):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict

    def spacy_tokenize(self, sentence, model_name="en_core_web_md"):
        tokens = []

        nlp = spacy.blank("en")
        doc = nlp(sentence)
        for token in doc:
            tokens.append(token.text)

        return tokens

    def whitespace_tokenize(self, sentence):
        return [w for w in sentence.rstrip().split() if w not in self.stopwords]

    def read_douban(self, filepath):
        """
        Read the contexts, responses and labels from Douban dataset's
        file and return them in a dictionary.

        Args:
            filepath: The path to a file containing some contexts, responses
                and labels that must be read.

        Returns:
            A dictionary containing three lists, one for the contexts, one for
            the responses, and one for the labels in the input data.
        """

        # match all zh char, en char, number and whitespace
        regex = re.compile(r'[^\u4E00-\u9FA5A-Za-z0-9 ]+')

        with open(filepath, "r", encoding="utf8") as f:
            contexts, responses, labels = [], [], []

            for line in f:
                line = line.strip().split("\t")

                context = " __eou__ __eot__ ".join(line[1:-1]) + " __eou__ __eot__"
                response = line[-1]
                label = int(line[0])

                # replace special tokens
                context = regex.sub("", context)
                response = regex.sub("", response)

                # tokenize text
                context_tokens = self.whitespace_tokenize(context)
                response_tokens = self.whitespace_tokenize(response)

                # add a sample
                contexts.append(context_tokens)
                responses.append(response_tokens)
                labels.append(label)

            return {"contexts": contexts,
                    "responses": responses,
                    "labels": labels}

    def read_udc(self, filepath, sample_size=4):
        """
        Read the contexts, responses and labels from UDC dataset's json
        file and return them in a dictionary.

        Args:
            filepath: The path to a file containing some contexts, responses
                and labels that must be read.
            sample_size: The size of negative sample responses in training set.

        Returns:
            A dictionary containing three lists, one for the contexts, one for
            the responses, and one for the labels in the input data.
        """

        # Translation tables to remove parentheses and punctuation from strings.
        parentheses_table = str.maketrans({"(": None, ")": None})
        punct_table = str.maketrans({key: " " for key in string.punctuation})

        with open(filepath, "r", encoding="utf8") as f:
            input_data = json.load(f)
            contexts, responses, labels = [], [], []

            for example in tqdm(input_data, "* Generating examples"):
                context_utterances = [t["utterance"] for t in example["messages-so-far"]]
                context = " __eou__ __eot__ ".join(context_utterances)
                context += " __eou__ __eot__"

                gt_id = example["options-for-correct-answers"][0]["candidate-id"]
                response = example["options-for-correct-answers"][0]["utterance"]

                # delete parentheses
                context = context.translate(parentheses_table)
                response = response.translate(parentheses_table)

                if self.lowercase:
                    context = context.lower()
                    response = response.lower()

                # delete punctuation
                if self.ignore_punctuation:
                    context = context.translate(punct_table)
                    response = response.translate(punct_table)

                # tokenize context
                context_tokens = self.whitespace_tokenize(context)
                response_tokens = self.whitespace_tokenize(response)

                # add positive example
                contexts.append(context_tokens)
                responses.append(response_tokens)
                labels.append(1)

                if sample_size <= 0:
                    sample_size = len(example["options-for-next"])

                # random select 4 negative example for same context
                for negative_turn in np.random.choice(example["options-for-next"], sample_size):
                    if gt_id == negative_turn["candidate-id"]:
                        continue

                    negative_response = negative_turn["utterance"]

                    if self.lowercase:
                        negative_response = negative_response.lower()

                    if self.ignore_punctuation:
                        negative_response = negative_response.translate(punct_table)

                    negative_response_tokens = self.whitespace_tokenize(negative_response)

                    contexts.append(context_tokens)
                    responses.append(negative_response_tokens)
                    labels.append(0)

        return {"contexts": contexts,
                "responses": responses,
                "labels": labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the contexts, responses and
                labels of some dialogue dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["contexts"]]
        [words.extend(sentence) for sentence in data["responses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1
        self.worddict["_BOS_"] = 2
        self.worddict["_EOS_"] = 3
        offset = 4

        for i, word in tqdm(enumerate(counts.most_common(num_words)), "* Building word dict"):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []

        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values()).index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the contexts and responses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of contexts, responses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed contexts, responses and
            labels.
        """
        transformed_data = {"contexts": [],
                            "responses": [],
                            "labels": []}

        for i, context in tqdm(enumerate(data["contexts"]), "* Transforming sentences to indices"):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            context_indices = self.words_to_indices(context)
            transformed_data["contexts"].append(context_indices)

            responses_indices = self.words_to_indices(data["responses"][i])
            transformed_data["responses"].append(responses_indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in tqdm(self.worddict.items(), "* Building embedding matrix"):
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian samples.
                embedding_matrix[i] = np.random.normal(size=embedding_dim)
        print("Missed words: ", missed)

        return embedding_matrix

    def build_pretrained_spacy_embedding_matrix(self, model_name="en_core_web_md"):
        nlp = spacy.load(model_name)

        num_words = len(self.worddict)
        embedding_dim = nlp.vocab.vectors_length
        embedding_matrix = np.zeros((num_words, embedding_dim))

        missed = 0
        for word, i in tqdm(self.worddict.items(), "* Loading spacy vectors"):
            lexeme = nlp.vocab[word]
            if not lexeme.is_oov:
                embedding_matrix[i] = lexeme.vector
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian samples.
                embedding_matrix[i] = np.random.normal(size=embedding_dim)
        print("Missed words: ", missed)

        return embedding_matrix


class DialogueDataset(Dataset):
    """
    Dataset class for retrieval dialogue datasets.

    The class can be used to read preprocessed datasets where the contexts,
    responses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_context_length=None,
                 max_response_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed contexts,
                responses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_context_length: An integer indicating the maximum length
                accepted for the sequences in the context. If set to None,
                the length of the longest context in 'data' is used.
                Defaults to None.
            max_response_length: An integer indicating the maximum length
                accepted for the sequences in the response. If set to None,
e                Defaults to None.
        """
        self.context_lengths = [len(seq) for seq in data["contexts"]]
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = max(self.context_lengths)

        self.response_lengths = [len(seq) for seq in data["responses"]]
        self.max_response_length = max_response_length
        if self.max_response_length is None:
            self.max_response_length = max(self.response_lengths)

        self.num_sequences = len(data["contexts"])

        self.data = {"contexts": torch.ones((self.num_sequences,
                                             self.max_context_length),
                                            dtype=torch.long) * padding_idx,
                     "responses": torch.ones((self.num_sequences,
                                              self.max_response_length),
                                             dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, context in enumerate(data["contexts"]):
            end = min(len(context), self.max_context_length)
            self.data["contexts"][i][:end] = torch.tensor(context[-end:])  # cut context in reverse direction

            response = data["responses"][i]
            end = min(len(response), self.max_response_length)
            self.data["responses"][i][:end] = torch.tensor(response[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"context": self.data["contexts"][index],
                "context_length": min(self.context_lengths[index],
                                      self.max_context_length),
                "response": self.data["responses"][index],
                "response_length": min(self.response_lengths[index],
                                       self.max_response_length),
                "label": self.data["labels"][index]}
