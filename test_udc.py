"""
Utility functions for testing models.
"""
import pickle

import torch
from torch.utils.data import DataLoader

from esim.data import DialogueDataset
from esim.model import ESIM
from utils import test


def main(test_file,
         embeddings_file,
         checkpoint,
         hidden_size=300,
         dropout=0.5,
         num_classes=2,
         k=1):
    """
    Train the ESIM model on the UDC dataset.

    Args:

        test_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 2.
        k: The number of recall at k.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------- Data loading ------------------- #

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = DialogueDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=100)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float).to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model"])

    test_time, test_recall = test(model, test_loader, k)
    print(f"Testing time: {test_time}, recall@{k}: {test_recall}")


if __name__ == "__main__":
    main(
        test_file="tmp/udc_test_data.pkl",
        embeddings_file="tmp/embeddings.pkl",
        checkpoint="./tmp/out/best.pth.tar"
    )
