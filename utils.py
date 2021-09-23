import time
import torch

import torch.nn as nn
from tqdm import tqdm


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def recall_at_k(output_probabilities, targets, k=1):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    max_value, out_classes = output_probabilities.max(dim=1)


def train(model, dataloader, optimizer, criterion, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        contexts = batch["context"].to(device)
        contexts_lengths = batch["context_length"].to(device)
        responses = batch["response"].to(device)
        responses_lengths = batch["response_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(contexts,
                              contexts_lengths,
                              responses,
                              responses_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.


    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            contexts = batch["context"].to(device)
            contexts_lengths = batch["context_length"].to(device)
            responses = batch["response"].to(device)
            responses_lengths = batch["response_length"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(contexts,
                                  contexts_lengths,
                                  responses,
                                  responses_lengths)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy


def test(model, transformed_contexts, transformed_responses, labels):
    """
    Run inference once with batched transformed contexts and responses.

    Args:
        model: A torch module for which the loss and accuracy must be computed.
        transformed_contexts: A DataLoader object to iterate over the validation data.
        transformed_responses: A loss criterion to use for computing the loss.

    Returns:
        batch_time: The total time to compute the loss and accuracy on the batch.
        batch_loss: The loss computed on the batch.
        batch_accuracy: The accuracy computed on the batch.
    """

    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        contexts = torch.tensor(transformed_contexts).to(device)
        contexts_lengths = torch.tensor([len(s) for s in transformed_contexts]).to(device)
        responses = torch.tensor(transformed_responses).to(device)
        responses_lengths = torch.tensor([len(s) for s in transformed_responses]).to(device)
        labels = torch.tensor(labels).to(device)

        _, probs = model(contexts,
                         contexts_lengths,
                         responses,
                         responses_lengths)

        matching_scores = probs[:, 0] - probs[:, 1]