"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the contexts and
                responses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            # self._rnn_dropout = RNNDropout(p=self.dropout)
            self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size, self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                contexts,
                contexts_lengths,
                responses,
                responses_lengths):
        """
        Args:
            contexts: A batch of varaible length sequences of word indices
                representing contexts. The batch is assumed to be of size
                (batch, contexts_length).
            contexts_lengths: A 1D tensor containing the lengths of the
                contexts in 'contexts'.
            responses: A batch of varaible length sequences of word indices
                representing responses. The batch is assumed to be of size
                (batch, responses_length).
            responses_lengths: A 1D tensor containing the lengths of the
                responses in 'responses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        contexts_mask = get_mask(contexts, contexts_lengths).to(self.device)
        responses_mask = get_mask(responses, responses_lengths).to(self.device)

        embedded_contexts = self._word_embedding(contexts)
        embedded_responses = self._word_embedding(responses)

        if self.dropout:
            embedded_contexts = self._rnn_dropout(embedded_contexts)
            embedded_responses = self._rnn_dropout(embedded_responses)

        encoded_contexts = self._encoding(embedded_contexts, contexts_lengths)
        encoded_responses = self._encoding(embedded_responses, responses_lengths)

        attended_contexts, attended_responses = self._attention(encoded_contexts, contexts_mask,
                                                                encoded_responses, responses_mask)

        enhanced_contexts = torch.cat([encoded_contexts,
                                       attended_contexts,
                                       encoded_contexts - attended_contexts,
                                       encoded_contexts * attended_contexts],
                                      dim=-1)
        enhanced_responses = torch.cat([encoded_responses,
                                        attended_responses,
                                        encoded_responses -
                                        attended_responses,
                                        encoded_responses *
                                        attended_responses],
                                       dim=-1)

        projected_contexts = self._projection(enhanced_contexts)
        projected_responses = self._projection(enhanced_responses)

        if self.dropout:
            projected_contexts = self._rnn_dropout(projected_contexts)
            projected_responses = self._rnn_dropout(projected_responses)

        v_ai = self._composition(projected_contexts, contexts_lengths)
        v_bj = self._composition(projected_responses, responses_lengths)

        v_a_avg = torch.sum(v_ai * contexts_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(contexts_mask, dim=1,
                                                                                                  keepdim=True)
        v_b_avg = torch.sum(v_bj * responses_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(responses_mask,
                                                                                                   dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, contexts_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, responses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
