# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import functional as F
# from transformer import Constants


def label_smoothed_nll_loss(input, target, epsilon, ignore_index=None, reduce=True):
    """
    new_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes
    """
    nll_loss = -input.gather(dim=-1, index=target)
    smooth_loss = -input.mean(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    loss = (1. - epsilon) * nll_loss + epsilon * smooth_loss
    return loss, nll_loss


class LabelSmoothedCrossEntropy(nn.Module):

    def __init__(self, ignore_index, label_smoothing=0.):
        """

        :param label_smoothing: epsilon for label smoothing, 0 means no label smoothing
        """
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        """ LabelSmoothedCrossEntropy """
        input = F.log_softmax(input, dim=-1)
        target = target.view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            input, target, self.label_smoothing, ignore_index=self.ignore_index
        )
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


if __name__ == '__main__':

    loss = nn.CrossEntropyLoss(ignore_index=0)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    print(input)
    print(target)
    print(output)
    output.backward()
    print(output)

    loss2 = LabelSmoothedCrossEntropy(ignore_index=0, label_smoothing=0.1)
    print(input)
    print(target)
    print(loss2(input, target))