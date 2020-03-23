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
    nll_loss = -input.gather(dim=-1, index=target)
    smooth_loss = -input.sum(dim=-1, keepdim=True)
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
    eps_i = epsilon / input.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
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
        return loss, nll_loss


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