# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    '''
    Calculate Triplet Loss
    Input:
        bases: batch_size * feature_size
        positives: batch_size * feature_size
        negatives: batch_size * feature_size
    Output:
        triplet_loss
    '''

    def __init__(self):
        super(TripletLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, bases, positives, negatives, confidences):
        positive_distance = 1 - self.cos(bases, positives)
        negative_distance = 1 - self.cos(bases, negatives)
        if confidences is None:
            loss = torch.mean(
                torch.clamp(
                    positive_distance - negative_distance + 0.2, min=0))
        else:
            loss = torch.mean(
                torch.clamp(
                    positive_distance - negative_distance + 0.2, min=0) *
                confidences)
        return loss, positive_distance, negative_distance


class TransRegularizer(nn.Module):
    '''
    Trans Feature Regularizer
    '''

    def __init__(self):
        super(TransRegularizer, self).__init__()
        self.identity = torch.eye(64).unsqueeze(0).cuda()

    def forward(self, trans):
        loss = torch.mean(
            torch.norm(
                torch.bmm(trans, trans.transpose(2, 1)) - self.identity,
                dim=(1, 2)))
        return loss
