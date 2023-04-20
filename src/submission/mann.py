import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True, num_layers=1)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True, num_layers=1)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
        B = input_images.size()[0]
        image = input_images.view(B, self.num_classes * self.samples_per_class, 784) # image = [B, (K+1)*N, 784]
        label = input_labels.view(B, self.num_classes * self.samples_per_class, self.num_classes) # label = [B, (K+1)*N, N]
        layer1_inp = torch.cat((image, label), -1) # inp = [B, (K+1)*N, 784+N]
        layer1_inp[:, -self.num_classes:, -self.num_classes:] = 0

        layer1_out, _ = self.layer1(layer1_inp) # layer1_out = [B, (K+1)*N, h]
        layer2_out, _ = self.layer2(layer1_out) # layer2_out = [B, (K+1)*N, N]
        out = layer2_out.view(B, self.samples_per_class, self.num_classes, self.num_classes) # out = [B, K+1, N, N]
        return out
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        B = preds.size()[0]
        preds = preds[:, -1, :, :].reshape(-1, self.num_classes) # preds = [B*N, N]
        labels = labels[:, -1, :, :].reshape(-1, self.num_classes) # labels = [B*N, N]
        # labels = labels.view(B, self.num_classes, self.num_classes, self.samples_per_class)
        loss = F.cross_entropy(preds, labels)
        ### END CODE HERE ###

        return loss
