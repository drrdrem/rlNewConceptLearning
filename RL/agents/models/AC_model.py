import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, V=149850, embedding_size=128,  action_size=10, seed=0):
        """Initialize neural network model 

        Arguments:
            V (int): number of Vocabularies
            embedding_size (int): embedding dimensions
            action_size (int): Number of actions
        """
        super(PolicyNetwork, self).__init__()
        torch.manual_seed(seed=seed)

        V = V # number of Vocabularies
        D = embedding_size # embedding dimensions
        C = action_size # number of Classes
        Ci = 1
        Co = 6 # number kernels
        Ks = [3,4,5,6] # kernels size

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*len(Ks)*Co, C)
        self.value_head = nn.Linear(2*len(Ks)*Co, 1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (B, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, pre_hidden):
        x = self.embed(x)  # (B, W, D)
        x = x.unsqueeze(1)  # (B, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(B, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(B, Co), ...]*len(Ks)
        x = torch.cat(x, 1) # (B,len(Ks)*Co)

        hidden = self.dropout(x)  # (B, len(Ks)*Co)
        
        # hidden = nn.LayerNorm(5, elementwise_affine = False)(hidden.unsqueeze(0))
        x = torch.cat((hidden, pre_hidden), 1) # (B, 2*len(Ks)*Co)
        x = F.relu(x)
        logit = self.fc1(x)  # (B, C)
        state_values = self.value_head(x) # (B, 1)
        return logit, hidden, state_values