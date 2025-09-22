import numpy as np

import torch
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import torch.autograd as autograd
from torchtext import data
import torch.nn.functional as F

from RL.agents.models.AC_model import PolicyNetwork

class A_C(object):
    """An agent based on A2C Algorithm.
       Modified from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
        Arguments:
            text_field (torchtext): text field.
            action_size (int): Size of Actions, default 20.
            embedding_size (int): 
            lr (float): learning rate, default 1e-3.
            gamma (float): discount factor, default 0.9.
            seed (int): Random Seed, default 0.
            model_dir: load model directory, default none.
        """
    def __init__(self, text_field, action_size=20, embedding_size=128, 
                lr=1e-3, gamma=.9, seed=0, model_dir=None):
        torch.manual_seed(seed=seed)
        self.action_size = action_size
        self.embedding_size = embedding_size
        
        self.text_field = text_field
        self.V = len(text_field.vocab.stoi)
        
        self.lr = lr
        self.gamma = gamma

        self.model =  PolicyNetwork(V = self.V, embedding_size = self.embedding_size, action_size = self.action_size, seed=seed)
        if model_dir:
            self.model.load_state_dict(torch.load(model_dir))
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
    

    def select_action(self, text, pre_states, train=True):
        """A step of agent acts 
        Arguments:
            prev_action (arr): The selected paper from previous action
            text (str): features text
            pre_states (tuple): pre_state cx, hx fromm RNN output

        Returns:
            Next paper id.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        text = text.split()
        text = [[self.text_field.vocab.stoi[x] for x in text if x in self.text_field.vocab.stoi]]
        
        feature = torch.tensor(text)
        feature = autograd.Variable(feature)
        probs, pre_states, value = self.model(feature, pre_states)       
        
        return probs, pre_states, value

    def learn(self, rewards, log_probs, values, entropies):
        """A step of agent learns 
        Arguments:
            rewards (arr): The selected paper from previous action
            log_probs (arr): features text
            entropies (arr): pre_state cx, hx fromm RNN output
        Returns:
            The total loss in the stage.
        """
        R = 0
        policy_losses = []
        value_losses = [] 
        returns = [] 
        for r in rewards:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        
        return loss.data.item()
    
    def save(self, model_dir):
        """ Save the agent. 
        Arguments:
            model_dir (str): the agent saving directory.
        """
        torch.save(self.model.state_dict(), model_dir+'_model.pkl')