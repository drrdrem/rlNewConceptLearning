import numpy as np
import pandas as pd
from RL.recom_papers import *
import heapq

class env(object):
    def __init__(self, papers_recommendation, k = 20, precom_sim = None):
        """The environment.

        Arguments:
            papers_recommendation (object): a recommendation instance.
            precom_sim (pd.dataframe): pre compute similarities (copus1, copus2, similarities).
        """
        self.recommendation = papers_recommendation
        self.k = k
        self.precom_sim = precom_sim
        self.com_set_fea = self.recommendation.com_set_fea if self.recommendation.com_set_fea else None
        
        self.current_paper = None
        self.current_paper_fea = None
        self.current_paper_lab  = None
        self.action_space = []
        self.sim_score = []

        self.visited = []
        # self.visited_bool = {file: False for file in self.recommendation.all_papers_fea['file']}

        self.done = False
        self.mask = False

        self.state_info = {}
        self.state_info['current_paper'] = self.current_paper
        self.state_info['reward'] = 0.
        self.state_info['next_paper'] = None
        self.state_info['done'] = False

    def get_state(self):
        """Get the current state (paper).

        returns:
            The current paper id.
        """
        return self.current_paper

    def jaccard_idx(self, a, b):
        c = a.intersection(b)
        return 1. - (float(len(c)) / (len(a) + len(b) - len(c)))

    def get_actions(self):
        """Retrieve the k nearest papers.
           The is a variant of k nearest neighbors.

        returns:
            neighbors: the nearest top k papers' ids
        """
        distances = []
        for paper_id in self.recommendation.get_all_papers_ids():
            if paper_id != self.current_paper and paper_id not in self.visited:
                if self.com_set_fea:
                    if paper_id in self.com_set_fea and self.current_paper in self.com_set_fea:
                        dist = self.jaccard_idx(self.com_set_fea[paper_id], self.com_set_fea[self.current_paper])
                    else:
                        dist = 1.
                else:
                    paper_fea = self.recommendation.get_paper_fea(paper_id)
                    dist = self.recommendation.sim(self.current_paper_fea, paper_fea)
                distances.append((paper_id, dist))

        n = len(distances)
        if n<self.k: 
            self.mask = True
            distances.sort(key=lambda tup: tup[1])
            neighbors = [dist[0] for dist in distances]
            self.sim_score = [dist[1] for dist in distances]
            neighbors += [None]*(self.k-n)
        else:
            distances = heapq.nsmallest(self.k, distances, key=lambda tup: tup[1])
            neighbors = [dist[0] for dist in distances]
            self.sim_score = [dist[1] for dist in distances]
        
        self.action_space = neighbors

    def get_actions_precom(self):
        """Retrieve the k nearest papers wit pre-compute simiarities.
           The is a variant of k nearest neighbors.

        returns:
            neighbors: the nearest top k papers' ids
        """
        df = self.precom_sim[(self.precom_sim['corpus1']==self.current_paper) | (self.precom_sim['corpus2']==self.current_paper)]
        df_tmp = np.where(df['corpus1']==self.current_paper, df['corpus2'], df['corpus1'])
        df_tmp = pd.DataFrame(df_tmp, columns=['neighbors'])
        df_tmp['similarity'] = df['similarity']
        df_tmp = df_tmp.nlargest(self.k, 'similarity')

        neighbors = list(df_tmp['neighbors'])
        n = len(neighbors)
        if n<self.k: 
            self.mask = True
            neighbors += [None]*(self.k-n)

        self.action_space = neighbors

    def get_action(self, action_id):
        """Get the next paper to be read.
        Arguments:
            action_id (int): an index of paper in action space.

        returns:
            Next paper id.
        """
        return self.action_space[action_id]

    def is_valid_action(self, action_id):
        """Check if a valid action.

        returns:
            (bool): if a valid action.
        """
        if self.action_space[action_id] in self.visited: return False
        elif self.action_space[action_id]==None: return False
        return True

    def get_reward(self):
        """Get the reward.

        returns:
            The reward in the state.
        """
        reward = (self.current_paper_lab == 1)
        return 1/(len(self.visited)+1) if reward else -.3

    def init_step(self, initial_paper_id):
        """Initialization of the first state.

        Arguments:
            initial_paper_id (str): an initial paper to be read

        returns:
            state_info (tuple): state, reward, done
        """
        self.current_paper = initial_paper_id
        self.current_paper_fea = self.recommendation.get_paper_fea(self.current_paper)
        self.current_paper_lab  = self.recommendation.get_paper_lab(self.current_paper)

        self.state_info['current_paper'] = self.current_paper
        self.state_info['reward'] = self.get_reward()
        self.state_info['next_paper'] = None
        if self.current_paper_lab==1:
            self.done = True
        else:
            if self.precom_sim: self.get_actions_precom()
            else: self.get_actions()
        self.state_info['done'] = self.done

        return self.current_paper, self.state_info['reward'], self.done

    def step(self, action_id):
        """A step of feadback from envornment.

        Arguments:
            action_id (int): an index of paper in action space.

        returns:
            state_info (tuple): state, reward, done
        """
        self.state_info['current_paper'] = self.get_state()
        self.visited.append(self.current_paper)

        # self.get_actions()
        self.state_info['next_paper'] = self.get_action(action_id)
        self.current_paper = self.state_info['next_paper']
        self.current_paper_fea = self.recommendation.get_paper_fea(self.current_paper)
        self.current_paper_lab  = self.recommendation.get_paper_lab(self.current_paper)
        if self.current_paper_lab==1:
            self.done = True
        else:
            if self.precom_sim: self.get_actions_precom()
            else: self.get_actions()
        self.state_info['done'] = self.done
        self.state_info['reward'] = self.get_reward()

        return self.current_paper, self.state_info['reward'], self.done