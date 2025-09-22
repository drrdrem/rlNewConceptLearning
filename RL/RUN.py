from RL.agents.REINFORCE import REINFORCE
from RL.environment import env
from RL.recom_papers.recom_papers import *

from tqdm import tqdm
import time

import random
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import torch.autograd as autograd
from torchtext import data
import torch.nn.functional as F


torch.manual_seed(seed=0)

def run(drugs, agent, text_field, allfea_df, alllab_df, A_C = False, 
        train = False, action_size = 20, weight = False, arg_max = False, 
        hybrid = None, content = None, max_episode = 1, seed = 0, log = False):
    """Run the simulation.
        Arguments:
            drugs (list): an index of paper in action space.
            agent (obj): 
            test_field ():
            allfea_df (pd.DataFrame)
            alllab_df (pd.DataFrame)
            train (bool)
            arg_max (bool)
            hybrid (dict)
            content (str)
            max_episode (int)
            seed (int)

        returns:
            Next paper id.
    """
    random.seed(seed)
    res = {drug: [] for drug in drugs}
    all_losses = []
    state_info = {}
    for drug in drugs:
        # 0. Build a subset of papers based on the query.
        recomm = papers_recommendation(allfea_df, alllab_df, com_set_fea=True)
        recomm.reocmmend(drug)

        print('# drug: {} ###############'.format(drug))
        print('# Total: {} ##############'.format(recomm.length))
        print('##########################')
        if recomm.length-1==0: continue
        start_t = time.time()

        i_episode = 0
        losses = []
        while i_episode < max_episode:

            # 1. Build an environment of the episode.
            episode = env(recomm, k=action_size)

            # 2. Random select an initial paper.
            if not hybrid:
                ini_paper_id = random.choice(recomm.get_all_papers_ids())
                state, rewards, done = episode.init_step(ini_paper_id)
                if done: continue # If the first paper is "the paper", skip

            else:
                ini_paper_id = hybrid[drug]
                state, rewards, done = episode.init_step(ini_paper_id)
                if done: 
                    res[drug].append(1)
                    break


            print('The {}th episode'.format(i_episode))
            print('============================================')
            print(ini_paper_id)


            # 3. Initialization of the history.
            hidden =torch.zeros(1, 24)
            rewards = []
            entropies = []
            log_probs = []
            values = []
            for t in tqdm(range(recomm.length-1)):
                # 4. The observations from environment.
                if log:
                    state_info[t] = {'id': state, 'k': episode.action_space, 'len':episode.sim_score}
                if not content:
                    paper_fea = recomm.get_paper_fea(state)
                else:
                    file_name = state.split('\\')
                    file_name = file_name[-1].split('.')[0]
                    paper_fea = open(content+file_name+'.txt', "r").read()
                text = str(paper_fea)+" "+str(drug)+" gene"

                # 5. The agent select next paper to be read.
                if not A_C:
                    probs, hidden = agent.select_action(text, hidden, train=train)
                else:
                    probs, hidden, value = agent.select_action(text, hidden, train=train)
    #             hidden = hidden.detach()

                if episode.mask:
                    condition = [True if valid else False for valid in episode.action_space]
                    condition = torch.tensor(condition)
                    # probs = probs.where(condition, torch.tensor(0.0))
                    probs = probs[:, [i for i in range(condition.sum())]]

                if weight:
                    weights = torch.FloatTensor(episode.sim_score)
                    probs = torch.mul(weights, probs)
                    
                probs = F.softmax(probs, dim=1)
                probs_samp = probs.clone().detach()
                # print(probs_samp)

                if train:
                    action = probs_samp.multinomial(num_samples=1).data
                    action_id = action[0].numpy()[0]
                    prob = probs[:, action[0,0]].view(1, -1)
                    log_prob = prob.log()
                    entropy = - (probs*probs.log()).sum()
                    
                else:
                    if arg_max:
                        action_id = np.argmax(probs_samp.data.numpy())
                    else:
                        action = probs_samp.multinomial(num_samples=1).data
                        action_id = action[0].numpy()[0]
                        # print(action_id)

                # 6. The environment gives feadback
                nxt_state, reward, done = episode.step(action_id)
                
                if train:
                    entropies.append(entropy)
                    log_probs.append(log_prob)
                    if A_C:
                        values.append(value)
                rewards.append(reward)

                state = nxt_state

                # 7. If done, break the episode
                if done: break
            if log:
                state_info[t+1] = {'id': state, 'k': episode.action_space, 'len':episode.sim_score}
            end_t = time.time()-start_t
            i_episode+=1

            print("# of papers to read: ",t+2," Time elapse (min): ", end_t/60, end=' ')
            if train:
                if not A_C:
                    loss = agent.learn(rewards, log_probs, entropies)
                else:
                    loss = agent.learn(rewards, log_probs, values, entropies)
                losses.append(loss)
                print('Loss: {}'.format(loss))
            print('============================================')
            res[drug].append(t+2)

            
        if train: 
            all_losses.append(losses)
            
    if train: 
        return all_losses
    if log:
        return state_info 
    return res