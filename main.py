import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from gym.wrappers import Monitor
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from cartpole import CartPoleAI
from acrobot import AcroBotAI
import argparse 

def init_weights(m):
    
        # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
        # nn.Conv2d bias is of shape [16] i.e. # number of filters
        
        # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
        # nn.Linear bias is of shape [32] i.e. # number of output features
        
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)
                
def return_random_agents(args, num_agents,game_actions):
    
    agents = []
    if args.env_name == 'cartpole':
        model = CartPoleAI(game_actions)
    if args.env_name == 'acrobot':
        model = AcroBotAI(game_actions)
    for _ in range(num_agents):
        
        agent = model
        
        for param in agent.parameters():
            param.requires_grad = False
            
        init_weights(agent)
        agents.append(agent)
        
        
    return agents
    
def run_agents(agents, env):
    
    reward_agents = []
    #env = gym.make("Acrobot-v1")
    
    for agent in agents:
        agent.eval()
    
        observation = env.reset()
        
        r=0
        s=0
        
        for _ in range(250):
            
            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            new_observation, reward, done, info = env.step(action)
            r=r+reward
            
            s=s+1
            observation = new_observation

            if(done):
                break

        reward_agents.append(r)        
        #reward_agents.append(s)
        
    
    return reward_agents
def return_average_score(env, agent, runs):
    score = 0.
    for i in range(runs):
        score += run_agents([agent], env)[0]
    return score/runs
def run_agents_n_times(env, agents, runs):
    avg_score = []
    cnt = 1
    for agent in agents:
        #print(cnt)
        #cnt = cnt +1
        
        avg_score.append(return_average_score(env, agent,runs))
    return avg_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="cartpole", help="name of the environment")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    args = parser.parse_args()

    return args



def mutate(agent):

    child_agent = copy.deepcopy(agent)
    
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
            
    for param in child_agent.parameters():
    
        if(len(param.shape)==4): #weights of Conv2D

            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            
                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
                                
                                    

        elif(len(param.shape)==2): #weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    
                    param[i0][i1]+= mutation_power * np.random.randn()
                        

        elif(len(param.shape)==1): #biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                
                param[i0]+=mutation_power * np.random.randn()

    return child_agent
def return_children(env, agents, sorted_parent_indexes, elite_index):
    
    children_agents = []
    
    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agents)-1):
        
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))

    #now add one elite
    elite_child = add_elite(env, agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1 #it is the last one
    
    return children_agents, elite_index
def add_elite(env,agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    
    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])
        
    top_score = None
    top_elite_index = None
    
    for i in candidate_elite_index:
        score = return_average_score(env, agents[i],runs=5)
        print("Score for elite i ", i, " is ", score)
        
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
            
    print("Elite selected with index ",top_elite_index, " and score", top_score)
    
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':

    args = get_args()

    if args.env_name == 'cartpole':
        env = gym.make("CartPole-v1")
        game_actions = 2
    if args.env_name == 'acrobot':
        env = gym.make("Acrobot-v1")
        game_actions = 3


    #game_actions = 3 #2 actions possible: left or right

    #disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize N number of agents
    num_agents = 200
    agents = return_random_agents(args, num_agents, game_actions)

    # How many top agents to consider as parents
    top_limit = 20

    # run evolution until X generations
    generations = 1000

    elite_index = None
    highest = []
    for generation in range(generations):

        # return rewards of agents
        rewards = run_agents_n_times(env, agents, 2) #return average of 3 runs

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        print("")
        print("")
        
        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        #print(rewards)
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)
        highest.append(top_rewards[0])
        
        """if (generation+1) % 5 == 0:
            play_agent(agents[sorted_parent_indexes[0]],(generation+1))"""
            
        
        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(env, agents, sorted_parent_indexes, elite_index)

        # kill all agents, and replace them with their children
        agents = children_agents
        
        
        
        
        """with open(args.env_name +'_pkl', 'wb') as fp:
                #pickle.dump(agents, fp)
                #pickle.dump(elite_index, fp)
                pickle.dump(highest, fp)"""