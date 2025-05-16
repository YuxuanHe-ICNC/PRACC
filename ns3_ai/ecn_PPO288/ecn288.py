
from py_interface import *
from ctypes import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from Buffer import ReplayBuffer
from Agent import PPO
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from train_PPO import train_offline_PPO
import json


parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')
parser.add_argument('--rl_action', action='store_true',
                    help='whether use rl action')
parser.add_argument('--save', action='store_true',
                    help='whether save PPO model')
parser.add_argument('--load', action='store_true',
                    help='load model')



class sEcnRlEnv(Structure):
    _pack_ = 1
    NUM_ELEMENTS = 432 # 6*12*2+288 = 432
    _fields_ = [
        ('egressqlen', c_uint32 * NUM_ELEMENTS), 
        ('linkrate', c_double * NUM_ELEMENTS),  
        ('ecnlinkrate', c_double * NUM_ELEMENTS),
        ('ecnmin', c_uint32 * NUM_ELEMENTS),
        ('ecnmax', c_uint32 * NUM_ELEMENTS),
        ('ecnpmax', c_double *NUM_ELEMENTS),
        ('envType', c_uint8),          
        ('simTime_us', c_int64)        
    ]
class EcnRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('newecnmin', c_uint32*sEcnRlEnv.NUM_ELEMENTS),
        ('newecnmax', c_uint32*sEcnRlEnv.NUM_ELEMENTS),
        ('newecnpmax', c_double*sEcnRlEnv.NUM_ELEMENTS)
    ]

if __name__ == "__main__":
    actor_lr = 1e-4  #训练步长
    critic_lr = 1e-3 
    num_episodes = 1000
    gamma = 0.98
    lmbda = 0.95
    eps = 0.2
    switch_num = 18

    target_update = 1000 #1000 #10000 
    buffer_size = 100000
    batch_size = 128  #64 128 256

    history_len = 3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    SPort_num = 432 

    past_state = np.zeros((SPort_num, history_len,6))
    past_action = np.zeros((SPort_num,3))
    state = np.zeros(6)

    replay_buffers = []
    for i in range(switch_num):
        replay_buffer = ReplayBuffer(buffer_size)
        replay_buffers.append(replay_buffer)

    if history_len:
        state_dim = int(history_len*6) # txrate qlen txrate_m min max pmax

    kmax = [128,10240] # [128,10240]
    kmin = [2,10240]# [2,10240] 
    action_dim = 3

    args = parser.parse_args()

    # model_file_name1 = 'Model/Agent1_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    # model_file_name2 = 'Model/Agent2_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    # model_file_name3 = 'Model/Agent3_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    # model_file_name4 = 'Model/Agent4_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    # model_file_name5 = 'Model/Agent5_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    # model_file_name6 = 'Model/Agent6_Sep_18_15_s18_a160.pt'  # 指定模型文件路径
    model_file_name = 'Model/'
    # TRAIN = 'T'
    TRAIN = 'F ' # T is Training 

    agents = []
    if args.use_rl:
        for i in range(switch_num):
            agent = PPO(state_dim, action_dim, actor_lr,critic_lr, gamma, lmbda,eps,
                target_update, device,kmax,kmin,model_file_name,replay_buffers[i],batch_size,i,TRAIN)
            agents.append(agent)
        
       
    if args.load:
        for i in range(switch_num):
            agents[i].load('actor_9000.pt','critic_9000.pt')

    # mempool_key = 1234
    # mem_size = 4096
    mempool_key = 1345          # memory pool key, arbitrary integer large than 1000
    mem_size = 18*1024*1024            # memory pool size in bytes
    Init(mempool_key, mem_size) # 1234为

    var = Ns3AIRL(2955, sEcnRlEnv, EcnRlAct)        #2955
    exp = Experiment(mempool_key, mem_size, 'ecn_n ecnmix/configfile/config.txt', '../../')
    exp.run(show_output=0)
    exp.reset()
    
    # replay_buffer2 = ReplayBuffer(buffer_size)
    # replay_buffer3 = ReplayBuffer(buffer_size)
    # replay_buffer4 = ReplayBuffer(buffer_size)
    # replay_buffer5 = ReplayBuffer(buffer_size)
    # replay_buffer6 = ReplayBuffer(buffer_size)

    train_offline_PPO(var, num_episodes,agents,
                      past_state,past_action,state,history_len,args)
    var.close()    
    if args.save:
        for i in range(switch_num):
            agents[i].save() 
           


