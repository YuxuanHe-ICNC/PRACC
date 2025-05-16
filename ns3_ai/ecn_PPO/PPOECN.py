
from py_interface import *
from ctypes import *
import torch
import numpy as np
import math
from Buffer import ReplayBuffer
from Agent import PPO
from plot import PlotResult
import argparse
from Action_List import Action,Action1,Action2,Action3
import os
from train_PPO import train_online_PPO
from datetime import datetime
import time

'''
    命令行参数设置
'''
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
                    help='load trained model')

# 环境数据传输类
class EcnRlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('egressqlen', c_uint32 ), 
        ('linkrate', c_double ),  
        ('ecnlinkrate', c_double ),
        ('ecnmin', c_uint32 ),
        ('ecnmax', c_uint32 ),
        ('ecnpmax', c_double ),
        ('envType', c_uint8),          
        ('simTime_us', c_int64)        
    ]
# 动作数据传输类
class EcnRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('newecnmin', c_uint32),
        ('newecnmax', c_uint32),
        ('newecnpmax', c_double)
    ]

class TimeoutError(Exception):
    pass

def wait_for_ns3(var, timeout=5.0):# 
    start_time = time.time()
    while time.time() - start_time < timeout:
        if var.IsEnabled():
            return True
        time.sleep(0.1)
    raise TimeoutError("NS3 not responding")

if __name__ == "__main__":
    '''
    PPO参数初始化
    '''
    actor_lr = 1e-4  #训练步长
    critic_lr = 1e-3 
    num_episodes = 500
    gamma = 0.98
    lmbda = 0.9
    eps = 0.2

    '''经验回放区参数初始化'''
    target_update = 1000  #10000 
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64  #64 128 256
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    print("using "+str(device)+" to train")
    print(device)
    '''state设置'''
    history_len = 3 # 历史信息条数
    past_state = np.zeros((history_len, 6))
    SPort_num = 1
    state = np.zeros(6)
    past_action = np.zeros(3)

    if history_len:
        state_dim = int((history_len)*6) # txrate qlen txrate_m min max pmax

    '''更改action维数'''
    kmax = [128,10240] # [128,10240]
    kmin = [2,10240]# [2,10240] 
    action_dim = 3

    '''读取命令行指令'''
    args = parser.parse_args()

    '''初始化ns3-ECN环境'''
    mempool_key = 1234          # memory pool key, arbitrary integer large than 1000
    mem_size = 4096             # memory pool size in bytes
    Init(mempool_key, mem_size) # 1234为
    # import from py_interface.py in Experiment
    # sharedmemorykey= 1234  sharedmemorysize = 4096
    var = Ns3AIRL(3230, EcnRlEnv, EcnRlAct)  
    # index = 3230，需要和ns3初始化一致,运行文件夹+配置文件，存储位置   
    exp = Experiment(mempool_key, mem_size, 'ecn_PPO ecnmix/configfile/config.txt', '../../')
    exp.run(show_output=0)
    exp.reset()
    
    replay_buffer = ReplayBuffer(buffer_size)
    train_num = 1
    iter_num = 9000

    Load_model_file = 'Model/model1/'  # 指定模型文件路径
    Actor_path = 'actor_'+str(iter_num)+'.pt'
    Critic_path = 'critic_'+str(iter_num)+'.pt'

    TRAIN = 'F' # index:T is training ,F is testing
    if args.use_rl:
        agent = PPO(state_dim, action_dim,actor_lr, critic_lr,gamma,lmbda,eps,
            target_update, device,kmax,kmin,Load_model_file,replay_buffer,batch_size,TRAIN)
    if args.load:
            agent.load(Actor_path,Critic_path)

    print("start train PPO")
    return_list = []
    DL_list = []
    linkrate_list = []
    actionout = []
    egress = []
    train_online_PPO(var,num_episodes,agent,replay_buffer,
                     past_state,past_action,state,history_len,minimal_size,
                     batch_size,args,return_list,DL_list,linkrate_list,
                     actionout,egress,iter_num)
    
    '''
    训练结果输出和可视化
    '''
    print("end training")
 
    PlotResult(return_list,DL_list,linkrate_list,action_dim,egress)
   
 
    
    # 使用with语句打开文件，并确保文件正确关闭
   
  
    '''
    egress输出
    '''
    
    # train_num = 2

    # if TRAIN == 'T':
    #     egress_file = 'result/egress_'+str(train_num)+'.txt'
    #     DL_file = 'result/DL_'+str(train_num)+'.txt'
    #     link_file = 'result/link_'+str(train_num)+'.txt'
    #     reward_file = 'result/reward_'+str(train_num)+'.txt'
    #     action_file = 'result/action_'+str(train_num)+'.txt'

    #     # 使用with语句打开文件，并确保文件正确关闭
    #     with open(egress_file, 'w') as file:
    #         # 遍历列表，将每个元素写入文件
    #         for item in egress:
    #             file.write(str(item) + '\n')
        
    #     with open(DL_file, 'w') as file:
    #         # 遍历列表，将每个元素写入文件
    #         for item in DL_list:
    #             file.write(str(item) + '\n')

    #     with open(link_file, 'w') as file:
    #         # 遍历列表，将每个元素写入文件
    #         for item in linkrate_list:
    #             file.write(str(item) + '\n')

    #     with open(reward_file, 'w') as file:
    #         # 遍历列表，将每个元素写入文件
    #         for item in return_list:
    #             file.write(str(item) + '\n')
    #     with open(action_file, 'w') as file:
    #         # 遍历列表，将每个元素写入文件
    #         for item in actionout:
    #             file.write(str(item) + '\n')

    if args.save:
        agent.save(99) 
    



