from py_interface import *
from ctypes import *
import numpy as np
import time
from Buffer import ReplayBuffer
from Agent import PPO
import argparse
import os
from datetime import datetime
from tqdm import tqdm

def train_offline_PPO(var,num_episodes,agents,past_state,past_action,state,history_len,args):
    # normalize filter
    iter = 0
    while not var.isFinish():
        progress_bar = tqdm(total=int(num_episodes / 10))
        for i in range(int(num_episodes / 10)):
            iter = iter + 1
            progress_bar.update(1)
            print("wait for shared memory")
            with var as data:
                print("successful read shared memory")
                if not data:
                    break
                sim_time_ns = data.env.simTime_us
                if args.rl_action:
                     # 遍历每个端口
                    for index in range(432):
                        #判断当前端口属于哪个agent
                        if index < 360:  # 前360个端口属于叶子交换机(number:12 ,each port: 24+6
                            group = index // 30
                        else:  # 后72个端口属于脊交换机(number;6,each port:12)
                            group = 12 + (index - 360)//12
                        
                        egress_qlen = data.env.egressqlen[index]
                        link_rate = data.env.linkrate[index]
                        ecn_link_rate = data.env.ecnlinkrate[index]
                        ecn_min = data.env.ecnmin[index]
                        ecn_max = data.env.ecnmax[index]
                        ecn_pmax = data.env.ecnpmax[index]

                        # using past two state and current state as network input
                            # normalized egress_qlen to [0,1],Dl is a decreasing function 
                        k = 0
                        while 20000 * 2**k <= egress_qlen:
                            k += 1
                        if k <= 10:
                            Dl = (1-k/10)
                        else:
                            Dl = 0

                        # 234 action space
                        state = np.array([Dl, link_rate, ecn_link_rate, ecn_min/640000, ecn_max/2048000, ecn_pmax/100])

                        # 120 action space
                        # state = [Dl, link_rate, ecn_link_rate, ecn_min/32000, ecn_max/256000, ecn_pmax/100]

                        # 160 ACTION SPACE
                        #state = [Dl, link_rate, ecn_link_rate, ecn_min/320000, ecn_max/1280000, ecn_pmax/100]
                        # w_1 = 0.3 + 0.2*np.exp(5*(Dl-0.5))
                        # reward = Dl*w_1 +link_rate * (1-w_1) # 
                        reward = Dl*0.3  + link_rate * 0.7 

                        state_pair = np.concatenate((past_state[index][1], past_state[index][2], state),axis=0)
                        next_state = state_pair.reshape(3,6)

                        #each switch can be seen as an agent ,so there are 6 agent 
                        agent = agents[group]
  
                        #print("get action")
                        action = agent.take_action(state_pair)
                        data.act.newecnmin[index] = c_uint32(action[0])
                        data.act.newecnmax[index] = c_uint32(action[1])
                        data.act.newecnpmax[index] = c_double(action[2])

                        if agent.train == 'T':#Train 
                            agent.store_transition(past_state[index],past_action[index],reward,next_state)
                            # replay_buffer.add(last_state_pair,last_action[index],reward,state_pair)
                            #     # print(last_state_pair,last_action[i],reward,state_pair)
                            # if replay_buffer.size() > minimal_size:
                            #     b_s, b_a, b_r, b_ns = replay_buffer.sample(
                            #         batch_size)
                            #     #print(b_s, b_a, b_r, b_ns)
                            #     transition_dict = {
                            #         'states': b_s,
                            #         'actions': b_a,
                            #         'next_states': b_ns,
                            #         'rewards': b_r
                            #     }
                            if agent.step % (num_episodes / 10) == 0:#model update every (num_episodes / 10) step 
                                print('Training and Updating model......')
                                policy_loss,value_loss = agent.update()
                                print('finish ',(num_episodes/10),', loss: ',policy_loss,'\t',value_loss,'\t','R: ',reward,'\t','eps:',agent.eps)
                            if iter%1000 == 0: 
                                agent.save(iter)
                        else: #Testing  Model
                            print('Testing  Model......')
                            print(index,'\t',agent.id,'\t',action)

                        for j in range(len(state)):
                            for i in range(history_len-1):
                                past_state[index][i][j] = past_state[index][i+1][j]
                            past_state[index][history_len-1][j] = state[j]
                        past_action[index] = action

                else:
                    sim_time_ns = data.env.simTime_us
                    #print(sim_time_ns)
                    for i in range(432):
                        egress_qlen = data.env.egressqlen[i]
                        link_rate = data.env.linkrate[i]
                        ecn_link_rate = data.env.ecnlinkrate[i]
                        ecn_min = data.env.ecnmin[i]
                        ecn_max = data.env.ecnmax[i]
                        ecn_pmax = data.env.ecnpmax[i]
                        #print(egress_qlen,link_rate,ecn_link_rate,ecn_min,ecn_max,ecn_pmax)
                        

