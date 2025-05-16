from py_interface import *
from ctypes import *
import numpy as np
import time
from Buffer import ReplayBuffer
from Agent import PPO
from plot import PlotResult
import argparse
from Action_List import Action,Action1,Action2,Action3
import os
from datetime import datetime
from tqdm import tqdm



def train_online_PPO(var,num_episodes,agent,replay_buffer,past_state,
                     past_action,state,message_slot,minimal_size,batch_size,args,
                     return_list,DL_list,linkrate_list,actionout,egress,iter_num):
    # normalize filter 
    
    iter = 0
    while not var.isFinish() :
        progress_bar = tqdm(total=int(num_episodes / 10))
        for i in range(int(num_episodes / 10)):
            print("\nthis is the",iter,"iteration")
            iter = iter+1
            progress_bar.update(1)
            with var as data:
                if data == None:
                    print("break")
                    break
                sim_time_ns = data.env.simTime_us
                print("read MEMORY")
                if args.rl_action:    
                    egress_qlen = data.env.egressqlen
                    link_rate = data.env.linkrate
                    ecn_link_rate = data.env.ecnlinkrate
                    ecn_min = data.env.ecnmin
                    ecn_max = data.env.ecnmax
                    ecn_pmax = data.env.ecnpmax
                    print("data:")
                    print(egress_qlen,link_rate,ecn_min,ecn_max,ecn_pmax)
                    
                    # using past two state and current state as network input
                        # normalized egress_qlen to [0,1],Dl is a decreasing function
                    egress.append(egress_qlen)
                    k = 0
                    while 20000 * 2**k <= egress_qlen:
                        k += 1
                    if k <= 10:
                        Dl = (1-k/10)
                    else:
                        Dl = 0

                    state = np.array([Dl, link_rate, ecn_link_rate, ecn_min/640000, ecn_max/2048000, ecn_pmax/100])

                    w_1 = 0.3 + 0.2*np.exp(5*(Dl-0.5))
                    reward = Dl*w_1 +link_rate * (1-w_1) # 
                    #reward = Dl*0.3+link_rate*0.7
                    return_list.append(reward)
                    DL_list.append(Dl)
                    linkrate_list.append(link_rate)
                    #current state
                    state_pair = np.concatenate((past_state[1], past_state[2], state),axis=0)
                    next_state = state_pair.reshape(3,6)
                    # print("state = ",state.shape,past_state[1].shape)
                    print("next state shape = ",next_state.shape," state shape = ",past_state.shape)

                    #each switch can be seen as an agent ,so there are 6 agent 
                    print("get action")
                    action = agent.take_action(state_pair)
                    print(action)
                    actionout.append(action)
                   
                    data.act.newecnmin = c_uint32(action[0])
                    data.act.newecnmax = c_uint32(action[1])
                    data.act.newecnpmax = c_double(action[2])
                    # write action
                    # with open(file_name, 'a') as f:
                    #     f.write(f"{act_list[action][0]} {act_list[action][1]} {act_list[action][2]}\n")
                    # data.act.newecnmin= act_list[action][0]
                    # data.act.newecnmax = act_list[action][1]
                    # data.act.newecnpmax = act_list[action][2]
                   
                    

                    if agent.train == 'T':#Train 
                        print("start training")
                        agent.store_transition(past_state,past_action,reward,next_state)
                        # replay_buffer.add(past_state,past_action,reward,next_state)
                        #     # print(last_state_pair,last_action[i],reward,state_pair)
                        # if replay_buffer.size() > minimal_size:
                        #     b_s, b_a, b_r, b_ns = replay_buffer.sample(
                        #         batch_size)
                            #print(b_s, b_a, b_r, b_ns)for_test
                            # transition_dict = {
                            #     'states': b_s,
                            #     'actions': b_a,
                            #     'next_states': b_ns,
                            #     'rewards': b_r
                            # }
                        if agent.step % (num_episodes / 10) == 0:#model update every (num_episodes / 10) step 
                            print('Training and Updating model......')
                            policy_loss,value_loss = agent.update()
                            print('finish ',(num_episodes/10),', loss: ',policy_loss,'\t',value_loss,'\t','R: ',reward,'\t','eps:',agent.eps)
                        if iter%10000 == 0:
                            agent.save(iter+iter_num)
                    else: #Testing  Model
                        print('Testing  Model......')
                        print('\t',action)

                    for j in range(len(state)):
                        for i in range(message_slot-1):
                            past_state[i][j] = past_state[i+1][j]
                        past_state[message_slot-1][j] = state[j]
                    past_action = action

                else:
                    sim_time_ns = data.env.simTime_us
                    #print(sim_time_ns)
                    egress_qlen = data.env.egressqlen
                    link_rate = data.env.linkrate
                    ecn_link_rate = data.env.ecnlinkrate
                    ecn_min = data.env.ecnmin
                    ecn_max = data.env.ecnmax
                    ecn_pmax = data.env.ecnpmax
                        #print(egress_qlen,link_rate,ecn_link_rate,ecn_min,ecn_max,ecn_pmax)