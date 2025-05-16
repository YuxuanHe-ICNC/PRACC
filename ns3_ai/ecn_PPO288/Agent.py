import torch
from Net import PolicyNet,ValueNet
import  numpy as np
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
from Buffer import ReplayBuffer
from ctypes import *
from py_interface import *
import random
from torch.distributions import Normal
from collections import namedtuple

def constraint(kmax_idx,kmin_idx):
    if kmax_idx < kmin_idx:
        return kmin_idx,kmax_idx
    return kmax_idx,kmin_idx

def map_to_range(x, original_range):    
    min_original, max_original = original_range

    return ((x - min_original) * 2) / (max_original - min_original) - 1

class PPO:
    ''' PPO算法 ,全部采用policy&value'''
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr, 
                 critic_lr,
                 gamma,
                 lmbda,
                 eps,
                 target_update,
                 device,
                 Kmax,
                 Kmin,
                 model_file,replaybuffer,batch_size,agent_id,train = 'T'):
        self.state_dim = state_dim    
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.id = agent_id
        self.Kmax = Kmax
        self.Kmin = Kmin
        self.device = device
        self.policy = PolicyNet(self.state_dim).to(self.device)
        self.value = ValueNet(self.state_dim).to(self.device)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(),lr=self.actor_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.critic_lr)
                    
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        

        self.target_update = target_update
        self.device = device
        self.model_file = model_file
        self.step = 0
        self.replay_buffer = replaybuffer
        self.batch_size = batch_size
        self.train = train


    def take_action(self, state):
        if self.train == 'T' and np.random.random() < self.eps:# train
                kmax_idx = random.randint(self.Kmax[0], self.Kmax[1])
                kmin_idx = random.randint(self.Kmin[0], self.Kmin[1])
                kmax_idx,kmin_idx = constraint(kmax_idx,kmin_idx)
            # 随机选择连续动作
                pmax_action = random.uniform(0.0, 1.0)
                action = [kmin_idx,kmax_idx,pmax_action]
                #随机动作选择  
        else:
                #网络输出动作选择
                state = torch.from_numpy(state).float().to(self.device)
                #print("take action state_shape",state.shape)
                #state = torch.tensor([state], dtype=torch.float).to(self.device)
        
                action_info = self.remapping(self.policy(state))

                kmax_mu, kmax_std = action_info["kmax"]
                kmin_mu, kmin_std = action_info["kmin"]
                pmax_mu, pmax_std = action_info["pmax"]
                
                kmax_dist = Normal(kmax_mu, kmax_std)
                kmin_dist = Normal(kmin_mu, kmin_std)
                pmax_dist = Normal(pmax_mu, pmax_std)
                
                kmax_action = kmax_dist.sample().item()
                kmin_action = kmin_dist.sample().item()
                pmax_action = pmax_dist.sample().clamp(0.0, 1.0).item()
                
                # 取整操作
                kmax_action = int(round(kmax_action))
                kmin_action = int(round(kmin_action))
                
                # 确保 kmax 和 kmin 在合法范围内
                kmax_action = max(self.Kmax[0], min(self.Kmax[1], kmax_action))
                kmin_action = max(self.Kmin[0], min(self.Kmin[1], kmin_action))
                kmax_idx,kmin_idx = constraint(kmax_action,kmin_action)
                action =[kmin_idx,kmax_idx,pmax_action]
                #print("type:",type(action))
        self.step  = self.step + 1 
        return action
    
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return 0,0
        else:
            state, action, reward, next_states = self.replay_buffer.sample(self.batch_size)
            transition_dict = {
                'states': state,
                'actions': action,
                'rewards': reward,
                'next_states': next_states,
            }
            states = torch.tensor(transition_dict['states'],
                                dtype=torch.float).to(self.device)
            states = states.view(states.size(0), -1)
            actions = torch.tensor(transition_dict['actions'],dtype= torch.long).to(
                self.device)
            #print("actions shape: ",actions.shape)
            rewards = torch.tensor(transition_dict['rewards'],
                                dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'],
                                    dtype=torch.float).to(self.device)
            next_states = next_states.view(next_states.size(0), -1)
            # print("state shape = ",states.shape) #for_test
            #print("next state  = ",next_states)
            with torch.no_grad():
                td_target = rewards + self.gamma * self.value(next_states)
            
            td_errors = td_target - self.value(states)
            advantages = self.compute_gae(td_errors, self.gamma, self.lmbda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 归一化
            #print("advantages: ",advantages)
            
            # 获取旧策略的对数概率
            policy_output = self.policy(states)
            #print(policy_output)
            kmax_dist = torch.distributions.Normal(policy_output["kmax"][0], policy_output["kmax"][1])
            kmin_dist = torch.distributions.Normal(policy_output["kmin"][0], policy_output["kmin"][1])
            pmax_dist = torch.distributions.Normal(policy_output["pmax"][0], policy_output["pmax"][1])

            # 假设 actions 是一个多维向量，分别对应 kmax, kmin, pmax
            kmin_actions = map_to_range(actions[:, 0].unsqueeze(-1),self.Kmin)   # 第一维是 kmax
            kmax_actions = map_to_range(actions[:, 1].unsqueeze(-1),self.Kmax)   # 第二维是 kmin
            pmax_actions = actions[:, 2].unsqueeze(-1)*2-1  # 第三维是 pmax

            # 计算旧策略的对数概率
            old_kmax_log_probs = kmax_dist.log_prob(kmax_actions).sum(dim=-1, keepdim=True).detach()
            old_kmin_log_probs = kmin_dist.log_prob(kmin_actions).sum(dim=-1, keepdim=True).detach()
            old_pmax_log_probs = pmax_dist.log_prob(pmax_actions).sum(dim=-1, keepdim=True).detach()

            # 多次更新策略网络和价值网络
            policy_losses, value_losses = [], []
            for i in range(self.target_update):
                # 计算新策略的对数概率
                policy_new = self.policy(states)
                #print("this is ",i," iteration")
                kmax_dist = torch.distributions.Normal(policy_new["kmax"][0], policy_new["kmax"][1])
                kmin_dist = torch.distributions.Normal(policy_new["kmin"][0], policy_new["kmin"][1])
                pmax_dist = torch.distributions.Normal(policy_new["pmax"][0], policy_new["pmax"][1])

                # 计算新策略的对数概率
                new_kmax_log_probs = kmax_dist.log_prob(kmax_actions).sum(dim=-1, keepdim=True)
                new_kmin_log_probs = kmin_dist.log_prob(kmin_actions).sum(dim=-1, keepdim=True)
                new_pmax_log_probs = pmax_dist.log_prob(pmax_actions).sum(dim=-1, keepdim=True)

                # 计算 ratio 和 clipped surrogate objective
                kmax_ratio = torch.exp(new_kmax_log_probs - old_kmax_log_probs)
                kmin_ratio = torch.exp(new_kmin_log_probs - old_kmin_log_probs)
                pmax_ratio = torch.exp(new_pmax_log_probs - old_pmax_log_probs)
                #print("ratio = ",kmax_ratio,'/t',kmin_ratio,'/t',pmax_ratio)

                kmax_ratio = torch.clamp(kmax_ratio, min=1e-6, max=1e6) 
                kmin_ratio = torch.clamp(kmin_ratio, min=1e-6, max=1e6) 
                pmax_ratio = torch.clamp(pmax_ratio, min=1e-6, max=1e6) 

                kmax_surr1 = kmax_ratio * advantages
                kmin_surr1 = kmin_ratio * advantages
                pmax_surr1 = pmax_ratio * advantages

                kmax_surr2 = torch.clamp(kmax_ratio, 1 - self.eps, 1 + self.eps) * advantages
                kmin_surr2 = torch.clamp(kmin_ratio, 1 - self.eps, 1 + self.eps) * advantages
                pmax_surr2 = torch.clamp(pmax_ratio, 1 - self.eps, 1 + self.eps) * advantages

                # 计算每个参数的策略损失
                kmax_policy_loss = -torch.min(kmax_surr1, kmax_surr2).mean()
                kmin_policy_loss = -torch.min(kmin_surr1, kmin_surr2).mean()
                pmax_policy_loss = -torch.min(pmax_surr1, pmax_surr2).mean()

                # 联合策略损失（加权求和）
                policy_loss = kmax_policy_loss + kmin_policy_loss + pmax_policy_loss

                # 计算价值函数损失
                value_loss = F.mse_loss(self.value(states), td_target)

                # 更新策略网络

                self.policy_opt.zero_grad()
                policy_loss.backward()
                # for name, param in self.policy.named_parameters():
                #     if param.grad is not None:
                #         print(f"Policy gradient {name}: {param.grad.abs().max().item()}")
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
                self.policy_opt.step()

                # 更新价值网络
                self.value_opt.zero_grad()
                value_loss.backward()
                self.value_opt.step()

                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            # 返回平均损失
            return np.mean(policy_losses), np.mean(value_losses)
        
    def compute_gae(self,td_errors, gamma, lmbda):
        advantages = []
        for t in reversed(range(len(td_errors))):
            if t == len(td_errors) - 1:
                next_advantage = 0
            else:
                next_advantage = advantages[-1]
            advantage = td_errors[t] + gamma * lmbda * next_advantage 
            advantages.append(advantage)
        advantages.reverse()
        return torch.tensor(advantages).to(td_errors.device)
    
    def remapping(self, action_info):
        # 将 kmax 元组转换为列表
        kmax_list = list(action_info["kmax"])
        kmax_list[0] = self.Kmax[1] + (self.Kmax[0] - self.Kmax[1]) * (kmax_list[0] + 1) / 2  # [-1, 1] -> [2, 10240]
        kmax_list[1] = 2.0 * kmax_list[1]
        action_info["kmax"] = tuple(kmax_list)  # 如果需要，可以转换回元组

        # 同样的方法处理 kmin 和 pmax
        kmin_list = list(action_info["kmin"])
        kmin_list[0] = self.Kmin[1] + (self.Kmin[0] - self.Kmin[1]) * (kmin_list[0] + 1) / 2
        kmin_list[1] = 2.0 * kmin_list[1]
        action_info["kmin"] = tuple(kmin_list)

        pmax_list = list(action_info["pmax"])
        pmax_list[0] = (pmax_list[0] + 1) / 2
        pmax_list[1] = 2.0 * pmax_list[1]
        action_info["pmax"] = tuple(pmax_list)

        return action_info
        # print(f"kmax_mu: {kmax_mu}")
        # print(f"kmax_std: {kmax_std}")
        # print(f"kmin_mu: {kmin_mu}")
        # print(f"kmin_std: {kmin_std}")
        # print(f"pmax_mu: {pmax_mu}")
        # print(f"pmax_std: {pmax_std}")

  

    def load(self,policy_name,value_name):
        if os.path.exists(self.model_file):
            policy_dict = torch.load(self.model_file+policy_name)
            self.policy.load_state_dict(policy_dict)
            print("agent load actor_net successfully")
            value_dict = torch.load(self.model_file+value_name)
            self.value.load_state_dict(value_dict)
            
            #print(self.q_net.state_dict())
            print("agent load critic_net successfully")
            if self.train == 'F':
                self.value.eval()
                self.policy.eval()

        else:
            print("load failed")


    def save(self,iter):
        now = datetime.now()
        # 格式化日期时间字符串
        formatted_now = now.strftime("%d_%H")

        actor_name = f'Model/model1/actor_'+str(iter)+'.pt'
        critic_name = f'Model/model1/critic_'+str(iter)+'.pt'
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(actor_name), exist_ok=True)
        os.makedirs(os.path.dirname(critic_name), exist_ok=True)
        torch.save(self.policy.state_dict(),actor_name)
        torch.save(self.value.state_dict(),critic_name)

    def store_transition(self, state, action, reward, next_state):
        self.replay_buffer.add(state, action, reward, next_state)






