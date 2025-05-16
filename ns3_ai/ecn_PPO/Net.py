import  torch
import torch.nn.functional as F
import torch.nn as nn
import math
#from torchinfo import summary
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入: state + kmax + kmin + pmax
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        x = self.fc(state)
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim,hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(# 共享网络结构
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # kmin,: [-1, 1]
        self.kmin_fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.kmin_fc_std = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # kmax: [-1, 1] 
        self.kmax_fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.kmax_fc_std = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # pmax: [-2, 2] -> [0, 1]
        self.pmax_fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.pmax_fc_std = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.shared(x)
        
        # kmax参数转换
        kmax_mu = self.kmax_fc_mu(x)
        kmax_std = 2.0*self.kmax_fc_std(x)

        kmin_mu = self.kmin_fc_mu(x)
        kmin_std = 2.0*self.kmin_fc_std(x)

        pmax_mu = self.pmax_fc_mu(x)
        pmax_std = 2.0*self.pmax_fc_std(x)

        return {
            "kmax": (kmax_mu,kmax_std),
            "kmin": (kmin_mu,kmin_std),
            "pmax": (pmax_mu, pmax_std)     
        }
       
# # 创建模型实例
# model = ACC_Qnet(state_dim=18, action_dim=234)
# print(model)
# # 打印模型结构
# summary(model, input_size=(64, 18))
    
    # kmax_mu =  self.kmax[1] + (self.kmax[0] - self.kmax[1]) * (self.kmax_fc_mu(x) + 1) / 2  # [-2, 2] -> [2, 10240]
    # #标准差最小为1，增大探索以适应大动作空间
    # kmax_std = 2.0 *self.kmax_fc_std(x)

    # # kmin参数转换
    # kmin_mu = self.kmin[1] + (self.kmin[0] - self.kmin[1]) * (self.kmin_fc_mu(x) + 1) / 2
    # kmin_std = 2.0 * self.kmin_fc_std(x)
    # # pmax参数转换
    # pmax_mu = (self.pmax_fc_mu(x) + 1)/2
    # pmax_std = 2.0 * self.pmax_fc_std(x)

    # kmax_std = torch.clamp(kmax_std, min=1e-6, max=1e3)
    # kmin_std = torch.clamp(kmin_std, min=1e-6, max=1e3)
    # pmax_std = torch.clamp(pmax_std, min=1e-6, max=1e3) 

    # print(f"kmax_mu: {kmax_mu}")
    # print(f"kmax_std: {kmax_std}")
    # print(f"kmin_mu: {kmin_mu}")
    # print(f"kmin_std: {kmin_std}")
    # print(f"pmax_mu: {pmax_mu}")
    # print(f"pmax_std: {pmax_std}")
       