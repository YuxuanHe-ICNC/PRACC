import matplotlib.pyplot as plt
import numpy as np

flow_num = 600
train_num = 1
def plot_return(return_list,action_dim):
    plt.figure()
    plt.plot(range(len(return_list)), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on ECN')
    plt.savefig('figure/reward/ECN'+'_'+str(action_dim)+
                '_'+str(flow_num)+'_'+str(train_num)+'.png')


def plot_egress(egress,action_dim):
    plt.figure()
    plt.plot(range(len(egress)), egress)
    plt.xlabel('Episodes')
    plt.ylabel('Egress')
    plt.title('DQN on ECN')
    plt.savefig('figure/egress/ECN'+'_'+str(action_dim)+
                '_'+str(flow_num)+'_'+str(train_num)+'.png')

def plot_DL(DL_list,action_dim):
    plt.figure()
    #print("DL_list",DL_list,'\n')
    plt.plot(range(len(DL_list)), DL_list)
    plt.xlabel('Episodes')
    plt.ylabel('qlenth')
    plt.title('DQN on ECN DL(qlength)')
    plt.savefig('figure/DL/ECN'+'_'+str(action_dim)+
                '_'+str(flow_num)+'_'+str(train_num)+'.png')

def plot_linkrate(linkrate_list,action_dim):
    plt.figure()
    #print("linkrate_list",linkrate_list,'\n')
    plt.plot(range(len(linkrate_list)), linkrate_list)
    plt.xlabel('Episodes')
    plt.ylabel('linkrate')
    plt.title('DQN on ECN link_rate')
    plt.savefig('figure/Linkrate/ECN'+'_'+str(action_dim)+
                '_'+str(flow_num)+'_'+str(train_num)+'.png')

def PlotResult(return_list,DL_list,link_rate_list,action_dim,egress):
    plot_return(return_list,action_dim)
    plot_DL(DL_list,action_dim)
    plot_linkrate(link_rate_list,action_dim)
    plot_egress(egress,action_dim)