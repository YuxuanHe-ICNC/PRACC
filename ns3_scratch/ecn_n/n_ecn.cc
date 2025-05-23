/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License version 2 as
* published by the Free Software Foundation;
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include <ns3/log.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <time.h> 
#include "ns3/core-module.h"
#include "ns3/qbb-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/error-model.h"
#include <ns3/rdma.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/switch-node.h>
#include <ns3/sim-setting.h>
#include <ns3/netanim-module.h>
#include "ns3/constant-position-mobility-model.h"
#include <ns3/flow-monitor-module.h>
#include <ns3/switch-mmu.h>
#include <ns3/traced-callback.h>

#include "ns3/ns3-ai-module.h"
#include <string>
#include <vector>

using namespace ns3;
using namespace std;

uint32_t activeQPs = 0;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");
char *p1=NULL;
uint32_t cc_mode = 1;
bool enable_qcn = true, use_dynamic_pfc_threshold = true;
uint32_t packet_payload_size = 1000, l2_chunk_size = 0, l2_ack_interval = 0;
double pause_time = 5; 
double simulator_stop_time = 3.01;
std::string data_rate, link_delay, topology_file, flow_file, trace_file, trace_output_file;


double alpha_resume_interval = 55, rp_timer, ewma_gain = 1 / 16;
double rate_decrease_interval = 4;
uint32_t fast_recovery_times = 5;
std::string rate_ai, rate_hai, min_rate = "100Mb/s";
std::string dctcp_rate_ai = "1000Mb/s";

bool clamp_target_rate = false, l2_back_to_zero = false;
double error_rate_per_link = 0.0;
uint32_t has_win = 1;
uint32_t global_t = 1;
uint32_t mi_thresh = 5;
bool var_win = false, fast_react = true;
bool multi_rate = true;
bool sample_feedback = false;
double pint_log_base = 1.05;
double pint_prob = 1.0;
double u_target = 0.95;
uint32_t int_multi = 1;
bool rate_bound = true;

uint32_t ack_high_prio = 0;
uint64_t link_down_time = 0;
uint32_t link_down_A = 0, link_down_B = 0;

uint32_t enable_trace = 1;

uint32_t buffer_size = 16;

uint32_t qlen_dump_interval = 100000000, qlen_mon_interval = 100;
uint64_t qlen_mon_start = 2000000000, qlen_mon_end = 2100000000;
string qlen_mon_file;

std::string fct_output_file = "output_fct.txt";
std::string pfc_output_file = "output_pfc.txt";
const std::string outputFileName = "/home/hyxx/High-Precision-Congestion-Control/ns3-rdma/ns-3.33/output/ns3switch/switch_node_data_f70.txt";


unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
unordered_map<uint64_t, double> rate2pmax;

/************************************************
 * Runtime varibles
 ***********************************************/
 //读取拓扑文件 流文件和跟踪文件
std::ifstream topof, flowf, tracef;
	
NodeContainer n;//存储网络中的节点

uint64_t nic_rate;//无线网卡速率的变量

uint64_t maxRtt, maxBdp;//最大往返时间和最大带宽延乘积

//定义网络接口属性的结构体变量  索引 接口状态 延迟 带宽
struct Interface{
	uint32_t idx;
	bool up;
	uint64_t delay;
	uint64_t bw;

	Interface() : idx(0), up(false){}
};
map<Ptr<Node>, map<Ptr<Node>, Interface> > nbr2if;//用于将节点映射到其邻接节点以及对应的接口信息
// Mapping destination to next hop for each node: <node, <dest, <nexthop0, ...> > >
//映射每个节点到其目标节点的下一跳，可能是一个节点或多个节点的向量
map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node> > > > nextHop;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairDelay;//映射每一对节点之间的延迟
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairTxDelay;//映射每一对节点之间的传输延迟
map<uint32_t, map<uint32_t, uint64_t> > pairBw;//映射每一对节点之间的 带宽
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairBdp;//映射每一对节点之间的 带宽延迟积
map<uint32_t, map<uint32_t, uint64_t> > pairRtt;//映射每一对节点之间的 往返时间

std::vector<Ipv4Address> serverAddress;//存储IPv4地址的向量，可能表示服务器的地址

// maintain port number for each host pair
std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t> > portNumder;//维护每个主机对的端口号
//结构体表示网络流的输入参数，包括源地址 (src)、目标地址 (dst)、数据生成速率 (pg)、最大数据包数量 (maxPacketCount)、端口号 (port, dport)、开始时间 (start_time) 和索引 (idx)


struct FlowInput{
	uint32_t src, dst, pg, maxPacketCount, port, dport;
	double start_time;
	uint32_t idx;
};
FlowInput flow_input = {0};
uint32_t flow_num;



/////////////////////////////////////////////////////////////
//           ns3-ai  interface
////////////////////////////////////////////////////////////

// Global pointer for getting switch node data
std::vector<Ptr<SwitchNode>> globalSwList;

constexpr uint32_t NUM_PORT = 432; // sum switch egress port(used) 
const size_t leafswitch_portnum = 30; // leaf switch connect 24 server and 6 spine
const size_t spineswitch_portnum = 12; // spine connect 12 leaf switch 

struct sEcnRlEnv {
    uint32_t egressqlen[NUM_PORT]; // egress qlen list
    double linkrate[NUM_PORT]; // link rate list
    double ecnlinkrate[NUM_PORT]; // ecn marked rate list
    uint32_t ecnmin[NUM_PORT]; 
    uint32_t ecnmax[NUM_PORT];
    double ecnpmax[NUM_PORT];
    uint8_t envType; // 
    int64_t simTime_us; // simlation time 
}Packed ;
struct EcnRlAct
{
    uint32_t newecnmin[NUM_PORT];
    uint32_t newecnmax[NUM_PORT];
    double newecnpmax[NUM_PORT];
};


class EcnRlEnv : public Ns3AIRL<sEcnRlEnv,EcnRlAct>
{

public:
    EcnRlEnv () = delete;
    EcnRlEnv (uint16_t id);
	std::vector<Ptr<SwitchNode>>  m_globalSwList;//全部交换机的列表
	void SetGlobalSwList(std::vector<Ptr<SwitchNode>> SwList) {
        m_globalSwList = SwList;
    }
    
    void ScheduleNextStateRead(Time startTime) {
        Simulator::Schedule(startTime, &EcnRlEnv::DoScheduleNextStateRead, this);
    }
	void DoScheduleNextStateRead ();
 	bool m_started{false};
  	Time m_timeStep{NanoSeconds(500000)}; // Monitoring interval
protected:

  //游戏是否结束  奖励
  bool m_isGameOver;
  float m_envReward;
  
  //存储action的结果
  uint32_t m_new_min[NUM_PORT];
  uint32_t m_new_max[NUM_PORT];
  double m_new_pmax[NUM_PORT];

};

EcnRlEnv::EcnRlEnv(uint16_t id) : Ns3AIRL<sEcnRlEnv, EcnRlAct>(id) {
    SetCond(2, 0);      ///< Set the operation lock (even for ns-3 and odd for python).
}

//EcnRlEnv::ScheduleNextStateRead(Ptr<SwitchNode> Switch)
void
EcnRlEnv::DoScheduleNextStateRead(){

    auto env = EnvSetterCond();
    env->envType = 1;
    env->simTime_us = Simulator::Now().GetNanoSeconds();
	const size_t totalPorts = 432;
    // Get data from first switch

   // 初始化全局端口索引
    size_t globalPortIndex = 0;
	//std::cout << "current time (ns) : " << env->simTime_us << std::endl;
    for (size_t i = 0; i <= m_globalSwList.size() && globalPortIndex < totalPorts; ++i) {
		// 18 switches,432 ports
        auto& switchNode = m_globalSwList[i];
		//std::cout << "Switch device: " <<i<<'\t'<< switchNode->GetNDevices() << '\t'<< switchNode->GetId()<< std::endl;
         // 获取当前交换机的端口数量
		if (globalPortIndex <= 359){
			// get information from leaf switch egresss port,先访问叶子，再访问spine
			for (size_t j = 1; j <= leafswitch_portnum && globalPortIndex < totalPorts; ++j, ++globalPortIndex) 
			{
			//std::cout <<globalPortIndex<<'\t'<<j<<std::endl;
			//ensure the port index,from 1 to 30
				env->egressqlen[globalPortIndex] = switchNode->average_queue_port_list[j];
				env->linkrate[globalPortIndex] = switchNode->txRate[j];
				env->ecnlinkrate[globalPortIndex] = switchNode->ecn_txRate[j];
				env->ecnmin[globalPortIndex] = switchNode->min_thresh[j];
				env->ecnmax[globalPortIndex] = switchNode->max_thresh[j];
				env->ecnpmax[globalPortIndex] = switchNode->pmax_thresh[j];
        	}
		}else{
			// get information from spine switch egresss port
			for (size_t j = 1; j <= spineswitch_portnum && globalPortIndex < totalPorts; ++j, ++globalPortIndex) 
			{
				//ensure the port index ,from 1 to 12
			//std::cout <<globalPortIndex<<'\t'<<j<<std::endl;
				env->egressqlen[globalPortIndex] = switchNode->average_queue_port_list[j];
				env->linkrate[globalPortIndex] = switchNode->txRate[j];
				env->ecnlinkrate[globalPortIndex] = switchNode->ecn_txRate[j];
				env->ecnmin[globalPortIndex] = switchNode->min_thresh[j];
				env->ecnmax[globalPortIndex] = switchNode->max_thresh[j];
				env->ecnpmax[globalPortIndex] = switchNode->pmax_thresh[j];
			}
		}
        
		
    }
    SetCompleted();
    
    auto act = ActionGetterCond();
	//std::cout<<"Read ECN from ns3-ai"<<std::endl;
  	for(size_t i = 0; i < totalPorts; i++) {
		m_new_min[i] = act->newecnmin[i];
		m_new_max[i] = act->newecnmax[i];
		m_new_pmax[i] = act->newecnpmax[i];
	}
  	GetCompleted();
  	

	// 遍历每个交换机
	for (size_t switchIdx = 0; switchIdx < 18; ++switchIdx) {
		Ptr<SwitchNode> Switch = m_globalSwList[switchIdx];
		//std::cout<<"configure ECN"<<std::endl;
		if(switchIdx <12){
			// 开始的全局端口索引，基于当前交换机在列表中的位置
			// config ECN threshold to leaf swtich
			size_t startGlobalPortIndex = switchIdx * leafswitch_portnum;
			for (size_t localPortIdx = 0; localPortIdx < leafswitch_portnum; ++localPortIdx) 
			{
				size_t globalPortIndex = startGlobalPortIndex + localPortIdx;
				//std::cout << Switch->GetId() << '\t' <<Switch->GetNDevices() << '\t'<<globalPortIndex << ' '<< localPortIdx +1 <<std::endl;
				// 使用全局端口索引从总数组中获取值
				Switch->m_mmu->ConfigEcn(localPortIdx + 1, 
									m_new_min[globalPortIndex], 
									m_new_max[globalPortIndex], 
									m_new_pmax[globalPortIndex]);
			}
		}else{
			// config ECN threshold to spine swtich
			size_t startGlobalPortIndex = 360 + (switchIdx-12) * 12;
			for (size_t localPortIdx = 0; localPortIdx < spineswitch_portnum; ++localPortIdx) 
			{
				size_t globalPortIndex = startGlobalPortIndex + localPortIdx;
				//std::cout << Switch->GetId() << '\t' <<Switch->GetNDevices() << '\t'<<globalPortIndex << ' '<< localPortIdx +1 <<std::endl;
				// 使用全局端口索引从总数组中获取值
				
				Switch->m_mmu->ConfigEcn(localPortIdx + 1, 
									m_new_min[globalPortIndex], 
									m_new_max[globalPortIndex], 
									m_new_pmax[globalPortIndex]);
			}
		}
	}
	
	Simulator::Schedule(m_timeStep, &EcnRlEnv::DoScheduleNextStateRead, this);	
}



//读取流输入信息
void ReadFlowInput(){
	if (flow_input.idx < flow_num){
		flowf >> flow_input.src >> flow_input.dst >> flow_input.pg >> flow_input.dport >> flow_input.maxPacketCount >> flow_input.start_time;
		NS_ASSERT(n.Get(flow_input.src)->GetNodeType() == 0 && n.Get(flow_input.dst)->GetNodeType() == 0);
	}
}

//特定时间安装网络流的客户端应用程序
void ScheduleFlowInputs(){
	while (flow_input.idx < flow_num && Seconds(flow_input.start_time) == Simulator::Now()){
		uint32_t port = portNumder[flow_input.src][flow_input.dst]++; // get a new port number 
		RdmaClientHelper clientHelper(flow_input.pg, serverAddress[flow_input.src], serverAddress[flow_input.dst], port, flow_input.dport, flow_input.maxPacketCount, has_win?(global_t==1?maxBdp:pairBdp[n.Get(flow_input.src)][n.Get(flow_input.dst)]):0, global_t==1?maxRtt:pairRtt[flow_input.src][flow_input.dst]);
		ApplicationContainer appCon = clientHelper.Install(n.Get(flow_input.src));
		appCon.Start(Time(0));

		// get the next flow input
		flow_input.idx++;
		ReadFlowInput();
	}

	// schedule the next time to run this function
	if (flow_input.idx < flow_num){
		Simulator::Schedule(Seconds(flow_input.start_time)-Simulator::Now(), ScheduleFlowInputs);
	}else { // no more flows, close the file
		flowf.close();
		/////////////////////////////
		
	}
}

//节点ID和IP地址之间的转换
Ipv4Address node_id_to_ip(uint32_t id){
	return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
}

uint32_t ip_to_node_id(Ipv4Address ip){
	return (ip.Get() >> 8) & 0xffff;
}

//RDMA qp
void qp_finish(FILE* fout,unsigned int *ptr, Ptr<RdmaQueuePair> q){
	static uint16_t num_close=0;
	uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);//给定的RDMA队列中获取源节点和目标节点id
	uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];//计算Round Trip Time，往返时间和bw(bps)
	//计算数据需要多少个数据包来传输，得到数据大小加上数据包头部大小后的total_bytes
	uint32_t total_bytes = q->m_size + ((q->m_size-1) / packet_payload_size + 1) * (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes required (with header but no INT)计算总的字节数
	//计算独立完成时间  基本的RTT加上理想数据传输时间，不考虑排队时延
	//total_bytes 单位为Byte，b的单位为bps
	//假设加上头部的总数据大小为100000 Bytes,带宽b为 100 Gpbs，
	uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
	
	// sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
	//输出到文件
	fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(), q->sport, q->dport, q->m_size, q->startTime.GetTimeStep(), (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
	fflush(fout);//刷新文件缓冲区

	// remove rxQp from the receiver 从接收方删除rx qp
	Ptr<Node> dstNode = n.Get(did);
	Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver> ();
	rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
	num_close++;
	std::cout << "finish flow  " << num_close << std::endl;
	if(ptr != nullptr){
		//////////////////////////////////////////////////////////////////////////
		// monitor flow finished number ,if it equals the sum flow ,stop simulation
		//////////////////////////////////////////////////////////////////////////
		if(num_close >= (*ptr)){
		Simulator::Now().GetNanoSeconds();
		Simulator::Stop();
		}
	}else{

	}

}

void get_pfc(FILE* fout, Ptr<QbbNetDevice> dev, uint32_t type){
	fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(), dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(), dev->GetIfIndex(), type);
}

struct QlenDistribution{
	vector<uint32_t> cnt; // cnt[i] is the number of times that the queue len is i KB

	void add(uint32_t qlen){
		uint32_t kb = qlen / 1000;
		if (cnt.size() < kb+1)
			cnt.resize(kb+1);
		cnt[kb]++;
	}
};
map<uint32_t, map<uint32_t, QlenDistribution> > queue_result;

void monitor_buffer(FILE* qlen_output, NodeContainer *n){
        //从所有节点中迭代
	for (uint32_t i = 0; i < n->GetN(); i++){
		//查看是否为交换机
		if (n->Get(i)->GetNodeType() == 1){ 
			//将节点转换为交换机节点
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
			if (queue_result.find(i) == queue_result.end())
				queue_result[i];
			// search all of the devices connect the switch
			for (uint32_t j = 1; j < sw->GetNDevices(); j++){
				uint32_t size = 0;
				//计算交换机 MMU 中每个队列的出口字节总大小
				for (uint32_t k = 0; k < SwitchMmu::qCnt; k++)
					size += sw->m_mmu->egress_bytes[j][k];
				// 将计算出的大小添加到queue_result映射中
				queue_result[i][j].add(size);

			}
		}
	}
	// 检查是否到了转储队列长度的时候
	if (Simulator::Now().GetTimeStep() % qlen_dump_interval == 0){
	        //打印当前时间节点
		fprintf(qlen_output, "time: %lu\n", Simulator::Now().GetTimeStep());
		for (auto &it0 : queue_result)
			for (auto &it1 : it0.second){
			//打印队列长度
				fprintf(qlen_output, "%u %u", it0.first, it1.first);
				auto &dist = it1.second.cnt;
				for (uint32_t i = 0; i < dist.size(); i++)
					fprintf(qlen_output, " %u", dist[i]);
				fprintf(qlen_output, "\n");
			}
		//关闭输出文件流
		fflush(qlen_output);
	}
	// 如果当前时间步长小于qlen_mon_end，则安排下一次调用monitor_buffer
	if (static_cast<uint64_t>(Simulator::Now().GetTimeStep()) < qlen_mon_end)
	//if (Simulator::Now().GetTimeStep() < qlen_mon_end)
		Simulator::Schedule(NanoSeconds(qlen_mon_interval), &monitor_buffer, qlen_output, n);
}

void CalculateRoute(Ptr<Node> host){
	// 用于广度优先搜索的队列（BFS）
	vector<Ptr<Node> > q;
	// 从主机到每个节点的距离
	map<Ptr<Node>, int> dis;
	map<Ptr<Node>, uint64_t> delay;//传播延迟
	map<Ptr<Node>, uint64_t> txDelay;//传输延迟
	map<Ptr<Node>, uint64_t> bw;//带宽
	// 初始化 BFS.
	q.push_back(host);
	dis[host] = 0;
	delay[host] = 0;
	txDelay[host] = 0;
	bw[host] = 0xfffffffffffffffflu;
	// 进行bfs
	for (int i = 0; i < (int)q.size(); i++){
		Ptr<Node> now = q[i];
		int d = dis[now];
		//遍历当前阶段的邻接节点
		for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++){
			// skip down link 跳过下行链路
			if (!it->second.up)
				continue;
			Ptr<Node> next = it->first;
			// If 'next' have not been visited.
			if (dis.find(next) == dis.end()){
				dis[next] = d + 1;
				delay[next] = delay[now] + it->second.delay;
				txDelay[next] = txDelay[now] + packet_payload_size * 1000000000lu * 8 / it->second.bw;
				bw[next] = std::min(bw[now], it->second.bw);
				// we only enqueue switch, because we do not want packets to go through host as middle point
				if (next->GetNodeType() == 1)
					q.push_back(next);
			}
			// if 'now' is on the shortest path from 'next' to 'host'.
			if (d + 1 == dis[next]){
				nextHop[next][host].push_back(now);
			}
		}
	}
	for (auto it : delay)
		pairDelay[it.first][host] = it.second;
	for (auto it : txDelay)
		pairTxDelay[it.first][host] = it.second;
	for (auto it : bw)
		pairBw[it.first->GetId()][host->GetId()] = it.second;
}


void CalculateRoutes(NodeContainer &n){
	for (int i = 0; i < (int)n.GetN(); i++){
		Ptr<Node> node = n.Get(i);
		if (node->GetNodeType() == 0)
			CalculateRoute(node);
	}
}

void SetRoutingEntries(){
	// For each node.
	for (auto i = nextHop.begin(); i != nextHop.end(); i++){
		Ptr<Node> node = i->first;
		auto &table = i->second;
		for (auto j = table.begin(); j != table.end(); j++){
			// The destination node.
			Ptr<Node> dst = j->first;
			// The IP address of the dst.
			Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
			// The next hops towards the dst.
			vector<Ptr<Node> > nexts = j->second;
			for (int k = 0; k < (int)nexts.size(); k++){
				Ptr<Node> next = nexts[k];
				uint32_t interface = nbr2if[node][next].idx;
				if (node->GetNodeType() == 1)
					DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
				else{
					node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface);
				}
			}
		}
	}
}

// take down the link between a and b, and redo the routing
void TakeDownLink(NodeContainer n, Ptr<Node> a, Ptr<Node> b){
	if (!nbr2if[a][b].up)
		return;
	// take down link between a and b
	nbr2if[a][b].up = nbr2if[b][a].up = false;
	nextHop.clear();
	CalculateRoutes(n);
	// clear routing tables
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 1)
			DynamicCast<SwitchNode>(n.Get(i))->ClearTable();
		else
			n.Get(i)->GetObject<RdmaDriver>()->m_rdma->ClearTable();
	}
	DynamicCast<QbbNetDevice>(a->GetDevice(nbr2if[a][b].idx))->TakeDown();
	DynamicCast<QbbNetDevice>(b->GetDevice(nbr2if[b][a].idx))->TakeDown();
	// reset routing table
	SetRoutingEntries();

	// redistribute qp on each host
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 0)
			n.Get(i)->GetObject<RdmaDriver>()->m_rdma->RedistributeQp();
	}
}

uint64_t get_nic_rate(NodeContainer &n){
	for (uint32_t i = 0; i < n.GetN(); i++)
		if (n.Get(i)->GetNodeType() == 0)
			return DynamicCast<QbbNetDevice>(n.Get(i)->GetDevice(1))->GetDataRate().GetBitRate();

        return 0;
		        
}

std::string prependBeforeDot(const std::string& path, const std::string& toPrepend) {
    size_t dotPos = path.find('.'); // 查找点的位置
    
    if (dotPos != std::string::npos) { // 确保找到了点
        // 在点之前插入字符串
        return path.substr(0, dotPos) + toPrepend + path.substr(dotPos);
    } else {
        std::cerr << "No dot found in the path." << std::endl;
        return path; // 如果没有找到点，直接返回原路径
    }
}
/////////////////////////////////////////////////////////////////
//////主函数程序开始
////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{	
	n = ns3::NodeContainer();
	clock_t begint, endt;
	begint = clock();
	CommandLine cmd;
	cmd.Parse(argc,argv);
	
	////////////////////////////////////////////////////////
	//Instantiate the rl environment and set the access lock
	///////////////////////////////////////////////////////
	
	int memblock_key = 2955;        ///< memory block key, need to keep the same in the python script
	cmd.AddValue ("key","memory block key",memblock_key);
	EcnRlEnv ecnrlenv(memblock_key);

	std::string outfct_file;
	//std::string outpfc_file;
	//std::string outqlen_file;
	//std::string outtrace_file;
		
		
#ifndef PGO_TRAINING
	if (argc > 1)
#else
	if (true)
#endif
	{
		//Read the configuration file
		std::ifstream conf;
#ifndef PGO_TRAINING
		conf.open(argv[1]);
#else
		conf.open(PATH_TO_PGO_CONFIG);
#endif
		while (!conf.eof())
		{
			std::string key;
			conf >> key;

			//std::cout << conf.cur << "\n";

			if (key.compare("ENABLE_QCN") == 0)
			{
				uint32_t v;
				conf >> v;
				enable_qcn = v;
				if (enable_qcn)
					std::cout << "ENABLE_QCN\t\t\t" << "Yes" << "\n";
				else
					std::cout << "ENABLE_QCN\t\t\t" << "No" << "\n";
			}
			else if (key.compare("USE_DYNAMIC_PFC_THRESHOLD") == 0)
			{
				uint32_t v;
				conf >> v;
				use_dynamic_pfc_threshold = v;
				if (use_dynamic_pfc_threshold)
					std::cout << "USE_DYNAMIC_PFC_THRESHOLD\t" << "Yes" << "\n";
				else
					std::cout << "USE_DYNAMIC_PFC_THRESHOLD\t" << "No" << "\n";
			}
			else if (key.compare("CLAMP_TARGET_RATE") == 0)
			{
				uint32_t v;
				conf >> v;
				clamp_target_rate = v;
				if (clamp_target_rate)
					std::cout << "CLAMP_TARGET_RATE\t\t" << "Yes" << "\n";
				else
					std::cout << "CLAMP_TARGET_RATE\t\t" << "No" << "\n";
			}
			else if (key.compare("PAUSE_TIME") == 0)
			{
				double v;
				conf >> v;
				pause_time = v;
				std::cout << "PAUSE_TIME\t\t\t" << pause_time << "\n";
			}
			else if (key.compare("DATA_RATE") == 0)
			{
				std::string v;
				conf >> v;
				data_rate = v;
				std::cout << "DATA_RATE\t\t\t" << data_rate << "\n";
			}
			else if (key.compare("LINK_DELAY") == 0)
			{
				std::string v;
				conf >> v;
				link_delay = v;
				std::cout << "LINK_DELAY\t\t\t" << link_delay << "\n";
			}
			else if (key.compare("PACKET_PAYLOAD_SIZE") == 0)
			{
				uint32_t v;
				conf >> v;
				packet_payload_size = v;
				std::cout << "PACKET_PAYLOAD_SIZE\t\t" << packet_payload_size << "\n";
			}
			else if (key.compare("L2_CHUNK_SIZE") == 0)
			{
				uint32_t v;
				conf >> v;
				l2_chunk_size = v;
				std::cout << "L2_CHUNK_SIZE\t\t\t" << l2_chunk_size << "\n";
			}
			else if (key.compare("L2_ACK_INTERVAL") == 0)
			{
				uint32_t v;
				conf >> v;
				l2_ack_interval = v;
				std::cout << "L2_ACK_INTERVAL\t\t\t" << l2_ack_interval << "\n";
			}
			else if (key.compare("L2_BACK_TO_ZERO") == 0)
			{
				uint32_t v;
				conf >> v;
				l2_back_to_zero = v;
				if (l2_back_to_zero)
					std::cout << "L2_BACK_TO_ZERO\t\t\t" << "Yes" << "\n";
				else
					std::cout << "L2_BACK_TO_ZERO\t\t\t" << "No" << "\n";
			}
			else if (key.compare("TOPOLOGY_FILE") == 0)
			{
				std::string v;
				conf >> v;
				topology_file = v;
				std::cout << "TOPOLOGY_FILE\t\t\t" << topology_file << "\n";
			}
			else if (key.compare("FLOW_FILE") == 0)
			{
				std::string v;
				conf >> v;
				flow_file = v;
				std::cout << "FLOW_FILE\t\t\t" << flow_file << "\n";
			}
			else if (key.compare("TRACE_FILE") == 0)
			{
				std::string v;
				conf >> v;
				trace_file = v;
				std::cout << "TRACE_FILE\t\t\t" << trace_file << "\n";
			}
			else if (key.compare("TRACE_OUTPUT_FILE") == 0)
			{
				std::string v;
				conf >> v;
				trace_output_file = v;
				if (argc > 2)
				{
					trace_output_file = trace_output_file + std::string(argv[2]);
				}
				std::cout << "TRACE_OUTPUT_FILE\t\t" << trace_output_file << "\n";
			}
			else if (key.compare("SIMULATOR_STOP_TIME") == 0)
			{
				double v;
				conf >> v;
				simulator_stop_time = v;
				std::cout << "SIMULATOR_STOP_TIME\t\t" << simulator_stop_time << "\n";
			}
			else if (key.compare("ALPHA_RESUME_INTERVAL") == 0)
			{
				double v;
				conf >> v;
				alpha_resume_interval = v;
				std::cout << "ALPHA_RESUME_INTERVAL\t\t" << alpha_resume_interval << "\n";
			}
			else if (key.compare("RP_TIMER") == 0)
			{
				double v;
				conf >> v;
				rp_timer = v;
				std::cout << "RP_TIMER\t\t\t" << rp_timer << "\n";
			}
			else if (key.compare("EWMA_GAIN") == 0)
			{
				double v;
				conf >> v;
				ewma_gain = v;
				std::cout << "EWMA_GAIN\t\t\t" << ewma_gain << "\n";
			}
			else if (key.compare("FAST_RECOVERY_TIMES") == 0)
			{
				uint32_t v;
				conf >> v;
				fast_recovery_times = v;
				std::cout << "FAST_RECOVERY_TIMES\t\t" << fast_recovery_times << "\n";
			}
			else if (key.compare("RATE_AI") == 0)
			{
				std::string v;
				conf >> v;
				rate_ai = v;
				std::cout << "RATE_AI\t\t\t\t" << rate_ai << "\n";
			}
			else if (key.compare("RATE_HAI") == 0)
			{
				std::string v;
				conf >> v;
				rate_hai = v;
				std::cout << "RATE_HAI\t\t\t" << rate_hai << "\n";
			}
			else if (key.compare("ERROR_RATE_PER_LINK") == 0)
			{
				double v;
				conf >> v;
				error_rate_per_link = v;
				std::cout << "ERROR_RATE_PER_LINK\t\t" << error_rate_per_link << "\n";
			}
			else if (key.compare("CC_MODE") == 0){
				conf >> cc_mode;
				std::cout << "CC_MODE\t\t" << cc_mode << '\n';
			}else if (key.compare("RATE_DECREASE_INTERVAL") == 0){
				double v;
				conf >> v;
				rate_decrease_interval = v;
				std::cout << "RATE_DECREASE_INTERVAL\t\t" << rate_decrease_interval << "\n";
			}else if (key.compare("MIN_RATE") == 0){
				conf >> min_rate;
				std::cout << "MIN_RATE\t\t" << min_rate << "\n";
			}else if (key.compare("FCT_OUTPUT_FILE") == 0){
				conf >> fct_output_file;
				std::cout << "FCT_OUTPUT_FILE\t\t" << fct_output_file << '\n';
			}else if (key.compare("HAS_WIN") == 0){
				conf >> has_win;
				std::cout << "HAS_WIN\t\t" << has_win << "\n";
			}else if (key.compare("GLOBAL_T") == 0){
				conf >> global_t;
				std::cout << "GLOBAL_T\t\t" << global_t << '\n';
			}else if (key.compare("MI_THRESH") == 0){
				conf >> mi_thresh;
				std::cout << "MI_THRESH\t\t" << mi_thresh << '\n';
			}else if (key.compare("VAR_WIN") == 0){
				uint32_t v;
				conf >> v;
				var_win = v;
				std::cout << "VAR_WIN\t\t" << v << '\n';
			}else if (key.compare("FAST_REACT") == 0){
				uint32_t v;
				conf >> v;
				fast_react = v;
				std::cout << "FAST_REACT\t\t" << v << '\n';
			}else if (key.compare("U_TARGET") == 0){
				conf >> u_target;
				std::cout << "U_TARGET\t\t" << u_target << '\n';
			}else if (key.compare("INT_MULTI") == 0){
				conf >> int_multi;
				std::cout << "INT_MULTI\t\t\t\t" << int_multi << '\n';
			}else if (key.compare("RATE_BOUND") == 0){
				uint32_t v;
				conf >> v;
				rate_bound = v;
				std::cout << "RATE_BOUND\t\t" << rate_bound << '\n';
			}else if (key.compare("ACK_HIGH_PRIO") == 0){
				conf >> ack_high_prio;
				std::cout << "ACK_HIGH_PRIO\t\t" << ack_high_prio << '\n';
			}else if (key.compare("DCTCP_RATE_AI") == 0){
				conf >> dctcp_rate_ai;
				std::cout << "DCTCP_RATE_AI\t\t\t\t" << dctcp_rate_ai << "\n";
			}else if (key.compare("PFC_OUTPUT_FILE") == 0){
				conf >> pfc_output_file;
				std::cout << "PFC_OUTPUT_FILE\t\t\t\t" << pfc_output_file << '\n';
			}else if (key.compare("LINK_DOWN") == 0){
				conf >> link_down_time >> link_down_A >> link_down_B;
				std::cout << "LINK_DOWN\t\t\t\t" << link_down_time << ' '<< link_down_A << ' ' << link_down_B << '\n';
			}else if (key.compare("ENABLE_TRACE") == 0){
				conf >> enable_trace;
				std::cout << "ENABLE_TRACE\t\t\t\t" << enable_trace << '\n';
			}else if (key.compare("KMAX_MAP") == 0){
				int n_k ;
				conf >> n_k;
				std::cout << "KMAX_MAP\t\t\t\t";
				for (int i = 0; i < n_k; i++){
					uint64_t rate;
					uint32_t k;
					conf >> rate >> k;
					rate2kmax[rate] = k;
					std::cout << ' ' << rate << ' ' << k;
				}
				std::cout<<'\n';
			}else if (key.compare("KMIN_MAP") == 0){
				int n_k ;
				conf >> n_k;
				std::cout << "KMIN_MAP\t\t\t\t";
				for (int i = 0; i < n_k; i++){
					uint64_t rate;
					uint32_t k;
					conf >> rate >> k;
					rate2kmin[rate] = k;
					std::cout << ' ' << rate << ' ' << k;
				}
				std::cout<<'\n';
			}else if (key.compare("PMAX_MAP") == 0){
				int n_k ;
				conf >> n_k;
				std::cout << "PMAX_MAP\t\t\t\t";
				for (int i = 0; i < n_k; i++){
					uint64_t rate;
					double p;
					conf >> rate >> p;
					rate2pmax[rate] = p;
					std::cout << ' ' << rate << ' ' << p;
				}
				std::cout<<'\n';
			}else if (key.compare("BUFFER_SIZE") == 0){
				conf >> buffer_size;
				std::cout << "BUFFER_SIZE\t\t\t\t" << buffer_size << '\n';
			}else if (key.compare("QLEN_MON_FILE") == 0){
				conf >> qlen_mon_file;
				std::cout << "QLEN_MON_FILE\t\t\t\t" << qlen_mon_file << '\n';
			}else if (key.compare("QLEN_MON_START") == 0){
				conf >> qlen_mon_start;
				std::cout << "QLEN_MON_START\t\t\t\t" << qlen_mon_start << '\n';
			}else if (key.compare("QLEN_MON_END") == 0){
				conf >> qlen_mon_end;
				std::cout << "QLEN_MON_END\t\t\t\t" << qlen_mon_end << '\n';
			}else if (key.compare("MULTI_RATE") == 0){
				int v;
				conf >> v;
				multi_rate = v;
				std::cout << "MULTI_RATE\t\t\t\t" << multi_rate << '\n';
			}else if (key.compare("SAMPLE_FEEDBACK") == 0){
				int v;
				conf >> v;
				sample_feedback = v;
				std::cout << "SAMPLE_FEEDBACK\t\t\t\t" << sample_feedback << '\n';
			}else if(key.compare("PINT_LOG_BASE") == 0){
				conf >> pint_log_base;
				std::cout << "PINT_LOG_BASE\t\t\t\t" << pint_log_base << '\n';
			}else if (key.compare("PINT_PROB") == 0){
				conf >> pint_prob;
				std::cout << "PINT_PROB\t\t\t\t" << pint_prob << '\n';
			}
			fflush(stdout);
		}
		conf.close();
	}
	else
	{
		std::cout << "Error: require a config file\n";
		fflush(stdout);
		return 1;
	}

	bool dynamicth = use_dynamic_pfc_threshold;

	Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(pause_time));
	Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(enable_qcn));
	Config::SetDefault("ns3::QbbNetDevice::DynamicThreshold", BooleanValue(dynamicth));

	// set int_multi
	
	IntHop::multi = int_multi;
	// IntHeader::mode
	if (cc_mode == 7) // timely, use ts
	{
		IntHeader::mode = IntHeader::TS;
		std::string mode_name = "_TIMELY";
		outfct_file = prependBeforeDot(fct_output_file, mode_name);
		//outpfc_file = prependBeforeDot(pfc_output_file,mode_name);
		//outqlen_file = prependBeforeDot(qlen_mon_file,mode_name);
		//outtrace_file = prependBeforeDot(trace_output_file,mode_name);
	}else if (cc_mode == 3) // hpcc, use int
	{
		IntHeader::mode = IntHeader::NORMAL;
		std::string mode_name = "_HPCC";
		outfct_file = prependBeforeDot(fct_output_file, mode_name);
		//outpfc_file = prependBeforeDot(pfc_output_file,mode_name);
		//outqlen_file = prependBeforeDot(qlen_mon_file,mode_name);
		//outtrace_file = prependBeforeDot(trace_output_file,mode_name);
	}else if (cc_mode == 10) // hpcc-pint
	{
		IntHeader::mode = IntHeader::PINT;
		std::string mode_name = "_PINT";
		outfct_file = prependBeforeDot(fct_output_file, mode_name);
		//outpfc_file = prependBeforeDot(pfc_output_file,mode_name);
		//outqlen_file = prependBeforeDot(qlen_mon_file,mode_name);
		//outtrace_file = prependBeforeDot(trace_output_file,mode_name);
	}	
	else // others, no extra header
	{
		IntHeader::mode = IntHeader::NONE;
		if (cc_mode == 0){
		std::string mode_name = "_PPO_fix";
		outfct_file = prependBeforeDot(fct_output_file, mode_name);
		//outpfc_file = prependBeforeDot(pfc_output_file,mode_name);
		//outqlen_file = prependBeforeDot(qlen_mon_file,mode_name);
		//outtrace_file = prependBeforeDot(trace_output_file,mode_name);
		}
		if (cc_mode == 1){
		std::string mode_name = "_DCQCN";
		outfct_file = prependBeforeDot(fct_output_file, mode_name);
		//outpfc_file = prependBeforeDot(pfc_output_file,mode_name);
		//outqlen_file = prependBeforeDot(qlen_mon_file,mode_name);
		//outtrace_file = prependBeforeDot(trace_output_file,mode_name);
		}
	}
		
	
	// Set Pint
	if (cc_mode == 10){
		Pint::set_log_base(pint_log_base);
		IntHeader::pint_bytes = Pint::get_n_bytes();
		printf("PINT bits: %d bytes: %d\n", Pint::get_n_bits(), Pint::get_n_bytes());
	}

	//SeedManager::SetSeed(time(NULL));

	topof.open(topology_file.c_str());
	flowf.open(flow_file.c_str());
	tracef.open(trace_file.c_str());
	
	uint32_t node_num, switch_num, link_num, trace_num;
	topof >> node_num >> switch_num >> link_num;
	flowf >> flow_num;

	tracef >> trace_num;

	
	//n.Create(node_num);
	std::vector<uint32_t> node_type(node_num, 0);//sender or reveiver node_type = 0;
	//读取switch的id
	std::cout<<"Reading switch ID"<<std::endl;
	for (uint32_t i = 0; i < switch_num; i++)
	{
		uint32_t sid;
		topof >> sid;
		node_type[sid] = 1;//switch node_type = 1
	}
	//创建node 和   switch node
	for (uint32_t i = 0; i < node_num; i++){
		if (node_type[i] == 0)
			n.Add(CreateObject<Node>());
		else{
			Ptr<SwitchNode> sw = CreateObject<SwitchNode>();

			n.Add(sw);
			//设置是否启用ecn
			sw->SetAttribute("EcnEnabled", BooleanValue(enable_qcn));
			
		}
	}

	
	NS_LOG_INFO("Create nodes.");
	
		//安装互联网协议栈
	InternetStackHelper internet;
	internet.Install(n);

	//
	// Assign IP to each server
	//对node看成server
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0){ // is server
			serverAddress.resize(i + 1);
			serverAddress[i] = node_id_to_ip(i);//安装ip地址
		}
	}

	NS_LOG_INFO("Create channels.");

	//
	// Explicitly create the channels required by the topology.
	//
		//创建速率误差模型对象
	Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
	//创建均匀分布的随机变量对象
	Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
	// 将随机变量对象设置为速率误差模型的随机变量
	rem->SetRandomVariable(uv);
	//设置随机变量的流
	uv->SetStream(50);
	//设置错误率属性
	rem->SetAttribute("ErrorRate", DoubleValue(error_rate_per_link));
	//错误单位属性
	rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
		
	//FILE *pfc_file = fopen(outpfc_file.c_str(), "w");
		// 创建一个QbbHelper对象
	QbbHelper qbb;
	// 创建一个Ipv4AddressHelper对象
	Ipv4AddressHelper ipv4;
	for (uint32_t i = 0; i < link_num; i++)
	{
			//从输入流读取链路信息 src dst data_rate link_delay error_rate 
		uint32_t src, dst;
		std::string data_rate, link_delay;
		double error_rate;
		topof >> src >> dst >> data_rate >> link_delay >> error_rate;
				// 获取源节点和目标节点
		Ptr<Node> snode = n.Get(src), dnode = n.Get(dst);
				//设置设备和信道属性
		qbb.SetDeviceAttribute("DataRate", StringValue(data_rate));
		qbb.SetChannelAttribute("Delay", StringValue(link_delay));
				//配置接收错误模型
		if (error_rate > 0)
		{
			Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
			Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
			rem->SetRandomVariable(uv);
			uv->SetStream(50);
			rem->SetAttribute("ErrorRate", DoubleValue(error_rate));
			rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
			qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
		}
		else
		{
			qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
		}
				
		fflush(stdout);

		// Assigne server IP
		// Note: this should be before the automatic assignment below (ipv4.Assign(d)),
		// because we want our IP to be the primary IP (first in the IP address list),
		// so that the global routing is based on our IP
		
		//安装qbb网络设备
		NetDeviceContainer d = qbb.Install(snode, dnode);

		// 输出端口吞吐量
		// create callback funtion
		//d.Get(9)->TraceConnectWithoutContext("PhyRxEnd", MakeCallback(&TraceSink));

		//配置源节点和目标节点为ipv4
		if (snode->GetNodeType() == 0){
			Ptr<Ipv4> ipv4 = snode->GetObject<Ipv4>();
			ipv4->AddInterface(d.Get(0));
			ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[src], Ipv4Mask(0xff000000)));
		}
		if (dnode->GetNodeType() == 0){
			Ptr<Ipv4> ipv4 = dnode->GetObject<Ipv4>();
			ipv4->AddInterface(d.Get(1));
			ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[dst], Ipv4Mask(0xff000000)));
		}

		// used to create a graph of the topology
		nbr2if[snode][dnode].idx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
		nbr2if[snode][dnode].up = true;
		nbr2if[snode][dnode].delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())->GetDelay().GetTimeStep();
		nbr2if[snode][dnode].bw = DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
		nbr2if[dnode][snode].idx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
		nbr2if[dnode][snode].up = true;
		nbr2if[dnode][snode].delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())->GetDelay().GetTimeStep();
		nbr2if[dnode][snode].bw = DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();

		// This is just to set up the connectivity between nodes. The IP addresses are useless
		//创建ip 地址
		char ipstring[20];
		sprintf(ipstring, "10.%d.%d.0", i / 254 + 1, i % 254 + 1);
		ipv4.SetBase(ipstring, "255.255.255.0");
		ipv4.Assign(d);

		// setup PFC trace 
		//DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(0))));
		//DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(1))));
	}
		//网卡速率计算
	nic_rate = get_nic_rate(n);

	// config switch
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 1){ // is switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
			uint32_t shift = 3; // by default 1/8
			//查找连接到交换机的设备,
			for (uint32_t j = 1; j < sw->GetNDevices(); j++){
				//将设备转换为qbb设备
				Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
				// set ecn
				//获取链路的带宽，并存入rate中
				uint64_t rate = dev->GetDataRate().GetBitRate();//bps
				//std::cout <<j << "\n";
				//std::cout << rate << "\n";
				
				//在switch-mmu.cc中进行配置和实现
				//####
								//####
				NS_ASSERT_MSG(rate2kmin.find(rate) != rate2kmin.end(), "must set kmin for each link speed");
				NS_ASSERT_MSG(rate2kmax.find(rate) != rate2kmax.end(), "must set kmax for each link speed");
				NS_ASSERT_MSG(rate2pmax.find(rate) != rate2pmax.end(), "must set pmax for each link speed");
				//对port j 设置ecn
				sw->m_mmu->ConfigEcn(j, rate2kmin[rate], rate2kmax[rate], rate2pmax[rate]);
				// std::cout<<rate2kmin[rate]<<"\t"<<rate2kmax[rate]<<"\t"<<rate2pmax[rate]<<'\n';
	
				// set pfc
				uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())->GetDelay().GetTimeStep();
				uint32_t headroom = rate * delay / 8 / 1000000000 * 3;//Bytes
			//std::cout<<"rate: "<<rate<<"delay: "<<delay<<"headroom: "<< headroom<<'\n';
				sw->m_mmu->ConfigHdrm(j, headroom);

				// set pfc alpha, proportional to link bw, rate=nic.rate
				sw->m_mmu->pfc_a_shift[j] = shift;//3
				while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0){
					sw->m_mmu->pfc_a_shift[j]--;
					rate /= 2;
				}
				//std::cout<<"rate: "<<rate<<"sw->m_mmu->pfc_a_shift:  "<<sw->m_mmu->pfc_a_shift[j]<<std::endl;
			}
			//configures the number of ports, buffer size, and node ID for the switch
			sw->m_mmu->ConfigNPort(sw->GetNDevices()-1);
			sw->m_mmu->ConfigBufferSize(buffer_size* 1024 * 1024);
			sw->m_mmu->node_id = sw->GetId();
		}
	}

	#if ENABLE_QP
	FILE *fct_output = fopen(outfct_file.c_str(), "w");
	//
	// install RDMA driver
	//

	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0){ // is server
			// create RdmaHw
			Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
			rdmaHw->SetAttribute("ClampTargetRate", BooleanValue(clamp_target_rate));
			rdmaHw->SetAttribute("AlphaResumInterval", DoubleValue(alpha_resume_interval));
			rdmaHw->SetAttribute("RPTimer", DoubleValue(rp_timer));
			rdmaHw->SetAttribute("FastRecoveryTimes", UintegerValue(fast_recovery_times));
			rdmaHw->SetAttribute("EwmaGain", DoubleValue(ewma_gain));
			rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate(rate_ai)));
			rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate(rate_hai)));
			rdmaHw->SetAttribute("L2BackToZero", BooleanValue(l2_back_to_zero));
			rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(l2_chunk_size));
			rdmaHw->SetAttribute("L2AckInterval", UintegerValue(l2_ack_interval));
			rdmaHw->SetAttribute("CcMode", UintegerValue(cc_mode));
			rdmaHw->SetAttribute("RateDecreaseInterval", DoubleValue(rate_decrease_interval));
			rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(min_rate)));
			rdmaHw->SetAttribute("Mtu", UintegerValue(packet_payload_size));
			rdmaHw->SetAttribute("MiThresh", UintegerValue(mi_thresh));
			rdmaHw->SetAttribute("VarWin", BooleanValue(var_win));
			rdmaHw->SetAttribute("FastReact", BooleanValue(fast_react));
			rdmaHw->SetAttribute("MultiRate", BooleanValue(multi_rate));
			rdmaHw->SetAttribute("SampleFeedback", BooleanValue(sample_feedback));
			rdmaHw->SetAttribute("TargetUtil", DoubleValue(u_target));
			rdmaHw->SetAttribute("RateBound", BooleanValue(rate_bound));
			rdmaHw->SetAttribute("DctcpRateAI", DataRateValue(DataRate(dctcp_rate_ai)));
			rdmaHw->SetPintSmplThresh(pint_prob);
			// create and install RdmaDriver
			Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
			Ptr<Node> node = n.Get(i);
			rdma->SetNode(node);
			rdma->SetRdmaHw(rdmaHw);
			node->AggregateObject (rdma);
			rdma->Init();

			// activeQPs++;
			rdma->TraceConnectWithoutContext("QpComplete", MakeBoundCallback (qp_finish, fct_output,&flow_num));
		}
	}
	#endif

	// set ACK priority on hosts
	if (ack_high_prio)
		RdmaEgressQueue::ack_q_idx = 0;
	else
		RdmaEgressQueue::ack_q_idx = 3;

	// setup routing
	CalculateRoutes(n);
	SetRoutingEntries();

	//
	// get BDP and delay
	//
	maxRtt = maxBdp = 0;
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() != 0)
			continue;
		for (uint32_t j = 0; j < node_num; j++){
			if (n.Get(j)->GetNodeType() != 0)
				continue;
			uint64_t delay = pairDelay[n.Get(i)][n.Get(j)];
			uint64_t txDelay = pairTxDelay[n.Get(i)][n.Get(j)];
			uint64_t rtt = delay * 2 + txDelay;
			uint64_t bw = pairBw[i][j];
			uint64_t bdp = rtt * bw / 1000000000/8; 
			pairBdp[n.Get(i)][n.Get(j)] = bdp;
			pairRtt[i][j] = rtt;
			if (bdp > maxBdp)
				maxBdp = bdp;
			if (rtt > maxRtt)
				maxRtt = rtt;
		}
	}
	//printf("maxRtt delay * 2 + txDelay =%lu maxBdp  rtt * bw / 1000000000/8 =%lu\n", maxRtt, maxBdp);

	// setup switch CC
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 1){ // switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
			sw->SetAttribute("CcMode", UintegerValue(cc_mode));
			sw->SetAttribute("MaxRtt", UintegerValue(maxRtt));
			for (uint32_t j = 1; j < sw->GetNDevices(); j++){
				std::cout << "port " << j <<std::endl;
			}
			// Store switch node in pointer,the globalSwList pointer will be passed as a parameter to the ecnrlenv.SetGlobalSwList function
			globalSwList.push_back(sw);
		}
	}
	#if 1
	// Print out the globalSwList
	for (auto& sw : globalSwList) 
	{
		std::cout << "Switch Node ID: " << sw->GetId() << std::endl;
		std::cout << "Switch device: " << sw->GetNDevices() << std::endl;
	}
	#endif

	// add trace
	/*
	NodeContainer trace_nodes;
	for (uint32_t i = 0; i < trace_num; i++)
	{
		uint32_t nid;
		tracef >> nid;
		if (nid >= n.GetN()){
			continue;
		}
		trace_nodes = NodeContainer(trace_nodes, n.Get(nid));
	}

	FILE *trace_output = fopen(outtrace_file.c_str(), "w");
	if (enable_trace)
		qbb.EnableTracing(trace_output, trace_nodes);

	// dump link speed to trace file
	
	{
		SimSetting sim_setting;
		for (auto i: nbr2if){
			for (auto j : i.second){
				uint16_t node = i.first->GetId();
				uint8_t intf = j.second.idx;
				uint64_t bps = DynamicCast<QbbNetDevice>(i.first->GetDevice(j.second.idx))->GetDataRate().GetBitRate();
				sim_setting.port_speed[node][intf] = bps;
			}
		}
		sim_setting.win = maxBdp;
		sim_setting.Serialize(trace_output);
	}*/
	
	Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	NS_LOG_INFO("Create Applications.");

	Time interPacketInterval = Seconds(0.0000005 / 2);

	// maintain port number for each host
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0)
			for (uint32_t j = 0; j < node_num; j++){
				if (n.Get(j)->GetNodeType() == 0)
					portNumder[i][j] = 10000; // each host pair use port number from 10000
			}
	}
		/*
		for (uint32_t i = 0; i < node_num; ++i) {
				Ptr<Node> node = n.Get(i);

				if (!node->GetObject<MobilityModel>()) {
						Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel>();
						node->AggregateObject(mobility);

						mobility->SetPosition(Vector(i, i,i));
				}
		}
		*/
	flow_input.idx = 0;
	if (flow_num > 0){
		ReadFlowInput();
		Simulator::Schedule(Seconds(flow_input.start_time)-Simulator::Now(), ScheduleFlowInputs);
	}

	topof.close();
	tracef.close();
	

	// schedule link down
	if (link_down_time > 0){
		Simulator::Schedule(Seconds(2) + MicroSeconds(link_down_time), &TakeDownLink, n, n.Get(link_down_A), n.Get(link_down_B));
	}

	// schedule buffer monitor
	//FILE* qlen_output = fopen(outqlen_file.c_str(), "w");
	//Simulator::Schedule(NanoSeconds(qlen_mon_start), &monitor_buffer, qlen_output, &n);


	std::cout << "Running Simulation.\n";
	fflush(stdout);


	//Simulator::Stop(Seconds(simulator_stop_time));
	// 在仿真的某个条件下，比如时间到达一定阈值时，提前终止仿真


	///////////////////////////////////////////////////////
	//Set the start time and schedule according to time steps
	///////////////////////////////////////////////////////
	
	Time startTime = Seconds(0.0); 
	ecnrlenv.SetGlobalSwList(globalSwList); //Pass in pointer for access to switch port data
	ecnrlenv.ScheduleNextStateRead(startTime);

	//Simulator::Schedule(startTime,&EcnRlEnv::ScheduleNextStateRead,&ecnrlenv,globalSw);
	
	Simulator::Run();
	Simulator::Destroy();
		

	NS_LOG_INFO("Done.");
	//fclose(trace_output);
			
	endt = clock();
	std::cout << (double)(endt - begint) / CLOCKS_PER_SEC << "\n";
}

