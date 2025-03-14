import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

class DistributionNetwork:
    """
    电力配网模拟器
    """
    @staticmethod
    def get_args(parser):
        parser.add_argument('--file_path', '-f', type=str, default='./structure/Node8_loop.txt', help='Path to the network data file')
        parser.add_argument('--time_steps', '-t', type=int, default=1000, help='Number of time steps to simulate')
        parser.add_argument('--outage_branch', '-o', type=int, default=None, help='Index of the branch to simulate outage')
        return parser

    def __init__(self, args):
        self.file_path = args.file_path
        self.time_steps = args.time_steps
        self.outage_branch_idx = args.outage_branch

        self.filename = self.file_path.split('/')[-1]
        self.structure = self._read_data(self.file_path)
        
        self.outage_pairs = []
        self.incidence_matrix = self.build_incidence_matrix(self.structure)
        self.branch_admittance = self.get_branch_admittance(self.n_branches)
        self.Y_e = np.diag(self.branch_admittance)
        # Generate random current injections for non-slack buses (dimensions: time_steps x (n_buses - 1))
        self.I = np.random.uniform(0, 0.2, size=(self.time_steps, self.n_buses-1))

        self.V, self.Y_reduced = self.generate_voltage_data(self.I, self.incidence_matrix, self.Y_e)

        self.incidence_matrix_outage, self.Y_e_outage = self.outage(self.incidence_matrix.copy(), self.Y_e.copy(), branch_indices=self.outage_branch_idx)

        self.V_outage, self.Y_reduced_outage = self.generate_voltage_data(self.I, self.incidence_matrix_outage, self.Y_e_outage)

    def _read_data(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Remove the trailing ';' and split by tab
                elements = line.rstrip(';\n').split(',')
                data.append(elements)
        columns = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC', 'tap', 'angle', 'status', 'angmin', 'angmax']
        df = pd.DataFrame(data, columns=columns, dtype=float)
        return df

    def build_incidence_matrix(self, structure):
        """
        根据DataFrame（仅包含'fbus'和'tbus'）构造 incidence 矩阵
        节点编号从1开始，矩阵按0-index存储
        """
        # 计算总母线数量（假设母线编号从1开始）
        self.n_buses = int(max(self.structure['fbus'].max(), self.structure['tbus'].max()))
        self.n_branches = len(self.structure)
        A = np.zeros((self.n_branches, self.n_buses))
        for i, row in structure.iterrows():
            tbus = int(row['tbus'])
            fbus = int(row['fbus'])
            A[i, fbus - 1] = 1   # 发电侧+1
            A[i, tbus - 1] = -1  # 接收侧-1
        return A
    
    def get_branch_admittance(self, n_branches=None):
        return np.random.uniform(0.4, 0.6, size=n_branches)

    def compute_Y(self, A, Y_e, slack_bus=0):
        """
        计算母线导纳矩阵 Y = A^T * Y_e * A，并移除参考母线（默认Bus 1，即index 0）
        """
        Y = A.T @ Y_e @ A
        Y_reduced = np.delete(Y, slack_bus, axis=0)
        Y_reduced = np.delete(Y_reduced, slack_bus, axis=1)
        return Y_reduced

    def generate_voltage_data(self, I, A, Y_e):
        """
        根据公式 V = I * Y^{-1} 生成电压数据（平衡母线为Bus 1）
        返回计算得到的电压数据及降阶后的母线导纳矩阵
        """
        Y_reduced = self.compute_Y(A, Y_e, slack_bus=0)
        Y_inv = np.linalg.inv(Y_reduced)

        # I的尺寸为 (time_steps, n_buses-1)
        V = I @ Y_inv
        return V, Y_reduced

    def outage(self, incidence_admittance, Y_e, branch_indices=None):
        """
        模拟线路跳闸：随机移除一条支路（不考虑第一条和最后一条以避免极端情况）
        同时更新incidence矩阵和分支导纳矩阵
        """
        if branch_indices is None:
            # 从1到 A.shape[0]-2 中随机选择一个支路
            self.outage_branch_idx = np.random.choice(np.arange(1, incidence_admittance.shape[0]-1))
        else:
            self.outage_branch_idx = branch_indices
        from_bus = int(np.where(incidence_admittance[self.outage_branch_idx] == 1)[0][0] + 1)
        to_bus = int(np.where(incidence_admittance[self.outage_branch_idx] == -1)[0][0] + 1)
        self.outage_pairs.append((from_bus, to_bus))
        # 移除该支路
        branch_admittance_outage = np.delete(incidence_admittance, self.outage_branch_idx, axis=0)
        Y_e_outage = np.delete(Y_e, self.outage_branch_idx, axis=0)
        Y_e_outage = np.delete(Y_e_outage, self.outage_branch_idx, axis=1)
        
        return branch_admittance_outage, Y_e_outage

    def print_info(self):
        print("\n[Network Information]")
        # print("Incidence Matrix A:\n", self.incidence_matrix)
        # print("Branch Admittance Y_e:\n", np.diag(self.branch_admittance))
        print("Normal Operation - Mean Voltage per bus:\n", np.mean(self.V, axis=0))
        print("Outage Operation - Mean Voltage per bus:\n", np.mean(self.V_outage, axis=0))
        print("Normal Operation - Voltage Covariance:\n", np.cov(self.V, rowvar=False))
        print("Outage Operation - Voltage Covariance:\n", np.cov(self.V_outage, rowvar=False))
        print("Buses #:\t", self.n_buses)
        print("Branches #:\t", self.n_branches)
        print("Outage Buses:\t", self.outage_pairs)
        print("Outage Branch:\t", self.outage_branch_idx)

    def plot(self, path=None):
        """
        绘制电压数据
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        for i in range(min(self.n_buses - 1, 5)):
            plt.plot(self.V[:, i], label=f'Bus {i+2}')
        plt.xlabel("Time Steps")
        plt.ylabel("Voltage Magnitude")
        plt.title("Normal Operation Voltage")
        plt.legend()

        plt.subplot(122)
        for i in range(min(self.n_buses - 1, 5)):
            plt.plot(self.V_outage[:, i], label=f'Bus {i+2}')
        plt.xlabel("Time Steps")
        plt.ylabel("Voltage Magnitude")
        plt.title("Outage Operation Voltage")
        plt.legend()

        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()
    
    def draw_network(self, path=None):
        """
        绘制配网拓扑图
        """
        plt.figure(figsize=(12, 6))
        plt.title(f"Network Topology - {self.filename}")
        plt.axis('off')
        plt.subplot(121)
        G = nx.Graph()
        for bus in range(self.n_buses):
            G.add_node(bus+1)
        for row in self.incidence_matrix:
            from_bus = np.where(row == 1)[0][0] + 1
            to_bus = np.where(row == -1)[0][0] + 1
            G.add_edge(from_bus, to_bus)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold')

        plt.subplot(122)
        G_outage = nx.Graph()
        for bus in range(self.n_buses):
            G_outage.add_node(bus+1)
        for row in self.incidence_matrix_outage:
            from_bus = np.where(row == 1)[0][0] + 1
            to_bus = np.where(row == -1)[0][0] + 1
            G_outage.add_edge(from_bus, to_bus)
        pos = nx.spring_layout(G_outage)
        nx.draw(G_outage, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()
        
    def save_data(self):
        """
        保存模拟数据
        """
        subfolder = self.filename.split('.')[-2]
        output_dir = f'data/{subfolder}'
        os.makedirs(output_dir, exist_ok=True)
        voltage_df = pd.DataFrame(self.V, columns=[f'Bus_{i+2}' for i in range(self.n_buses-1)])
        voltage_df.to_csv(f'{output_dir}/voltage_data.csv', index=False)
        voltage_outage_df = pd.DataFrame(self.V_outage, columns=[f'Bus_{i+2}' for i in range(self.n_buses-1)])  # Bus 1 is removed
        voltage_outage_df.to_csv(f'{output_dir}/voltage_data_outage.csv', index=False)

        self.plot(f'{output_dir}/voltage_data.pdf')
        self.draw_network(f'{output_dir}/network_topology.pdf')

if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Distribution Network Simulator')
    parser = DistributionNetwork.get_args(parser)
    args, _ = parser.parse_known_args()

    network = DistributionNetwork(args)

    network.save_data()
    network.print_info()
