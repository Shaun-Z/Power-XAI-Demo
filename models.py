import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math

import torchvision.models as models
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# CNN Model
class VoltageCNN(nn.Module):
    def __init__(self, n_buses):
        super(VoltageCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Input: (batch, 1, n_buses)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Max pooling (reduces size by half)

        # Placeholder for dynamically computing Flatten dim
        self.flatten_dim = None

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Placeholder; will be adjusted dynamically
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 🔥 Changed to 1 neuron for binary classification

    def forward_features(self, x):
        """Computes CNN feature extraction"""
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, n_buses)
        x = self.forward_features(x)

        # Dynamically compute flatten dimension
        if self.flatten_dim is None:
            self.flatten_dim = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.flatten_dim, 64)  # Now correctly initialized
                
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 🔥 Single neuron output (logit)
        return x 

class CnnClassifier(nn.Module):
    def __init__(self):
        super(CnnClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(64), # fc1
            nn.ReLU(inplace=True),
            nn.Linear(64, 32), # fc2
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # fc3 (32 -> 1)
        )
    
    def forward(self, x):
        # x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, n_buses)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

class PairwiseProd(nn.Module):
    def __init__(self):
        super(PairwiseProd, self).__init__()

    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, data)
        返回: tensor, 形状为 (batch, channel, n*(n-1)/2)，其中 n 为 data 的大小
        """
        B, C, D = x.shape
        # 获取上三角（不含对角线）的索引, shape: [2, n*(n-1)/2] where n = D
        idx = torch.triu_indices(D, D, offset=1)
        # 利用索引对最后一维进行选取，并计算对应元素的乘积
        return x[:, :, idx[0]] * x[:, :, idx[1]]

class PairwiseSum(nn.Module):
    def __init__(self):
        super(PairwiseSum, self).__init__()
    
    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, data)
        返回: tensor, 形状为 (batch, channel, n*(n-1)/2)，其中 n 为 data 的大小
        """
        B, C, D = x.shape
        # 获取上三角（不含对角线）的索引, shape: [2, n*(n-1)/2] where n = D
        idx = torch.triu_indices(D, D, offset=1)
        # 利用索引对最后一维进行选取，并计算对应元素的和
        return (x[:, :, idx[0]] + x[:, :, idx[1]]) / 2
    
class PariwiseDiff(nn.Module):
    def __init__(self):
        super(PariwiseDiff, self).__init__()
    
    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, data)
        返回: tensor, 形状为 (batch, channel, n*(n-1)/2)，其中 n 为 data 的大小
        """
        B, C, D = x.shape
        # 获取上三角（不含对角线）的索引, shape: [2, n*(n-1)/2] where n = D
        idx = torch.triu_indices(D, D, offset=1)
        # 利用索引对最后一维进行选取，并计算对应元素的差
        return torch.abs(x[:, :, idx[0]] - x[:, :, idx[1]])

class CnnClassifierPairwise(nn.Module):
    def __init__(self):
        super(CnnClassifierPairwise, self).__init__()
        self.features = nn.Sequential(
            PairwiseProd(),
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(64), # fc1
            nn.ReLU(inplace=True),
            nn.Linear(64, 32), # fc2
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # fc3 (32 -> 1)
        )
    
    def forward(self, x):
        # x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, n_buses)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------------------
# 简单的全连接多层感知机（MLP）
class MLPClassifier(nn.Module):
    def __init__(self, feature_len, channel=1, hidden_dim=64, output_dim=1):
        """
        channel: 输入通道数
        data: 每个通道的特征数量
        hidden_dim: 隐藏层神经元数量
        output_dim: 输出维度，分类任务通常为类别数或1（比如二分类时输出logit）
        """
        super(MLPClassifier, self).__init__()
        # 将 channel 和 data 展平后作为输入维度
        self.input_dim = channel * feature_len
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, data)
        返回: tensor, 形状为 (batch, output_dim)
        """
        # 展平 channel 和 data 维度
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FeatureAttentionClassifier(nn.Module):
    def __init__(self, feature_len, channel=1, embed_dim=32, num_heads=8, hidden_dim=64, output_dim=1, dropout=0.005):
        super(FeatureAttentionClassifier, self).__init__()
        self.num_features = channel * feature_len
        # Embed each scalar feature to an embed_dim-dimensional space
        self.embedding = nn.Linear(1, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention with dropout
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network (FFN) with residual connection
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Final classification layer
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: tensor, shape (batch, channel, data)
        Returns: tensor, shape (batch, output_dim)
        """
        B, C, D = x.shape
        # Flatten channel and data dimensions: (B, num_features, 1)
        x = x.view(B, self.num_features, 1)
        # Embed each feature
        x = self.embedding(x)  # (B, num_features, embed_dim)
        x = self.layer_norm1(x)
        # Prepare for MultiheadAttention: (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)  
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        # Restore shape to (batch, num_features, embed_dim)
        x = x.permute(1, 0, 2)
        x = self.layer_norm2(x)
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        # Aggregate features (mean pooling over the sequence dimension)
        x = x.mean(dim=1)  
        # Final classification
        out = self.fc(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 嵌入维度
        dropout: dropout 概率
        max_len: 序列最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 构造位置编码矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 公式：PE(pos, 2i) = sin(pos/10000^(2i/d_model))；PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整形状为 (max_len, 1, d_model) 以便与输入相加
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor, 形状为 (seq_len, batch, d_model)
        返回: tensor, 添加了位置编码后的 x
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, feature_len, channel=1, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.1):
        """
        feature_len: 每个通道的特征数量
        channel: 输入通道数
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        hidden_dim: TransformerEncoder 中前馈网络的隐藏单元数
        num_layers: TransformerEncoder 层数
        output_dim: 输出维度，通常为类别数或1（例如二分类时）
        dropout: dropout 概率
        """
        super(TransformerClassifier, self).__init__()
        self.num_features = channel * feature_len
        # 将每个标量特征投影到 embed_dim 维空间
        self.embedding = nn.Linear(1, embed_dim)
        # 添加位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        # 构造 TransformerEncoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 最终分类层
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, feature_len)
        返回: tensor, 形状为 (batch, output_dim)
        """
        B, C, D = x.shape
        # 将 (channel, feature_len) 展平为一个序列，共 num_features 个 token，每个 token 为一个标量
        x = x.view(B, self.num_features, 1)  # (B, num_features, 1)
        # 嵌入每个标量特征
        x = self.embedding(x)  # (B, num_features, embed_dim)
        # TransformerEncoder 需要的输入形状为 (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)  # (num_features, B, embed_dim)
        # 添加位置编码
        x = self.pos_encoder(x)
        # TransformerEncoder 提取特征
        x = self.transformer_encoder(x)  # (num_features, B, embed_dim)
        # 对序列进行均值池化得到全局表示
        x = x.mean(dim=0)  # (B, embed_dim)
        # 最终分类
        out = self.fc(x)
        return out

# --------------------------
# 新增模型：基于 LSTM 的检测方法
class LSTMClassifier(nn.Module):
    def __init__(self, feature_len, channel=1, hidden_size=128, num_layers=1, output_dim=1, bidirectional=False, dropout=0.1):
        """
        feature_len: 每个通道的特征数量
        channel: 输入通道数
        hidden_size: LSTM 隐藏状态维度
        num_layers: LSTM 层数
        output_dim: 输出维度，通常为类别数或1（例如二分类时）
        bidirectional: 是否使用双向 LSTM
        dropout: dropout 概率
        """
        super(LSTMClassifier, self).__init__()
        self.num_features = channel * feature_len
        self.bidirectional = bidirectional
        # 这里输入尺寸为 1，因为我们将每个标量作为一个 token 处理
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # 若为双向 LSTM，则输出维度为 hidden_size*2
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_dim, output_dim)
    
    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, feature_len)
        返回: tensor, 形状为 (batch, output_dim)
        """
        B, C, D = x.shape
        # 将 (channel, feature_len) 展平成一个序列：(B, channel*feature_len, 1)
        x = x.view(B, self.num_features, 1)
        # 通过 LSTM 模型
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (B, seq_len, hidden_size*(1 or 2))
        # 这里我们取序列最后一个时间步的输出作为全局特征
        if self.bidirectional:
            # 如果是双向 LSTM，可将正向和反向的输出拼接
            # 注意：具体取哪一个时间步的输出需根据实际情况调整
            out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size],
                             lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        else:
            out = lstm_out[:, -1, :]
        # 最终全连接层分类
        out = self.fc(out)
        return out

class VoltageDNN(nn.Module):
    def __init__(self, n_buses):
        super(VoltageDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_buses, 32),  # Fully connected layer (4 -> 32)
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),  # Hidden layer (32 -> 16)
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),  # Output layer (16 -> 1)
        )

    def forward(self, x):
        return self.model(x)


class VoltageGNN(nn.Module):
    def __init__(self, num_buses, hidden_dim=16):
        super(VoltageGNN, self).__init__()
        self.conv1 = GATConv(num_buses, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, 1, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(1)

class VoltageResNet(nn.Module):
    def __init__(self, n_buses):
        super(VoltageResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.resnet(x)
        return x


# --------------------------
# 新增模型：分支网络，每个通道独立处理后再融合
class BranchNetworkClassifier(nn.Module):
    def __init__(self, feature_len, channel=1, hidden_dim=32, output_dim=1):
        """
        feature_len: 每个通道的特征数量
        channel: 输入通道数
        hidden_dim: 每个分支子网络的隐藏层单元数量
        output_dim: 输出维度
        """
        super(BranchNetworkClassifier, self).__init__()
        self.channel = channel
        # 子网络：对每个通道的特征进行独立处理，参数共享或不共享都可以，这里采用共享结构
        self.sub_net = nn.Sequential(
            nn.Linear(feature_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 最终分类器：将各个通道的输出拼接后再进行全连接层融合
        self.final_classifier = nn.Sequential(
            nn.Linear(channel * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: tensor, 形状为 (batch, channel, feature_len)
        返回: tensor, 形状为 (batch, output_dim)
        """
        batch_size, channels, feature_len = x.shape
        # 将每个通道的特征独立处理
        # reshape 成 (batch * channel, feature_len)
        x_reshaped = x.view(batch_size * channels, feature_len)
        branch_out = self.sub_net(x_reshaped)  # (batch * channel, hidden_dim)
        # 恢复为 (batch, channel * hidden_dim)
        branch_out = branch_out.view(batch_size, channels * branch_out.size(-1))
        # 最终分类
        out = self.final_classifier(branch_out)
        return out

if __name__ == '__main__':

    # Test PairwiseProduct
    x = torch.randn(2, 1, 4)
    model = PairwiseProd()
    print(x)
    print(model(x))
