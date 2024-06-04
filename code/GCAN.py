import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads

        # Initialize weights for each attention head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, self.head_dim))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim))

        # Bias parameters
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # Initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: input node features (N x in_features)
        # adj: adjacency matrix (N x N)

        N = x.size(0)  # Number of nodes
        heads_out = []

        for i in range(self.num_heads):
            # Linear transformation
            Wx = torch.matmul(x, self.W[i])  # W_i * x

            # Attention mechanism
            a_input = torch.cat([Wx.repeat(1, N).view(N * N, -1),
                                 Wx.repeat(N, 1)], dim=1).view(N, -1, 2 * self.head_dim)  # [N,N,2*head_dim]
            e = torch.matmul(a_input, self.a[i]).squeeze()  # [N,N]
            e = F.leaky_relu(e, negative_slope=0.2)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)  # [N,N]

            # Apply attention to the input features
            h_prime = torch.matmul(attention, Wx)  # [N,head_dim]
            heads_out.append(h_prime)

        # Concatenate outputs of all heads
        out = torch.cat(heads_out, dim=1)  # [N, num_heads * head_dim]

        # Add bias term
        out += self.bias

        return F.relu(out)


class GCN_GAT(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_heads):
        super(GCN_GAT, self).__init__()
        self.gat_layer1 = GraphAttentionLayer(in_features, hidden_dim, num_heads)
        self.gat_layer2 = GraphAttentionLayer(hidden_dim * num_heads, out_features, 1)
        self.adaptive_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))  # Initial weights for adaptive fusion

    def forward(self, x, adj):
        gat_out1 = self.gat_layer1(x, adj)
        gat_out2 = self.gat_layer2(gat_out1, adj)

        # Adaptive fusion
        fused_out = self.adaptive_weights[0] * gat_out1 + self.adaptive_weights[1] * gat_out2

        return F.log_softmax(fused_out, dim=1)