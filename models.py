import torch
import torch.nn.functional as F
import torch_geometric as pyg

class GCN_DP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
class GCN_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(output_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        return self.mlp(edge_features).view(-1)
        
class GraphSAGE_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = pyg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = pyg.nn.SAGEConv(hidden_dim, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(output_dim)
        self.dropout = dropout

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(output_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        return self.mlp(edge_features).view(-1)
    
class GIN_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = pyg.nn.GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * hidden_dim),
            torch.nn.BatchNorm1d(2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim)
        ), train_eps=True)
        self.conv2 = pyg.nn.GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 2 * hidden_dim),
            torch.nn.BatchNorm1d(2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, output_dim)
        ), train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(output_dim)
        self.dropout = dropout

        # Define MLP for decoding
        self.mlp_decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        return self.mlp_decoder(edge_features).view(-1)