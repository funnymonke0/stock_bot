import torch
import torch.nn as nn

class StockModel(nn.Module):
    def __init__(self, feature_size, embed_size, embedding_dims = 8, hidden_layers = [64,32], dropout=0.3, output_size=3):
        super(StockModel, self).__init__()

        self.ticker_embedding = nn.Embedding(num_embeddings=embed_size, embedding_dim=embedding_dims)
        next_size = embedding_dims + feature_size

        self.sequential_layers = nn.ModuleList()
        for hidden_size in hidden_layers:
            self.sequential_layers.append(nn.Linear(next_size, hidden_size))
            self.sequential_layers.append(nn.BatchNorm1d(hidden_size))
            self.sequential_layers.append(nn.ReLU())
            self.sequential_layers.append(nn.Dropout(dropout))
            next_size = hidden_size
        self.sequential_layers.append(nn.Linear(next_size, output_size))
        # self.sequential_layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.sequential_layers)

    def forward(self, x_id, x_features):
        x_embed = self.ticker_embedding(x_id) #embed ticker ids
        x = torch.cat((x_embed, x_features), dim=1) #concatenate embeddings with other features and flatten to 1 dimension
        return self.model(x)
    
