import torch
import torch.nn as nn
import torch.nn.functional as F


class MYModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_, device = 'cuda', n_layers=2):
        super(MYModel,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_ = batch_
        self.device = device
        self.num_layers = n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.batch_)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, self.batch_, self.hidden_dim, device=self.device))

    def forward(self, x):
        # input_ = F.normalize(input_,p=2, dim=2)
        x1, h1 = self.lstm(x, self.hidden)
        fc = self.fc1(x1)
        return fc


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output