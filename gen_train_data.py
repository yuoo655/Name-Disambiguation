import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TripleData(Dataset):
    def __init__(self, batch = 1, device = 'cpu'):
        super(TripleData).__init__()
        self.anc = np.load('data/train/emb_data/anc_300.npy')
        self.pos = np.load('data/train/emb_data/pos_300.npy')
        self.neg = np.load('data/train/emb_data/neg_300.npy')

        self.anc = torch.from_numpy(self.anc).float()
        self.pos = torch.from_numpy(self.pos).float()
        self.neg = torch.from_numpy(self.neg).float()

        # self.anc = self.anc.clone().detach().requires_grad_(True)
        # self.pos = self.pos.clone().detach().requires_grad_(True)
        # self.neg = self.neg.clone().detach().requires_grad_(True)

    def __len__(self):
        return len(self.anc)

    def __getitem__(self, index):
        
        anc_ = self.anc[index]
        pos_ = self.pos[index]
        neg_ = self.neg[index]

        return anc_, pos_, neg_

