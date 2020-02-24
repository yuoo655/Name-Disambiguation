import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import  MYModel
from torch.utils.data import Dataset, DataLoader
from gen_train_data import TripleData

device = 'cuda'
data = TripleData(device=device)
batch_size = 8
train_data = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
lr = 0.0001
hidden_dim = 2048
n_layers = 2
batch_ = 8

def try_lr_hidden(lr, hidden_dim , n_layers,batch_):
    hidden_dim = hidden_dim
    learning_rate = lr
    num_layers = n_layers
    batch_size = batch_
    model = MYModel(300, hidden_dim, batch_size, device, num_layers).to(device)
    optimizer = optim.Adam(params = model.parameters(), lr=learning_rate)
    model.train().to(device)
    model.hidden = model.init_hidden()
    model.zero_grad()
    optimizer.zero_grad()
    batch = next(iter(train_data))
    anc = batch[0].requires_grad_(True).to(device)
    pos = batch[1].requires_grad_(True).to(device)
    neg = batch[2].requires_grad_(True).to(device)
    ANC = model(anc)
    POS = model(pos)
    NEG = model(neg)
    ANC = torch.mean(ANC, dim=0)
    POS = torch.mean(POS, dim=0)
    NEG = torch.mean(NEG, dim=0)
    dis_p = F.pairwise_distance(ANC, POS, p=1)
    dis_n = F.pairwise_distance(ANC, NEG, p=1)
    sim_p = torch.exp(-dis_p)
    sim_n = torch.exp(-dis_n)
    sim_p = sim_p.mean()
    sim_n = sim_n.mean()
    loss = F.relu(sim_n - sim_p + 1)
    print('loss:{:.4f} sim_p:{:.4f}  sim_n:{:.4f}'.format(loss.item(),sim_p.item(), sim_n.item()))
    for i in range(200):
        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
        ANC = model(anc)
        POS = model(pos)
        NEG = model(neg)
        ANC = torch.mean(ANC, dim=0)
        POS = torch.mean(POS, dim=0)
        NEG = torch.mean(NEG, dim=0)
        dis_p = F.pairwise_distance(ANC, POS, p=1)
        dis_n = F.pairwise_distance(ANC, NEG, p=1)
        sim_p = torch.exp(-dis_p)
        sim_n = torch.exp(-dis_n)
        sim_p = sim_p.mean()
        sim_n = sim_n.mean()
        loss = F.relu(sim_n - sim_p + 1)
        print('loss:{:.4f} sim_p:{:.4f}  sim_n:{:.4f}'.format(loss.item(),sim_p.item(), sim_n.item()))
        if loss.item() < 0.1 and sim_p> 0.9:
            print('converge after', i)
            return i
            break
    if loss.item() > 0.1 or sim_p < 0.9:
        print('unconverge') 
        return 200

try_lr_hidden(0.00005, 2048,2,8)

def test(lr, hidden, n_layers):
    count = 0
    for i in range(50):
        tmp = try_lr_hidden(lr, hidden, n_layers)
        count += tmp
    avg = count/50    
    print('avg:{:.4f}'.format(avg))

test(0.00005, 1024, 2)