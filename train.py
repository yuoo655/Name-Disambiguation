import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import  MYModel
from gen_train_data import TripleData
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

writer = SummaryWriter('log')
device = 'cuda'
batch_size = 8
hidden_dim = 2048
emb_dim = 300
num_layers = 2
lr = 0.00005  #0.00085 0.0001 0.0005

data = TripleData(device=device)
train_data = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
model = MYModel(emb_dim, hidden_dim, batch_size, device, num_layers).to(device)

optimizer = optim.Adam(params = model.parameters(), lr=lr)
scheduler  = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr = 0.00009, div_factor=2, final_div_factor=4, epochs=10,steps_per_epoch=len(train_data), pct_start=0.3)
model.train().to(device)

for epoch in range(10):
    model.hidden = model.init_hidden()
    count = 0
    avg_sim_p = 0
    avg_sim_n = 0
    start_time = time.time()
    for i, batch in enumerate(train_data):
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
        # print('loss:{:.4f} sim_p:{:.4f}  sim_n:{:.4f}'.format(loss.item(),sim_p.item(), sim_n.item()))
        avg_sim_p += sim_p.item()
        avg_sim_n += sim_n.item()
    
        count += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()

        if i % 100 == 0:
            end_time = time.time()
            speed = round(end_time - start_time, 2)
            start_time = time.time()
            print('Epoch:{} [{}/{}] avg loss:{:.4f} {:.1f}sec/100  sim_p:{:.4f} sim_n:{:.4f} '
                    .format(epoch, i, len(train_data),  count/(100), speed ,avg_sim_p/100 , avg_sim_n/100))
            writer.add_scalar('loss', count/100, epoch*len(train_data)+i)
            writer.add_scalar('sim pos', avg_sim_p/100, epoch*len(train_data)+i)
            writer.add_scalar('sim neg', avg_sim_n/100, epoch*len(train_data)+i)
            avg_sim_p = 0
            avg_sim_n = 0
            count = 0
print('done') 

path = 'model/' + str(time.strftime("%m-%d %H_%M_%S", time.localtime()))
all_path = 'model/' + str(time.strftime("%m-%d %H_%M_%S", time.localtime())) +'.all'
torch.save(model.state_dict(), path)
torch.save(model,  all_path)


