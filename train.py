import torch
from network import SGDVA
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import math

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

Dataname = 'Cifar100'
parser = argparse.ArgumentParser(description='train')
parser.add_argument("--seed", type=int, default=10) 
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f",type=float, default=0.5)
parser.add_argument("--temperature_l",type=float, default=0.6)
parser.add_argument("--learning_rate",type=float, default=0.0003)
parser.add_argument("--weight_decay",type=float, default=0.)
parser.add_argument("--workers",type=int, default=8)
parser.add_argument("--rec_epochs", type=int,default=100)
parser.add_argument("--fine_tune_epochs",type=int, default=100)
parser.add_argument("--low_feature_dim",type=int, default=512)
parser.add_argument("--high_feature_dim",type=int, default=128)

parser.add_argument("--walk_steps",type=int, default=4)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--gama", type=float, default=0.05)
parser.add_argument("--warmup_epochs", type=int, default=110)
parser.add_argument("--warmup_robust_instance", type=int, default=130)
parser.add_argument("--log_file", type=str, default="search_log.csv")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Cifar100":
    args.learning_rate = 0.0008
    args.alpha = 1.0
    args.beta = 10.0
    args.gama = 0.01
    args.walk_steps = 2
    args.fine_tune_epochs = 20
    seed = 10

alpha = args.alpha
beta = args.beta
gama = args.gama
warmup_epochs = args.rec_epochs +10
warmup_robust_instance = args.rec_epochs +30
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def robust_scale(epoch):
    
    if epoch <= args.rec_epochs:
        return 0.0
    if epoch >= warmup_robust_instance:
        return 1.0
    denom = max(1, warmup_robust_instance - args.rec_epochs)
    progress = (epoch - args.rec_epochs) / denom  
    ramp_cos = 0.5 * (1 - math.cos(math.pi * progress))
    return ramp_cos
def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _ = model(xs)
        loss_list = []                                                                                  
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def get_attention_temperature(epoch, warmup_epochs=110, start=2.0, end=1.0, ramp=40):
    if epoch <= warmup_epochs:
        return start
    if epoch >= warmup_epochs + ramp:
        return end
    t = (epoch - warmup_epochs) / ramp
    return start + (end - start) * t

def fine_tune(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    ramp = robust_scale(epoch)
    robust_coef = gama * (1e-4 + (1 - 1e-4) * ramp)
    temperature = get_attention_temperature(epoch, warmup_epochs)
    for batch_idx, (xs, _, _) in enumerate(data_loader):   
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, hs = model(xs)
        commonz, S ,weight= model.SGDVA(xs,epoch,temperature,warmup_epochs)
        
       
        loss_list = []
        
        loss_list.append(robust_coef * criterion.forward_robust_instance_contrast(commonz.detach(), commonz))
        for v in range(view):
            view_weight = weight[:, v]

            if epoch < warmup_epochs:
                loss_inst_vec = criterion.forward_instance(commonz, hs[v], reduction='mean')
                loss_list.append(alpha*loss_inst_vec)

                
            else:
               loss_inst_vec = criterion.forward_instance(commonz, hs[v], reduction='none')
               weighted_loss_inst = (view_weight * loss_inst_vec).mean()
               loss_list.append(alpha * weighted_loss_inst)

            loss_sgc_vec = criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S)
            loss_list.append(beta* loss_sgc_vec)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    
if not os.path.exists('./models'):
    os.makedirs('./models')

model = SGDVA(view, dims, args.low_feature_dim, args.high_feature_dim, device)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, args.temperature_l, args.temperature_f,args.walk_steps, device).to(device)
epoch = 1

accs = []
nmis = []
purs = []
best_acc, best_nmi, best_pur = 0, 0, 0
while epoch <= args.rec_epochs:
    pre_train(epoch)
    epoch += 1


while epoch <= args.rec_epochs + args.fine_tune_epochs:
    ft_loss =fine_tune(epoch)
    temperature = get_attention_temperature(epoch, warmup_epochs)
    acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num,epoch,temperature,warmup_epochs)
    
    if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
    #         state = model.state_dict()
    #         torch.save({
    #         "epoch": epoch,
    #         "temperature":temperature,
    #         "warmup_epochs": warmup_epochs,
    #         "state_dict": model.state_dict(),
    #     }, f"./models/{args.dataset}.pth")
    epoch += 1
accs.append(best_acc)
nmis.append(best_nmi)
purs.append(best_pur)
print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
