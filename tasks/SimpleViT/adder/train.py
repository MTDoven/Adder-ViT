import sys
sys.path.append(r"D:\Project\tangdongwen")
from operation.adder import OperateBase, ConvBase
from model.simple_vit import get_model
from data import get_cifar10 as get_data

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import torch
import time
import math
import os

# random.seed(20230926)
# torch.manual_seed(20230926)
# torch.cuda.manual_seed_all(20230926)



if __name__ == "__main__":

############### Multi#############
    # parameters
    name = "exp"
    project = "1024"
    to_log = True
    num_workers = 2
    start_epoch = 1
    epoches = 100
    batchsize = 64
    use_amp = False
    learning_rate = 0.0002
    weight_decay = 0.002
    img_size = 128
    resume = False
    
    # prepare logger
    if to_log:
        import wandb
        wandb.login()
        wandb.init(project=project, resume=resume)
    # init device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # init model
    # image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64           
    
    net = get_model(operate_base=OperateBase,
                    image_size = img_size, 
                    patch_size = 16, 
                    num_classes = 10,
                    dim = 128,
                    depth = 4,
                    heads = 4,
                    mlp_dim = 128,
                    channels = 3,
                    dim_head = 32,
                    ).to(device)
    
    # Prepare dataset
    trainloader, testloader, *_ = get_data(img_size, batchsize, num_workers)
    # use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    mseloss = nn.SmoothL1Loss(reduction='mean')
    # use Adam optimizer
    optimizer = optim.AdamW([{'params': net.parameters(), 'initial_lr':learning_rate}], 
                          lr=learning_rate, weight_decay=weight_decay)#, momentum=0.9)
    # use cosine scheduling
    T=epoches; t=5
    lambda0 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1 \
                if 0.5 * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else 0.5 * (1+math.cos(math.pi*(epoch - t)/(T-t)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0, last_epoch=start_epoch-1)
    # use amp autocast
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # resume
    if resume:
        checkpoint = torch.load("exp_498.state.pt")
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.100001-0.002*x )



# Training
def train():
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs= net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # log and save
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(batch_idx+1), correct/len(trainloader.dataset)

# Validation
best_acc = 0.
def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, "./checkpoint/{}_{}.state.pt".format(name,epoch))
        best_acc = acc
    return test_loss/(batch_idx+1), correct/len(testloader.dataset)

def to_log_local(**kwargs):
    string = str(kwargs)
    print(string)
    with open("./log.txt","a") as f:
        f.write(string+"\n")

if __name__=="__main__":
    
    for epoch in range(start_epoch, epoches+1):
        start = time.time()
        # train one epoch
        train_loss, train_acc = train()
        # test one epoch
        val_loss, val_acc = test()
        # scheduler step
        scheduler.step()
        # log and save
        to_log_local(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
        to_log_local(epoch=epoch, val_loss=val_loss, val_acc=val_acc)
        if to_log:
            wandb.log({'epoch': epoch,
                       'train_loss': train_loss,
                       'train_acc': train_acc,
                       'val_loss': val_loss,
                       "val_acc": val_acc,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time()-start, })