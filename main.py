import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import yaml

from model.resnet50 import CB
import tensorboardX as tx

import torch
from torch import nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader

from utils import *
from data.imagedata import img_attr_label

csv_attributions= ['Male','Young', 'Middle_Aged', 'Senior','Asian','White','Black','Rosy_Cheeks',
                'Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns',
                'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','No_Beard','Mustache',
                '5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin',
                'High_Cheekbones','Chubby','Obstructed_Forehead', 'Fully_Visible_Forehead','Brown_Eyes',
                'Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows','Mouth_Closed','Smiling',
                'Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup','Wearing_Hat','Wearing_Earrings',
                'Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']

target_attributions= ['Male','Young', 'Middle_Aged', 'Senior','Asian','White','Black','Rosy_Cheeks',
                'Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns',
                'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','No_Beard','Mustache',
                '5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin',
                'High_Cheekbones','Chubby','Obstructed_Forehead', 'Fully_Visible_Forehead','Brown_Eyes',
                'Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows','Mouth_Closed','Smiling',
                'Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup','Wearing_Hat','Wearing_Earrings',
                'Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']

def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default='/media/lijia/系统/data/train_align')
    parse.add_argument('--maad_path',type=str,default='/media/lijia/系统/data/vggface2/MAAD_Face.csv')
    parse.add_argument('--save_path',type=str,default='/home/lijia/codes/202210/cbw-face/checkpoints')
    parse.add_argument('--train_csv',type=str,default='train_id.csv')
    parse.add_argument('--train_mode',choices=['independent','sequential','joint','standard'])
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('-lr',type=float,default=0.01)
    parse.add_argument('--epoch',type=int,default=20)
    parse.add_argument('--idclass',type=int,default=8631)
    args=parse.parse_args()
    return args

def loadimage(args):
    maadfile=pd.read_csv(args.maad_path)
    train_csv=pd.read_csv(args.train_csv)

    csvfile=pd.merge(train_csv,maadfile,on='Filename')
    del maadfile
    del train_csv
    csvfile.index=csvfile['Filename']
    idfile=csvfile['id'].to_frame()
    csvfile=csvfile[target_attributions]
    csvfile[csvfile==-1]=0

    assert len(csvfile)==len(idfile)
    imagedataset=img_attr_label(args.image_dir,csvfile,idfile)
    train_size=int(0.8*len(imagedataset))
    test_size=len(imagedataset)-train_size
    train_dataset,test_dataset=torch.utils.data.random_split(imagedataset,[train_size,test_size])
    train_dl=DataLoader(train_dataset,args.batch_size)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl

def main():
    # 加载
    args=makeargs()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    writer=tx.SummaryWriter('/home/lijia/codes/202210/cbw-face/log')

    # 读取数据
    train_dl,test_dl=loadimage(args)

    #读取模型
    model=CB(len(target_attributions),args.idclass)
    model.load_state_dict(torch.load('/home/lijia/codes/202210/cbw-face/checkpoints/0_fc.pth.tar')['state_dict'])
    optimizer=torch.optim.SGD(model.parameters(),args.lr,momentum=0.9)
    bce=nn.BCELoss()
    cel=nn.CrossEntropyLoss()
    weight=1
    model.to(device)

    bs=0
    for e in range(args.epoch):
        losses=0
        timelosses=0
        model.train()
        for d in tqdm(train_dl):
            bs += 1
            c,y=model(d[0].to(device))
            loss=weight*bce(c,d[1].to(device))+cel(y,d[2].to(device))
            optimizer.zero_grad()
            losses=losses+loss
            timelosses+=loss
            loss.backward()
            optimizer.step()
            if bs%2000==0:
                writer.add_scalar('loss/train',timelosses,int(bs/2000))
                timelosses=0
        writer.add_scalar('loss/traine',losses,e)
        torch.save({'epoch': e, 'state_dict': model.state_dict()},
                   os.path.join(args.save_path, str(e) + '_cb.pth.tar'))
        total=0
        corr=0
        model.eval()
        for d in tqdm(test_dl):
            _,y=model(d[0].to(device))
            _,label=torch.max(y,1)
            total=total+label.size()[0]
            corr+=(d[2].to(device)==label).sum()
        print(float(corr)/float(total))
        writer.add_scalar('acc/test',float(corr)/float(total),e)

main()