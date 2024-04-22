import torch
import os
import logging
import json
import base64
from io import BytesIO
from dataclasses import dataclass
import numpy as np
from ipdb import set_trace
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from transformers import BertTokenizer
from time import *
from pdb import set_trace
import json as jsonmod


def _convert_to_rgb(image):
    return image.convert('RGB')


def build_transform(resolution=224):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
        Resize(resolution, interpolation=Image.BICUBIC),  # 扩大分辨率
        CenterCrop(resolution),  # 中心裁剪
        _convert_to_rgb,  # 转换成RGB三通道
        ToTensor(),
        normalize,
    ])






class iMiniUCM(Dataset):

    def __init__(self, task, task_num, train,split="val", max_txt_length=24):
        #super(iMiniUCM, self).__init__(train=train)
        self.transform = build_transform()
        
        self.image=[]
        self.caption=[]
        self.tt=[]
        self.td=[]
        self.tokenizers = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
        if split=='train':
            
            with open("./data/data"+str(task)+str(0)+"_5.txt") as f:
                for line in f:

                    self.image.append(line.strip('\n').split(',')[0])
                    # set_trace()
                    self.caption.append(line.strip('\n').split(',')[1])
                    self.tt.append(task_num)
                    self.td.append(task_num+1)
        if split=='val':
            with open("./data/data"+str(task)+str(1)+"_5.txt") as f:
                for line in f:
                    self.image.append(line.strip('\n').split(',')[0])
                    self.caption.append(line.strip('\n').split(',')[1])
                    self.tt.append(task_num)
                    self.td.append(task_num+1)

                
                
            


    def __getitem__(self, index):
        
        path = "./"+self.image[index].strip("'").strip("'")
        caption=self.caption[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # text = tokenize([str(caption)])

        token_ids = self.tokenizers.encode_plus(str(caption),
                                    padding="max_length",
                                    max_length=30,
                                    add_special_tokens=True,
                                    return_tensors='pt',
                                    return_attention_mask=True,
                                    truncation=True
                                    ) 

        tt, td = self.tt[index], self.td[index]


        return image, token_ids['input_ids'][0],token_ids['attention_mask'][0],tt, td


    def __len__(self):
        return len(self.image)

class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()


        self.batch_size=args.batch_size
        
        self.task_ids=[1,3,4,5,6]
       
        self.train_set = {}
        self.val_set = {}
        self.dataloaders = {}





    def get(self, task_id):

        self.dataloaders[task_id] = {}
 
        self.train_set[task_id] = iMiniUCM(task=self.task_ids[task_id],
                                                task_num=task_id, train=True,split='train', max_txt_length=24)

        self.val_set[task_id] = iMiniUCM(task=self.task_ids[task_id],
                                                task_num=task_id, train=False,split='val', max_txt_length=24)
        

        
        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=self.batch_size, num_workers=0,
                                                   pin_memory=True,shuffle=True,drop_last=True)
        train1_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=1, num_workers=0,
                                                   pin_memory=True,shuffle=False,drop_last=True)
        valid_loader = torch.utils.data.DataLoader(self.val_set[task_id], batch_size=1,
                                                   num_workers=0, pin_memory=True,shuffle=False,drop_last=False)
        


        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['replay'] = train1_loader
        self.dataloaders[task_id]['val'] = valid_loader
      
        print("Task ID: ", task_id)
        logging.info("______________data____________________:" + str(len(train_loader.dataset)))
        logging.info("______________data____________________:" + str(len(valid_loader.dataset)))

        return self.dataloaders



    
