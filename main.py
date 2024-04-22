import os
import time
import os
import torch
import numpy as np
import argparse
import logging
import torchvision.transforms as transforms
import random
from pdb import set_trace
from scheduler import cosine_lr
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
from timm import create_model as creat
from kdr import KDR as approach
import net as network
import json as jsonmod
import data
from transformers import BertTokenizer
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
setup_seed(0)



def get_buf_data(buffer,task_id):
    
    imgbuff=[]
    txtbuff=[]
    labelbuff=[]
    img=[]
    txt=[]
    task=[1,3,4,5,6]
        
      
            
    with open("./data/data"+str(task[task_id])+str(0)+"_5.txt") as f:
        for line in f:

            img.append(line.strip('\n').split(',')[0])
         
            txt.append(line.strip('\n').split(',')[1])
         
    for i in buffer[task_id]:
        imgbuff.append(img[i])
        txtbuff.append(txt[i])    
    

       
    return imgbuff,txtbuff


def main():
    # Hyper Parameters

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    dataloader = data.DatasetGen(args)
    #set_trace()
    
    
    model = network.Net(args)
    appr = approach(args,network)
    old_model = None
    buffer={}
    image={}
    cap={}
    img_rig={}
    txt_rig={}
    label={}
    
    for t in tqdm(range(5)): #tasks count 
        torch.set_num_threads(1)
        dataset = dataloader.get(t)

        model, _ = appr.train(t, dataset[t], model, old_model,buffer,image,cap,img_rig,txt_rig)

    
        #sample current task data and feature  
        #buf: id   img:image feature according id  txt:text feature according id
        buf,img,txt= appr.get_buffer(dataset[t]['replay'], t, model=model)
        #inject id and feature into dic
        
        buffer[t]=buf
        image[t]=img
        cap[t]=txt

        #sample raw data name and caption from dataset
        img_rig[t],txt_rig[t]=get_buf_data(buffer,t)
        
        for u in range(t+1):
            #set_trace()
            # test_model = appr.load_model(t)
            print('TEST')
            test_res,r1, r5, r10,r1i, r5i, r10i = appr.test(dataset[u]['val'], u, model=model)
            file = open('test.txt', 'a')
            file.writelines(
                [str(t), ' ', str(u), '\n', str(test_res), '\n'])
            file.writelines(
                [str(r1), ' ', str(r5), ' ', str(r10), '\n', str(r1i), ' ', str(r5i), ' ',
                 str(r10i), '\n'])



if __name__ == '__main__':
    
    main()




