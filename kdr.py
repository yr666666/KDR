
import torch.nn
import os
import numpy as np
import copy
import json as jsonmod
import random

from scheduler import cosine_lr

from copy import deepcopy
from pdb import set_trace
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from torch.autograd import Variable
from transformers import BertTokenizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenize = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
def i2t( sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    sims=sims.detach().cpu().numpy()
    sims = np.array([sims[i] for i in range(0, len(sims), 5)])
    #set_trace()
    npts = sims.shape[0]
    ranks = np.zeros(npts) #N
    top1 = np.zeros(npts)
    for index in range(npts):

        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            #set_trace()
            tmp = np.where(inds == i)[0][0] #找到索引，即检索到的排位
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sims,  return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    sims = sims.detach().cpu().numpy()
    sims = np.array([sims[i] for i in range(0, len(sims), 5)])
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)






def valid1(model, val_loader,task_id,buffer):

    model.eval()
    #model : current trained model
    with torch.no_grad():
        img_emds = np.zeros((len(buffer), 768))
        cap_emds = np.zeros((len(buffer), 768))
        #img_emds = np.zeros((len(val_loader.dataset), 768))
        #cap_emds = np.zeros((len(val_loader.dataset), 768))
        ii=0
        for i, (image, text, attention_mask,tt, td) in enumerate(val_loader):
            #set_trace()
            # ii=0
            if i in buffer:    
                image = image.to(device)
                text = text.to(device)
                attention_mask=attention_mask.to(device)
                img_shared, txt_shared, img_pri, txt_pri, img_res, txt_res = model(
                    image, text,attention_mask, tt, td, task_id)
    
                img_emds[ii] = img_shared.data.cpu().numpy().copy()
                cap_emds[ii] = txt_shared.data.cpu().numpy().copy()
                ii+=1

        img_emds = torch.as_tensor(img_emds)
        cap_emds = torch.as_tensor(cap_emds)
        
    
        return buffer,img_emds,cap_emds
    
def valid(model, val_loader,task_id):

    model.eval()

    with torch.no_grad():
        img_emds = np.zeros((len(val_loader.dataset), 768))
        cap_emds = np.zeros((len(val_loader.dataset), 768))
        for i, (image, text,attention_mask, tt, td) in enumerate(val_loader):
            #set_trace()
            image = image.to(device)
            text = text.to(device)
            attention_mask=attention_mask.to(device)
            img_shared, txt_shared, img_pri, txt_pri, img_res, txt_res = model(
                image, text,attention_mask, tt, td, task_id)




            img_emds[i] = img_res.data.cpu().numpy().copy()
            cap_emds[i] = txt_res.data.cpu().numpy().copy()

        img_emds = torch.as_tensor(img_emds)
        cap_emds = torch.as_tensor(cap_emds)
        similarities = cosine_sim(img_emds, cap_emds)

        rsum = validate(similarities)
        return rsum


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    norm_im = torch.norm(im, 2, 1, True)
    im = torch.div(im, norm_im)
    norm_s = torch.norm(s, 2, 1, True)
    s = torch.div(s, norm_s)
    return im.mm(s.t())





def validate(sim):

    sim_i2t = sim
    sim_t2i = sim_i2t
    (r1, r5, r10, medr, meanr) = i2t(sim_i2t)

    (r1i, r5i, r10i, medri, meanri) = t2i(
        sim_t2i)

    score = r1 + r5 + r10 + r1i + r5i + r10i
    file = open('evaluation_test.txt', 'a')
    file.writelines(
        [str(r1), ' ', str(r5), ' ', str(r10), ' ', str(medr), ' ', str(meanr), '\n', str(r1i), ' ', str(r5i), ' ',
         str(r10i), ' ', str(medri), ' ', str(meanri), '\n', str(score), '\n'])


    print('r1=%f' % (r1))
    print('r5=%f' % (r5))
    print('r10=%f' % (r10))
    print('r1i=%f' % (r1i))
    print('r5i=%f' % (r5i))
    print('r10i=%f' % (r10i))
    score = r1 + r5 + r10 + r1i + r5i + r10i
    print('score=%f' % (score))

    return score,r1, r5, r10,r1i, r5i, r10i






def calcul_loss(scores, size, margin, loss_type="mse",max_violation=True, text_sim_matrix=None, param = "0.8 | 5"):
    diagonal = scores.diag().view(size, 1)

    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda(device)
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        # set_trace()
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()
























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

def get_dis_data(buffer,image,cap,img_rig,txt_rig,task_id):
    #                    #get raw data tensor and feature 
                        #buffer : id  dic for replay task data
                        #image : image feature dic
                        #cap: text feature dic
                        #img_rig:sample image file name dic
                        #txt_rig:sample caption name dic
                        #task_id:for sampling batch repaly for which task or tasks
    transform = build_transform()#image predeal
    root = "./"
    #batch buffer for using
    img_buf=[]
    cap_buf=[]
    img_rig_buf=[]
    txt_rig_buf=[]
    txt_rig_att_buf = []
    #label_buf=[]
    # set_trace()
    num=[]
    for i in range(500):  #buffer size
        num.append(i)

    ids=random.sample(num,8)  #buffer sample of a batch

    
    for j in range(task_id):
        for i in ids:
            # set_trace()
            img_buf.append(image[j][i])
            cap_buf.append(cap[j][i])
            #label_buf.append(label[0][i])
                       
            img0 = Image.open(os.path.join(root, img_rig[j][i].strip("'").strip("'"))).convert('RGB')
            
            img0 = transform(img0)
            
            token_ids = tokenize.encode_plus(str(txt_rig[j][i]),
                                        padding="max_length",
                                        max_length=30,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        return_attention_mask=True,
                                        truncation=True
                                        ) 
        
            txt0 = token_ids['input_ids'][0]
            txt0_attmask = token_ids['attention_mask'][0]
            
            img_rig_buf.append(img0)
            txt_rig_buf.append(txt0)
            txt_rig_att_buf.append(txt0_attmask)
            

     
    return img_buf,cap_buf,img_rig_buf,txt_rig_buf,txt_rig_att_buf
            
def element_rank_loss(features, embedding,features1, embedding1, tem):

    cosine_distance_l = 0.5 * (1 + torch.mm(features.detach(), embedding.detach().t()))  # cross-similarity of anchor and positive
    # cosine_distance_l = 0.5 * (1 + torch.mm(features, features.t()))  # cross-similarity of anchor and positive
    cosine_distance_h = 0.5 * (1 + torch.mm(features1, embedding1.t()))  # cross-similarity of anchor and positive
    
    W_h0 = torch.softmax(cosine_distance_h / tem, dim=0).t()
    W_l0 = torch.softmax(cosine_distance_l / tem, dim=0).t()
  
    cross_loss0 = torch.nn.functional.kl_div(W_l0.log(), W_h0, reduction='mean')

    knowledge_loss = cross_loss0.mean() 
    return knowledge_loss            
            
        
        
        
        
    
        
        
        


class KDR(object):
    def __init__(self, args,network):
        self.args = args
        self.device=device
        self.network = network
        self.mu = 0.0
        self.sigma = 1.0



    def train(self, task_id, dataset,model,old_model,buffer,image_buf,cap_buf,img_rig_buf,txt_rig_buf):
        self.model=model

        self.model.train()
        loss_img = torch.nn.CrossEntropyLoss().cuda(device)
        loss_txt = torch.nn.CrossEntropyLoss().cuda(device)
                
        lr_v=0.00001
        lr=0.00001  
        params=list(self.model.head.parameters())
        params+=list(self.model.imgprivate.parameters())
        params+=list(self.model.txtprivate.parameters())
        params+=list(self.model.imageembed0.parameters())
        self.optimizer_1 = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

     
        #share:
        params_=list(self.model.vit.parameters())
        params_+=list(self.model.bert_model_share.parameters())
        params_+=list(self.model.shared.parameters())
        params_+=list(self.model.imageembed.parameters())
        self.optimizer_2=torch.optim.Adam(params_, lr=lr_v, betas=(0.9, 0.999))

        best_rsum = 0
        best_epoch = 0
        total_steps = len(dataset['train']) * self.args.num_epochs
        for epoch in tqdm(range(self.args.num_epochs)):
            for batch_id,(image, text,attention_mask, tt, td) in enumerate(dataset['train']):



                tt=tt.cuda(device)
                td=td.cuda(device)
       
                image = image.to(device=self.device)
                text = text.to(self.device)
                attention_mask=attention_mask.to(self.device)

                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
    
                img_shared, txt_shared, img_pri, txt_pri, img_res, txt_res=self.model(image,text,attention_mask, tt, td, task_id)
                norm_im = torch.norm(img_res, 2, 1, True)
                img_res = torch.div(img_res, norm_im)

                norm_s = torch.norm(txt_res, 2, 1, True)
                txt_res = torch.div(txt_res, norm_s)
                #simlarity matrix
                logits_per_image = img_res @ txt_res.t()
                self.total_loss = calcul_loss(logits_per_image, 
                logits_per_image.shape[0], 
                margin=0.2, 
                max_violation=True,)
 
                
                # use buffer data and feature in each batch 
                if task_id>0:
                    #get raw data tensor and feature 
                    #buffer : id  dic for replay task data
                    #image_buf : image feature dic
                    #image_buf: text feature dic
                    #img_rig_buf:sample image file name dic
                    #txt_rig_buf:sample caption name dic
                    #task_id:for sampling batch repaly for which task or tasks
                    
                    image1,cap1,img_rig1,txt_rig1,txt_rig1_att=get_dis_data(buffer,image_buf,cap_buf,img_rig_buf,txt_rig_buf,task_id) #from buffer get train pre-data
                    # set_trace()
                    image1= torch.tensor([item.cpu().detach().numpy() for item in image1]).cuda(device)  #feature
                    cap1= torch.tensor([item.cpu().detach().numpy() for item in cap1]).cuda(device)
                    img_rig1= torch.tensor([item.cpu().detach().numpy() for item in img_rig1]).cuda(device)#raw data
                    txt_rig1= torch.tensor([item.cpu().detach().numpy() for item in txt_rig1]).cuda(device)
                    txt_rig1_att= torch.tensor([item.cpu().detach().numpy() for item in txt_rig1_att]).cuda(device)
                    img_shared1, txt_shared1, _, _, _,_,=self.model(img_rig1,txt_rig1,txt_rig1_att, tt, td, task_id)
                
                    loss1=0.1*element_rank_loss(image1,cap1,img_shared1,txt_shared1,1)
                    self.total_loss+=loss1


                self.total_loss.backward()
                self.optimizer_1.step()
                self.optimizer_2.step()
                

                if batch_id % 100 ==0:
                    print('epoch=%i,batch_id=%i,total_loss=%f'%(epoch,batch_id,self.total_loss))
       
            if epoch % 1 == 0:
                print("VAL")
                rsum,r1, r5, r10,r1i, r5i, r10i= valid(self.model, dataset['val'],task_id)
            if rsum > best_rsum:
                best_epoch=epoch
                best_model = deepcopy(self.model.state_dict())
                best_rsum = max(rsum, best_rsum)
            print('best_epoch=%i'%(best_epoch))
        
        old_model=0
        self.model.load_state_dict(copy.deepcopy(best_model))
        model_cpt = deepcopy(self.model.state_dict())

        torch.save({'model_state_dict': model_cpt,
                   }, os.path.join('./checkpoints/', 'model_{}.pth.tar'.format(task_id)))

        return self.model,old_model

    def load_checkpoint(self, task_id):
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a previous model
        net = self.network.Net(self.args)
        checkpoint0 = torch.load(os.path.join('./checkpoints/', 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint0['model_state_dict'])
        checkpoint1 = torch.load(os.path.join('./checkpoints/', 'vit_{}.pth.tar'.format(task_id)))
        net.vit.load_state_dict(checkpoint1['model_state_dict'])
        checkpoint2 = torch.load(os.path.join('./checkpoints/', 'bert_{}.pth.tar'.format(task_id)))
        net.bert.load_state_dict(checkpoint2['model_state_dict'])
        checkpoint3 = torch.load(os.path.join('./checkpoints/', 'imgemd_{}.pth.tar'.format(task_id)))
        net.imageembed.load_state_dict(checkpoint3['model_state_dict'])
        net = net.to(self.device)
        return net

    def load_model(self, task_id):

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join('./checkpoints/', 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        current_shared_module = deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)
        current_vit = deepcopy(self.model.vit.state_dict())
        net.vit.load_state_dict(current_vit)
        current_bert = deepcopy(self.model.bert.state_dict())
        net.bert.load_state_dict(current_bert)
        current_imgemd = deepcopy(self.model.imageembed.state_dict())
        net.imageembed.load_state_dict(current_imgemd)


        net=net.to(self.device)
        return net
    def save_all_models(self, task_id):
        print("Saving all models for task {} ...".format(task_id+1))


        model=utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def test(self, data_loader, task_id, model):
        
        rsum,r1, r5, r10,r1i, r5i, r10i = valid(model, data_loader, task_id)
        return rsum,r1, r5, r10,r1i, r5i, r10i
    
    #get buffer for every task
    def get_buffer(self, data_loader, task_id, model):
        list1=[]
    #get a list from 0 to len(dataset)
        for i in range(len(data_loader.dataset)):
                list1.append(i)
        random.seed(1)
    #randomly sample 100 ids
        buffer=random.sample(list1,500)  #buffer size
    #use buffer to get image and text feature
        buf,img,txt = valid1(model, data_loader, task_id,buffer)
        return buf,img,txt



