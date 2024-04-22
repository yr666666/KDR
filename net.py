import numpy as np
import torch
import shutil
from torch._C import set_autocast_enabled
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import torch.nn
import torch.backends.cudnn as cudnn
import timm
from timm import create_model as creat
from torch.autograd import Variable
from pdb import set_trace
from scheduler import cosine_lr
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class bertmodel(BertModel):
    def __init__(self, config,add_pooling_layer=True):
        super().__init__(config,add_pooling_layer)

    def forward(
            self,
            input_ids=None,
            image_feature=None,
            attention_mask=None,
            key=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:  # 64*40
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if key == False:
            # set_trace()
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]



            # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return sequence_output

class shared(torch.nn.Module):
    def __init__(self,args):
        super(shared, self).__init__()

        #self.img=img
        self.dim1=768
        self.dim2=768
        self.encoded=torch.nn.Sequential(
                torch.nn.Linear(self.dim1, self.dim2),
        )
    def forward(self,img):
        # set_trace()
        return self.encoded(img.type(torch.float))

class imageprivate(torch.nn.Module):
    def __init__(self,args):
        super(imageprivate, self).__init__()

        #self.img=img
        self.dim1=768
        self.dim2=768
        self.num_tasks = 5
        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):

            self.encoded=torch.nn.Sequential(
                    torch.nn.Linear(self.dim1, self.dim2),
                    torch.nn.Sigmoid()
            )
            self.task_out.append(self.encoded)
    def forward(self,img,task_id):
        return self.task_out[task_id].forward(img.type(torch.float))

class txtprivate(torch.nn.Module):
    def __init__(self,args):
        super(txtprivate, self).__init__()

        #self.img=img
        self.dim1=768
        self.dim2=768
        self.num_tasks = 5
        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):

            self.encoded=torch.nn.Sequential(
                    torch.nn.Linear(self.dim1, self.dim2),
                    torch.nn.Sigmoid()
            )
            self.task_out.append(self.encoded)
    def forward(self,txt,task_id):
        return self.task_out[task_id].forward(txt.type(torch.float))

class imageembed0(torch.nn.Module):
    def __init__(self,args):
        super(imageembed0, self).__init__()

        #self.img=img
        self.dim1=1000
        self.dim2=768
        self.num_tasks = 5
        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.encoded=torch.nn.Sequential(
                torch.nn.Linear(self.dim1, self.dim2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.dim2, self.dim2),
                torch.nn.LeakyReLU(),
        )
            self.task_out.append(self.encoded)
        
    def forward(self,img,task_id):
        img=img.view_as(img)
        out=self.task_out[task_id].forward(img)
        return out

class imageembed(torch.nn.Module):
    def __init__(self,args):
        super(imageembed, self).__init__()

        #self.img=img
        self.dim1=1000
        self.dim2=768
        
        self.encoded=torch.nn.Sequential(
                torch.nn.Linear(self.dim1, self.dim2),
        )
    def forward(self,img):
        return self.encoded(img)


class head(torch.nn.Module):
    def __init__(self,args):
        super(head, self).__init__()
        
        self.dim1 = 768
        self.dim2 = 768
        self.num_tasks = 5
        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.task_out.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.dim1, self.dim2),
                    # torch.nn.Linear(self.dim2, self.dim2),
                    # torch.nn.Dropout(0.3),
                    # torch.nn.LeakyReLU(),
                    #torch.nn.Linear(self.dim2, self.dim2),
                ))
    def forward(self,img,task_id):
        img=img.view_as(img)
        out=self.task_out[task_id].forward(img)
        return out




class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        # self.taskcla = args.taskcla
        self.num_tasks = 3
        self.samples = 21
        self.args = args


        self.device = device
        self.shared = shared(args).to(self.device)
        ###
        self.imgprivate = imageprivate(args).to(self.device)
        self.txtprivate = txtprivate(args).to(self.device)
        ###
        # self.imgtxtprivate = imageprivate(args).to(self.device)
        self.head = head(args).to(self.device)
        #self.txthead = txthead(args).to(self.device)

        self.l2norm = L2Norm()


        
        self.vit = creat('vit_base_patch16_224',  checkpoint_path="./checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",pretrained=True, num_classes=1000)
        self.vit_pri=creat('vit_base_patch16_224',checkpoint_path="./checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz", pretrained=True, num_classes=1000)
        self.imageembed = imageembed(args).to(self.device)
        self.imageembed0 = imageembed0(args).to(self.device)

        self.bert_name = './bert-base-uncased'
        bert_config = BertConfig.from_pretrained(self.bert_name)
        self.bert_model_pri = bertmodel.from_pretrained(self.bert_name,config=bert_config)
        self.bert_model_share = bertmodel.from_pretrained(self.bert_name,config=bert_config)
        bert_vocab = self.bert_name.join('/vocab.txt')
        self.tokenizers = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
   
        
        if torch.cuda.is_available():
            
            self.vit.cuda(self.device)
            self.bert_model_pri.cuda(self.device)
            self.bert_model_share.cuda(self.device)
            self.vit_pri.cuda(self.device)
            
            cudnn.benchmark = True
        for param in self.bert_model_pri.parameters():
            param.requires_grad=False
        for param in self.vit_pri.parameters():
            param.requires_grad=False
       

    def forward(self, image, caption,attention_mask, tt, td, task_id):
        if torch.cuda.is_available():
            images = image.cuda(self.device)
            captions = caption.cuda(self.device)
       
        # set_trace()
        captions = torch.squeeze(captions,1)
        # set_trace()
        txtoutput_pri = self.bert_model_pri(input_ids=captions, attention_mask=attention_mask, key=False)
        txtoutput_pri = txtoutput_pri[:, 0, :]
        txtoutput_share = self.bert_model_share(input_ids=captions, attention_mask=attention_mask, key=False)
        txtoutput_share = txtoutput_share[:, 0, :]



        imgoutput = self.vit(images)
        imgoutput_share = self.imageembed(imgoutput)
        imgoutput0=self.vit_pri(images)
        imgoutput_pri=self.imageembed0(imgoutput0,task_id)
        
        img_shared = self.shared(imgoutput_share)
        txt_shared = self.shared(txtoutput_share)
        img_pri = self.imgprivate(imgoutput_pri,task_id)
        txt_pri = self.txtprivate(txtoutput_pri,task_id)

    
        imgfus = img_shared * img_pri
        txtfus = txt_shared * txt_pri
        img_res = self.head(imgfus,task_id)
        txt_res = self.head(txtfus,task_id)

  
        
        return img_shared,txt_shared,img_pri, txt_pri, img_res, txt_res

