import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from utils import ManualLogger
import json
import logging
import gc
import random
import pickle
from argparse import ArgumentParser


#global variables
base_dir = os.path.dirname(os.path.realpath(__file__))
logger = ManualLogger('bert-logger', os.path.join(base_dir, 'optimized_accumlate_gd_rnn_xlmr.log'), use_stdout=False)
BATCH_SIZE=10

# implemented graph Neural network similar to https://arxiv.org/abs/1710.10903
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, nonlinear_fn=torch.tanh):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.nonlinear_fn = nonlinear_fn

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1)) #node_num * node_num
        assert N==e.shape[0] and N==e.shape[1]

        # zero_vec = -9e15*torch.ones_like(e)  #assign minimum value for attention calculation
        #adj = torch.sigmoid(adj)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.sigmoid(adj)*F.softmax(e, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return self.nonlinear_fn(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nonlinear_func, use_concat):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.concat = use_concat
        self.nonlinear_func = nonlinear_func
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=use_concat, nonlinear_fn=self.nonlinear_func) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        x = self.dropout(x)
        if self.concat:
          x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
          x = torch.stack([att(x, adj) for att in self.attentions],  dim=1).mean(dim=1)
          x = self.nonlinear_func(x)
        x = self.dropout(x)
        return x

class GNNStack(nn.Module):
    """
    hidden dimensio should be divisible by attention heads
    [ hdim % nheads == 0 ]
    """
    def __init__(self, layer_count, nfeat, nhid, dropout, alpha, nheads, nonlinear_func, internal_layer_concat=True, last_layer_concat=False):
          super(GNNStack, self).__init__()
          use_concat = last_layer_concat if layer_count==1 else internal_layer_concat
          internal_node_size = nhid
          if internal_layer_concat:
            assert nhid % nheads == 0  
            internal_node_size = int(nhid/nheads)
          self.GNNs = [GAT(nfeat, internal_node_size, dropout, alpha, nheads, nonlinear_func, use_concat)]
          for i in range(1, layer_count):
              if i != layer_count-1:
                  self.GNNs.append(GAT(nhid, internal_node_size, dropout, alpha, nheads, nonlinear_func, internal_layer_concat))
              else:
                  self.GNNs.append(GAT(nhid, internal_node_size, dropout, alpha, nheads, nonlinear_func, last_layer_concat))
          #register with model
          for i, gnn in enumerate(self.GNNs):
              self.add_module('GAT_{}'.format(i), gnn)

    def forward(self, x, adj):
        for gnn in self.GNNs:
            x = gnn(x, adj)
        return x

def multilabel_accuracy(output, labels, thresholds=None):
    if thresholds is None:
        thresholds = torch.tensor([0.5]*output.shape[1]).cpu()
    output = torch.sigmoid(output).cpu()
    final_classes = torch.zeros_like(output)
    final_classes[output>=thresholds]=1
    return accuracy_score(labels.cpu(), final_classes.cpu())

class LitModel(pl.LightningModule):

    def __init__(self, rnn_hidden_dim, hidden_dim, num_class, layers=1, dropout_rate=0.5, bi_dir=False, emb_train=True,
               graph_layer_count=2, graph_hidden_size=128, graph_node_size=32, alpha=0.2, nheads=2, nonlinear_func=torch.tanh,
               bert_option_name='xlm-roberta-large', bert_active_layers=[23]):
        super().__init__()
        self.hidden_dim = rnn_hidden_dim
        self.layer_count = layers
        self.dropout = nn.Dropout(dropout_rate)
        #load pretrained bert model
        self.bert = XLMRobertaModel.from_pretrained(bert_option_name, output_hidden_states = True)
        if emb_train:
            #optional flow of graidents through bert
            self.layer_freezing(freeze_layers=[0])
        #self.unfreeze_bert_embedding()
        else:
            #default: don't train bert
            self.freeze_bert_embedding()
        self.bidirectional=bi_dir
        self.rnn = nn.GRU(self.bert.config.hidden_size, self.hidden_dim, num_layers=self.layer_count, bidirectional=bi_dir, batch_first=True)
        if bi_dir:
            rnn_hidden_dim = 2 * rnn_hidden_dim
        
        self.param1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(rnn_hidden_dim, hidden_dim).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.param2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(hidden_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        
        
        self.graph_nodes = num_class
        self.graph_embedding = nn.Embedding(num_class, graph_node_size)
        self.adjacency_matrix = nn.Parameter(
            nn.init.xavier_uniform_(torch.FloatTensor(num_class, num_class).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), 
                                    gain=np.sqrt(2.0)), requires_grad=True)
        self.gnn = GNNStack(graph_layer_count, graph_node_size, graph_hidden_size, dropout_rate, alpha, nheads, nonlinear_func, internal_layer_concat=False)
        
        self.connect_layer = nn.Linear(rnn_hidden_dim, graph_hidden_size)
        self.linear1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_class)
        self.bert_active_layers = bert_active_layers

    def forward(self, input_seq, input_len, attention_mask=None, token_type_ids=None):
        #input dimensions : [seq_length, batch_size]
        # input_seq, input_len = input_data
        batch_size = input_seq.shape[0]
        bert_hidden = self.bert(input_seq, attention_mask=attention_mask, token_type_ids=token_type_ids)[2]
        bert_hidden = torch.stack(bert_hidden, dim=0)
        bert_hidden = torch.sum(bert_hidden[self.bert_active_layers], dim=0)
        embedding = bert_hidden
        #packing padded sequence
        packed_embedding = pack_padded_sequence(embedding, input_len, enforce_sorted=False, batch_first=True)
        packed_hidden_states, hidden = self.rnn(packed_embedding)
        hidden_outputs, _ = pad_packed_sequence(packed_hidden_states, batch_first=True, total_length=embedding.shape[1])
        
        #batch_size = hidden_outputs.shape[0] 
        max_seq_length = hidden_outputs.shape[1]
        factor = 2 if self.bidirectional else 1
        interaction1 = torch.tanh(hidden_outputs.reshape(batch_size*max_seq_length, factor*self.hidden_dim).mm(self.param1))
        interaction2 = interaction1.mm(self.param2)
        
        hidden_state_weights = torch.softmax(interaction2.view(batch_size, max_seq_length), dim=1)
        zero_vec = -9e15*torch.ones_like(hidden_state_weights)
        attention = torch.where(attention_mask > 0, hidden_state_weights, zero_vec)
        output_rep = (hidden_outputs*attention.unsqueeze(-1)).sum(dim=1)    

        #graph_features
        combined_rep = output_rep #+ context_rep
        graph_weights = self.connect_layer(combined_rep).mm(self.gnn(self.graph_embedding.weight, self.adjacency_matrix).T)
        #graph_weights = torch.softmax(graph_weights, dim=1)
        output = self.linear1(self.dropout(combined_rep))
        output = self.linear2(F.leaky_relu(self.dropout(output)))
        return output + graph_weights

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = tuple(t for t in batch)
        page_id, input_ids, input_masks, label_ids, input_lengths = batch
        pred = self.forward(input_ids, input_lengths, attention_mask=input_masks)
        
        loss = F.binary_cross_entropy_with_logits(pred.view(-1, self.graph_nodes), label_ids.view(-1, self.graph_nodes))
        batch_train_accuracy = torch.tensor(multilabel_accuracy(pred, label_ids))
        
        return {'loss': loss, 'train_acc': batch_train_accuracy}
    
    def training_epoch_end(self, train_step_output_result):
        avg_loss = avg_loss = torch.stack([x['loss'] for x in train_step_output_result]).mean()
        avg_accuracy = torch.stack([x['train_acc'] for x in train_step_output_result]).mean()
        logger.info(' train | avg_train_loss : %s, avg_train_acc : %s' % (avg_loss.item(), avg_accuracy.item()))

        pbar = {'avg_train_acc': avg_accuracy}
        return {'avg_train_loss': avg_loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        batch = tuple(t for t in batch)
        page_id, input_ids, input_masks, label_ids, input_lengths = batch
        pred = self.forward(input_ids, input_lengths, attention_mask=input_masks)
        
        loss = F.binary_cross_entropy_with_logits(pred.view(-1, self.graph_nodes), label_ids.view(-1, self.graph_nodes))
        batch_val_accuracy = torch.tensor(multilabel_accuracy(pred, label_ids))

        return {'val_step_loss': loss, 'val_acc': batch_val_accuracy}
    
    def validation_epoch_end(self, valid_step_output_result):
        avg_loss = avg_loss = torch.stack([x['val_step_loss'] for x in valid_step_output_result]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in valid_step_output_result]).mean()
        logger.info('--'*30)
        logger.info('epoch : %s' % self.current_epoch)
        logger.info(' val   | avg_val_loss : %s, avg_val_acc : %s' % (avg_loss.item(), avg_accuracy.item()))

        pbar = {'avg_val_acc': avg_accuracy}
        return {'val_loss': avg_loss, 'progress_bar': pbar, 'avg_val_acc': avg_accuracy}
    
    def layer_freezing(self, freeze_layers=[], freeze_embedding=True):
        if freeze_embedding:
            for param in list(self.bert.embeddings.parameters()):
                param.requires_grad = False
            logger.info("[XLMR] frozed embedding layer")
        
        for layer_idx in freeze_layers:
            for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            logger.info("[XLMR] frozed internal layer: %s" % layer_idx)  

    def freeze_bert_embedding(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_embedding(self):
            for param in self.bert.parameters():
                param.requires_grad = True


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def start_training(args):
    #rnn features
    rnn_hidden_dim=384
    hidden_dim=768
    bi_directional=True
    rnn_layers_count=3
    dropout=0.1
    emb_train=True

    #graph features
    graph_layer_count=1
    graph_node_size=8
    graph_hidden_size=768
    alpha=0.2
    nheads=12

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Load datasets
    multi_lingual_languages = ['ar', 'bg', 'ca', 'cs', 'da', 'de', 'en', 'es', 'fa', 'fi', 'fr', 
                            'he', 'hi', 'hu', 'id', 'it', 'ko', 'nl', 'no', 'pl', 'pt', 'ro', 
                            'ru', 'sv', 'th', 'tr', 'uk', 'vi', 'zh']
    dm = TextDataModule(args.data_path,
                        multi_lingual_languages=multi_lingual_languages,
                        batch_size=args.batch_size, use_cache=True)

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=args.checkpoint_path,
        save_top_k=1,
        verbose=True,
        monitor='avg_val_acc',
        mode='max',
        prefix='rnn_xlmr'
    )

    model = LitModel(rnn_hidden_dim, hidden_dim, len(dm.le.classes_), layers=rnn_layers_count, 
                          dropout_rate=dropout, bi_dir=bi_directional,
                          emb_train=emb_train, graph_layer_count=graph_layer_count, graph_node_size=graph_node_size,
                          alpha=alpha, nheads=nheads, graph_hidden_size=graph_hidden_size, bert_active_layers=[24])
    logger.info(model)
    logger.info('Model has %s trainable parameters' % (count_parameters(model)))

    trainer = pl.Trainer(max_epochs=args.epochs, min_epochs=args.epochs, gpus=args.gpus, accumulate_grad_batches=args.acc_grads, checkpoint_callback=checkpoint_callback, distributed_backend='ddp')
    trainer.fit(model, dm)

if __name__=="__main__":
    #configure the parser
    parser = ArgumentParser()
    
    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--data_path', default=base_dir, type=str,
                        help='directory where training, validation and label encoders are stored')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=4, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--acc_grads', default=4, type=int,
                        help='accumulate gradients for n batches')
    
    args = parser.parse_args()

    #logging the configs 
    logger.debug('cmdline arguments captured')
    logger.debug('--'*30)
    for arg in vars(args):
        logger.debug("%s - %s" % (arg, getattr(args, arg)))
    logger.debug('=='*30)

    start_training(args)
