# -*- coding: utf-8 -*-
"""
@author: lyman
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import Bio
from c_rnn_gan import Generator, Discriminator
import train_utils as U

G_LRN_RATE = 0.0002
D_LRN_RATE = 0.0002
MAX_GRAD_NORM = 5.0

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)
OUTPUT_DIM      = 1
NUM_FEATS       = 25
HIDDEN_DIM      = 256
LSTM_LAYERS     = 2
DROPOUT         = 0.25

SEQ_LEN         = 1273
BATCH_SIZE      = 64
EPOCHS          = 50
device = 'cpu'




class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)




class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss
        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator
        loss = -(ylog(p) + (1-y)log(1-p))
        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)

    



def train(train_loader, test_loader, freeze_g=False, freeze_d=False):
    
    #create network structures
    model = {
        'g': Generator(NUM_FEATS, use_cuda=False),
        'd': Discriminator(NUM_FEATS, use_cuda=False)
    }
    optimizer = {
            'g': optim.Adam(model['g'].parameters(), G_LRN_RATE),
            'd': optim.Adam(model['d'].parameters(), D_LRN_RATE)
        }
    criterion = {
        'g': GLoss(),
        'd': DLoss()
    }
    
    for epoch in range(EPOCHS):
        
        model['g'].train()
        model['d'].train()

        loss = {}
        g_loss_total = 0.0
        d_loss_total = 0.0
        num_corrects = 0
        num_real_corrects = 0
        num_fake_corrects = 0
        num_sample = 0
        
        for i, (seqs, true_data) in enumerate(train_loader):
            if i % 1 == 0: 
                print('training... {}/{}'.format(i, len(train_loader)))
            
            real_batch_sz = true_data.shape[0]
            g_states = model['g'].init_hidden(real_batch_sz)
            d_state = model['d'].init_hidden(real_batch_sz)
            #GENERATOR
            if not freeze_g:
                optimizer['g'].zero_grad()
            # prepare inputs
            z = torch.empty([real_batch_sz, SEQ_LEN, NUM_FEATS]).uniform_() # random vector
            true_data = torch.Tensor(true_data)
            
            # feed inputs to generator
            g_feats, _ = model['g'](z, g_states)
            # calculate loss, backprop, and update weights of G
            if isinstance(criterion['g'], GLoss):
                d_logits_gen, _, _ = model['d'](g_feats, d_state)
                loss['g'] = criterion['g'](d_logits_gen)
            else: # feature matching
                # feed real and generated input to discriminator
                _, d_feats_real, _ = model['d'](true_data, d_state)
                _, d_feats_gen, _  = model['d'](g_feats, d_state)
                loss['g'] = criterion['g'](d_feats_real, d_feats_gen)

            if not freeze_g:
                loss['g'].backward()
                nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
                optimizer['g'].step()
            
            #### DISCRIMINATOR ####
            if not freeze_d:
                optimizer['d'].zero_grad()
            # feed real and generated input to discriminator
            true_embeds = U.get_embeds(true_data)
            d_logits_real, _, _ = model['d'](true_embeds, d_state)
            # need to detach from operation history to prevent backpropagating to generator
            d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
            # calculate loss, backprop, and update weights of D
            loss['d'] = criterion['d'](d_logits_real, d_logits_gen)
            if not freeze_d:
                loss['d'].backward()
                nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
                optimizer['d'].step()

            g_loss_total += loss['g'].item()
            d_loss_total += loss['d'].item()
            
            nrc = (d_logits_real > 0.5).sum().item()
            nfc = (d_logits_gen < 0.5).sum().item()
            num_real_corrects += nrc
            num_fake_corrects += nfc
            num_corrects += nrc + nfc
            num_sample += real_batch_sz
        
        trn_g_loss, trn_d_loss = 0.0, 0.0
        trn_acc = 0.0
        trn_acc_real = 0.0
        trn_acc_fake = 0.0
        if num_sample > 0:
            trn_g_loss = g_loss_total / num_sample
            trn_d_loss = d_loss_total / num_sample
            trn_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)
            trn_acc_real  = 100 * num_real_corrects / num_sample
            trn_acc_fake  = 100 * num_fake_corrects / num_sample
            
        if trn_acc > 90:
            freeze_d = True
        else:
            freeze_d = False
        
        #EVALUATE
        val_g_loss, val_d_loss, val_acc, val_acc_real, val_acc_fake = run_validation(model, criterion, test_loader)
        print(' ')
        print('EPOCH ', epoch)
        print(' ')
        print('TRAINING SET:')
        print('Gen Loss = {:0.4f}  Disc Loss = {:0.4f}  Disc Acc = {:0.4f}'.format(trn_g_loss, trn_d_loss, trn_acc))
        print('{:0.2f}% of fakes detected'.format(trn_acc_fake))
        print('{:0.2f}% of reals accepted'.format(trn_acc_real))
        print(' ')
        print('Validation SET:')
        print('Gen Loss = {:0.4f}  Disc Loss = {:0.4f}  Disc Acc = {:0.4f}'.format(val_g_loss, val_d_loss, val_acc))
        print('{:0.2f}% of fakes detected'.format(val_acc_fake))
        print('{:0.2f}% of reals accepted'.format(val_acc_real))
        print(' ')
        
        
        
def run_validation(model, criterion, test_loader):
    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_real_corrects = 0
    num_fake_corrects = 0
    num_sample = 0

    for i, (seqs, true_data) in enumerate(test_loader):
        if i % 1 == 0: 
            print('testing... {}/{}'.format(i, len(test_loader)))
        real_batch_sz = true_data.shape[0]

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, SEQ_LEN, NUM_FEATS]).uniform_() # random vector
        true_data = torch.Tensor(true_data)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        true_embeds = U.get_embeds(true_data)
        d_logits_real, d_feats_real, _ = model['d'](true_embeds, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss
        if isinstance(criterion['g'], GLoss):
            g_loss = criterion['g'](d_logits_gen)
        else: # feature matching
            g_loss = criterion['g'](d_feats_real, d_feats_gen)

        d_loss = criterion['d'](d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        
        nrc = (d_logits_real > 0.5).sum().item()
        nfc = (d_logits_gen < 0.5).sum().item()
        num_real_corrects += nrc
        num_fake_corrects += nfc
        num_corrects += nrc + nfc
        
        num_sample += real_batch_sz


    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc_total = 0.0
    d_acc_real = 0.0
    d_acc_fake = 0.0
    if num_sample > 0:
        g_loss_avg  = g_loss_total / num_sample
        d_loss_avg  = d_loss_total / num_sample
        d_acc_total = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)
        d_acc_real  = 100 * num_real_corrects / num_sample
        d_acc_fake  = 100 * num_fake_corrects / num_sample

    return g_loss_avg, d_loss_avg, d_acc_total, d_acc_real, d_acc_fake
    
if __name__ == '__main__':
    rbd = False
    if rbd:
        data = U.load_rbd()
    else:
        data = U.load_spike()
        
    SEQ_LEN = len(data[0])
    
    split = [data[i:i + 5000] for i in range(0, len(data), 5000)]
    test_seqs = split[0]
    test_seqs = test_seqs[0:1000]
    test_seq_str = [U.str_to_int(i) for i in test_seqs]
    
    real_seqs = split[1]
    real_seq_str = [U.str_to_int(i) for i in real_seqs]
    
    encoding_seq_tensor = (torch.FloatTensor(5000, SEQ_LEN).uniform_(0, 25)).int()
    encoding_test_tensor = (torch.FloatTensor(1000, SEQ_LEN).uniform_(0, 25)).int()
    
    real_seq_tensor = torch.Tensor(real_seq_str)
    real_test_tensor = torch.Tensor(test_seq_str)

    train_dataset = TensorDataset(encoding_seq_tensor, real_seq_tensor)
    test_dataset = TensorDataset(encoding_test_tensor, real_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = True, drop_last= True)

    
    train(train_loader, test_loader)
    
    
    #EPOCH 0: 
    
    
    