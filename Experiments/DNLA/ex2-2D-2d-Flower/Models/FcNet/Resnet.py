#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[ ]:


class Block(nn.Module):
    
    def __init__(self,in_N,width,out_N,phi):
        super(Block,self).__init__()
        self.L1=nn.Linear(in_N,width)
        self.L2=nn.Linear(width,out_N)
        self.phi=phi
        
    def forward(self,x):
        residual=x
        out=self.phi(self.L2(self.phi(self.L1(x))))
        out=self.phi(out)+residual
        return out


class ResNet(nn.Module):
    
    def __init__(self,in_N,width,out_N,depth):
        super(ResNet,self).__init__()
        self.in_N=in_N
        self.width=width
        self.out_N=out_N
        self.depth=depth
        self.stack=nn.ModuleList()
        self.phi=nn.Tanh()
        
        self.stack.append(nn.Linear(in_N,width))
        for i in range(self.depth):
            self.stack.append(Block(width,width,width,nn.Tanh())) 
        self.stack.append(nn.Linear(width,out_N))
        
    def forward(self,x):
        for j in range(len(self.stack)):
            x=self.stack[j](x)
        return x
    
    def Xavier_initi(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

