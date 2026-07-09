import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
import os

from typing import Optional
from torch.nn.modules.module import Module
from torch.autograd import Variable
from collections import OrderedDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

CUDA_LAUNCH_BLOCKING = 1



class Contrast_single(nn.Module):
	def __init__(self, hidden_dim, tau):
		super(Contrast_single, self).__init__()
		
		self.tau = tau

	def sim(self, z):
		z_norm = torch.norm(z, dim=-1, keepdim=True)
		dot_numerator   = torch.mm(z, z.t())
		dot_denominator = torch.mm(z_norm, z_norm.t())
		sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
		return sim_matrix

	def forward(self, z, pos):
		matrix  = self.sim(z)
		matrix  = matrix/(torch.sum(matrix, dim=1).view(-1, 1) + 1e-8)
		lori    = -torch.log(matrix.mul(pos).sum(dim=-1)).mean()

		return lori

class intra_att_LR(nn.Module):
	def __init__(self, hidden_dim, attn_drop):
		super(intra_att_LR, self).__init__()
		self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
		nn.init.xavier_normal_(self.att.data, gain=1.414)
		if attn_drop:
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			self.attn_drop = lambda x: x
			
		self.softmax   = nn.Softmax(dim=1)
		self.leakyrelu = nn.LeakyReLU()

		self.map_l     = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.uniform_(self.map_l.weight, a=0, b=1)
		nn.init.uniform_(self.map_l.bias, a=0, b=0.01)

		self.map_r     = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.uniform_(self.map_r.weight, a=0, b=1)
		nn.init.uniform_(self.map_r.bias, a=0, b=0.01)

	def forward(self, nei, h, h_refer):
		h         = F.relu(self.map_l(h))
		h_refer   = F.relu(self.map_r(h_refer))

		nei_emb   = F.embedding(nei, h)
		h_refer_n = torch.unsqueeze(h_refer, 1)
		h_refer_n = h_refer_n.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer_n, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		nei_emb   = (att*nei_emb).sum(dim=1)
		nei_emb   = F.relu(nei_emb*h_refer)
		return nei_emb

	def return_attention(self, nei, h, h_refer):
		h         = F.relu(self.map_l(h))
		h_refer   = F.relu(self.map_r(h_refer))
		nei_emb   = F.embedding(nei, h)
		h_refer   = torch.unsqueeze(h_refer, 1)
		h_refer   = h_refer.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		att       = torch.squeeze(att, dim=-1)

		return att

class LRP_attention(nn.Module):
	def __init__(self, cci_pairs, hidden_dim, attn_drop):
		super(LRP_attention, self).__init__()
		
		self.intra_cci  = nn.ModuleList([intra_att_LR(hidden_dim, attn_drop) for _ in range(cci_pairs)])
		self.cci_pairs  = cci_pairs

	def forward(self, sele_nei, ligand_exp, receptor_exp):
		LR_embeds     = []
		for z in range(self.cci_pairs):
			temp_emb  = self.intra_cci[z](sele_nei, ligand_exp[:,z].view(-1,1), receptor_exp[:,z].view(-1,1))
			LR_embeds.append( temp_emb.view(1,-1) )

		LR_embeds  = torch.cat(LR_embeds, dim=0)
		LR_embeds  = LR_embeds.t().cuda()

		return LR_embeds

	def return_LR_atten_spot(self, sele_nei, ligand_exp, receptor_exp):

		atten_list = []
		for z in range(self.cci_pairs):
			temp_atten = self.intra_cci[z].return_attention(sele_nei, ligand_exp[:,z].view(-1,1), receptor_exp[:,z].view(-1,1))
			atten_list.append(temp_atten)

		atten_list = torch.cat(atten_list, dim=1)

		return atten_list
