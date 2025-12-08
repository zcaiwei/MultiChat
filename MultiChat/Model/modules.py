import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import collections
import time
import math
import random

import torch.utils.data as data_utils
import scipy.sparse as sp
import torch.nn.modules.loss

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.distributions import Normal, kl_divergence as kl
from torch import optim

from .Layers import LRP_attention, Contrast_single
from .utilities import adjust_learning_rate




class CCI_model(nn.Module):

	def __init__(self, cci_pairs, hidden_dim, attn_drop, layers, tau):
		super(CCI_model, self).__init__()

		self.LRP_attention = LRP_attention(cci_pairs, hidden_dim, attn_drop)
		self.enco_latent   = nn.Linear(layers[0], layers[1], bias=False)
		self.contrast      = Contrast_single(hidden_dim, tau)

	def forward(self, nei_adj, ligand_exp, receptor_exp, pos):

		embeds_LRs   = self.LRP_attention(nei_adj, ligand_exp, receptor_exp)
		latent       = self.enco_latent(embeds_LRs)
		lori         = self.contrast(latent, pos)
		return lori

	def return_LRP_strength(self, nei_adj, ligand_exp, receptor_exp):

		embeds_LRs = self.LRP_attention(nei_adj, ligand_exp, receptor_exp)

		return embeds_LRs

	def return_LR_atten(self, nei_adj, ligand_exp, receptor_exp):

		atten_list = self.LRP_attention.return_LR_atten_spot(nei_adj, ligand_exp, receptor_exp)

		return atten_list





def log_nb_positive(x, mu, theta, eps=1e-8):
	
	x = x.float()
	
	if theta.ndimension() == 1:
		theta = theta.view(
			1, theta.size(0)
		)  # In this case, we reshape theta for broadcasting

	log_theta_mu_eps = torch.log(theta + mu + eps)

	res = (
		theta * (torch.log(theta + eps) - log_theta_mu_eps)
		+ x * (torch.log(mu + eps) - log_theta_mu_eps)
		+ torch.lgamma(x + theta)
		- torch.lgamma(theta)
		- torch.lgamma(x + 1)
	)

	#print(res.size())

	return - torch.sum( res, dim = 1 )

def mse_loss(y_true, y_pred):

	y_pred = y_pred.float()
	y_true = y_true.float()

	ret = torch.pow( (y_pred - y_true) , 2)

	return torch.sum( ret, dim = 1 )

