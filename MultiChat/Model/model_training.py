import os, sys
import argparse
import time
import random
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import warnings

from torch.utils.data.dataloader import default_collate
from torch import optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utilities import load_ccc_data
from .modules import  CCI_model






def Train_CCC_model_parallel(args):
    nei_adj, spots_ligand, spots_recep, pos, cellName, LRP_name = load_ccc_data(args)
    args.cci_pairs = spots_ligand.size(1)
    print('Size of CCC pairs: ' + str(args.cci_pairs))
    
    model = CCI_model(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr_cci, weight_decay=args.l2_coef)
    
    print('Start model training')
    # if args.use_cuda:
    #     model.cuda()
    #     nei_adj = nei_adj.cuda()
    #     spots_ligand = spots_ligand.cuda()
    #     spots_recep = spots_recep.cuda()
    #     pos = pos.cuda()
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        model.to(device)
        nei_adj = nei_adj.to(device)
        spots_ligand = spots_ligand.to(device)
        spots_recep = spots_recep.to(device)
        pos = pos.to(device)
        print(f'Using GPU: {args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    cnt_wait = 0
    best = 1e9
    best_t = 0
    rela_loss = 1000
    starttime = datetime.datetime.now()
    
    train_loss_list = []
    
    for epoch in range(1000):
        model.train()
        optim.zero_grad()
        
        cost = model(nei_adj, spots_ligand, spots_recep, pos)
        cost = cost*100
        
        train_loss_list.append(cost)
        
        if epoch % 10 == 0:
            if len(train_loss_list) >= 2:
                print(f"{epoch} cost: {cost.data.cpu()} {abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2]}")
            else:
                print(f"{epoch} cost: {cost.data.cpu()}")
        
        if (epoch > 50) and (len(train_loss_list) >= 2):
            if (abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2]) <= 0.005:
                print(abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2])
                print(f"{train_loss_list[-1]} {train_loss_list[-2]} converged!!!")
                print(epoch)
                break
        
        cost.backward()
        optim.step()
    
    model.eval()
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    LR_activity = model.return_LRP_strength(nei_adj, spots_ligand, spots_recep)
    
    # Use the filename from args instead of hardcoding it
    pd.DataFrame(data=LR_activity.data.cpu().numpy(), 
                 index=cellName.tolist(), 
                 columns=LRP_name.tolist()).to_csv(args.lrp_strength_file, sep='\t')
    
    
    
def Train_CCC_model( args):

	nei_adj, spots_ligand, spots_recep, pos, cellName, LRP_name = load_ccc_data(args)

	args.cci_pairs = spots_ligand.size(1)
	print('Size of CCC pairs: ' + str(args.cci_pairs))
	
	model = CCI_model(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
	optim = torch.optim.Adam(model.parameters(), lr=args.lr_cci, weight_decay=args.l2_coef)
	
	print('Start model training')
	if args.use_cuda and torch.cuda.is_available():
		device = torch.device(f'cuda:{args.gpu_id}')
		model.to(device)
		nei_adj = nei_adj.to(device)
		spots_ligand = spots_ligand.to(device)
		spots_recep = spots_recep.to(device)
		pos = pos.to(device)
		print(f'Using GPU: {args.gpu_id}')
	else:
		device = torch.device('cpu')
		print('Using CPU')

	cnt_wait  = 0
	best      = 1e9
	best_t    = 0
	rela_loss = 1000
	starttime = datetime.datetime.now()

	train_loss_list = []

	for epoch in range(1000):
		model.train()
		optim.zero_grad()

		cost = model(nei_adj, spots_ligand, spots_recep, pos)
		cost = cost*100

		train_loss_list.append( cost  )

		if epoch %10==0 :
			if len(train_loss_list) >= 2 :
				print( str(epoch) + " cost: " + str(cost.data.cpu()) + " " + str(abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2]) )
			else:
				print( str(epoch) + " cost: " + str(cost.data.cpu()) )

		if (epoch>50) and (len(train_loss_list) >= 2) :
			if (abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2])  <= 0.005:
				print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )
				print( str(train_loss_list[-1])+ " " + str(train_loss_list[-2]) + " converged!!!" )
				print( epoch )
				break

		cost.backward()
		optim.step()

	model.eval()
	endtime   = datetime.datetime.now()
	time      = (endtime - starttime).seconds
	print("Total time: ", time, "s")

	#torch.save(model.state_dict(), args.outPath + 'CCC_module.pkl')

	LR_activity  = model.return_LRP_strength(nei_adj, spots_ligand, spots_recep)

	pd.DataFrame(data=LR_activity.data.cpu().numpy(), index = cellName.tolist(), columns = LRP_name.tolist() ).to_csv( args.outPath + args.InterCCC_Name, sep='\t')