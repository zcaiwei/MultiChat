import os
import time
import argparse
import torch
import random
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

import sys 
sys.path.append('/home/nas2/biod/zhencaiwei/RegChatz_V2/')

if 'ipykernel' in sys.modules:     
    sys.argv = sys.argv[:1]  


def parameter_setting():
    parser = argparse.ArgumentParser(description='Spatial transcriptomics analysis by HIN')
    
    parser.add_argument('--inputPath', '-IP', type=str, default='Datasets/MISAR/', help='data directory')
    parser.add_argument('--outPath', '-od', type=str, default='Datasets/MISAR/CCC/', help='Output path')
    parser.add_argument('--utilitePath', '-uP', type=str, default='Datasets/MISAR/inputs/', help='data directory')
    parser.add_argument('--spatialLocation', '-sLocation', type=str, default='Coord.csv', help='spot physical location')
    parser.add_argument('--annoFile', '-aFile', type=str, default='CellType.csv', help='annotation file')
    parser.add_argument('--pos_pair', '-posP', type=str, default='Spot_positive_pairs.txt', help='positive pairs between spots')
    parser.add_argument('--Ligands_exp', '-Ligands_exp', type=str, default='ligands_expression.txt', help='Expression of ligands per spot')
    parser.add_argument('--Receptors_exp', '-Receptors_exp', type=str, default='receptors_expression.txt', help='Expression of receptors per spot')
    parser.add_argument('--cci_pairs', '-cci_pairs', type=int, default = 4019, help='The number of receptors for each spot')
    parser.add_argument('--locMeasure', '-locMeas', type=str, default='euclidean', help='Calculate spot location similarity by euclidean')
    parser.add_argument('--Cell_pos_nos', '-CellPN', type=int, default=6, help='The number of positive cells for each cell')
    parser.add_argument('--tau', '-tau', type=float, default=0.8)
    parser.add_argument('--attn_drop', '-attn_drop', type=float, default=0.5)
    parser.add_argument('--lr_cci', '-lr_cci', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--l2_coef', '-l2_coef', type=float, default=0)
    parser.add_argument('--patience', '-patience', type=int, default=30)
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help="whether use cuda(default: True)")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--selected_cell_type', '-sct', type=str, default=None, help='Specify cell type to select nodes from (e.g. "Astro")')
    parser.add_argument('--cell_type_column', '-ctc', type=str, default='cell_type', help='Column name in annotation file containing cell type information')
    parser.add_argument('--k_neighbors', '-k', type=int, default=11, help='Number of nearest neighbors to include (including self)')
    parser.add_argument('--InterCCC_Name', '-InterCCC_Name', type=str, default='CCC_module_LRP_strength.txt', help='Name of the InterCCC output file')
    
    return parser



# def load_ccc_data( args ):

# 	print("spot location for adjancy")
# 	spot_loc     = pd.read_table(args.spatialLocation, header = 0, index_col = 0, sep = ',')
# 	dist_loc     = pairwise_distances(spot_loc.values, metric = args.locMeasure)

# 	sorted_knn    = dist_loc.argsort(axis=1)
# 	selected_node = []
# 	#used_spots    = []
# 	for index in list(range( np.shape(dist_loc)[0] )):
# 		selected_node.append( sorted_knn[index, :11] )
# 		#used_spots.extend( sorted_knn[index, :11] )
# 	selected_node_df = pd.DataFrame(selected_node)
# 	selected_node = np.array(selected_node)
# 	selected_node  = torch.LongTensor(selected_node)
# 	#used_spots     = torch.LongTensor(list(set(used_spots)))

# 	print("spot-ligand data")
# 	spots_ligand    = pd.read_table(args.Ligands_exp, header = 0, index_col = 0)
# 	spots_ligand_n  = torch.FloatTensor(spots_ligand.values)

# 	print("spot-receptor data")
# 	spots_recep   = pd.read_table(args.Receptors_exp, header = 0, index_col = 0)
# 	spots_recep_n = torch.FloatTensor(spots_recep.values)

# 	pos   = pd.read_table(args.pos_pair, header = None, index_col = None).values
# 	pos   = torch.FloatTensor(pos)

# 	return selected_node, spots_ligand_n, spots_recep_n, pos, spots_ligand.index, spots_ligand.columns




def get_cell_positive_pairs(cell_clus, cell_loc, args):
	'''
 	get edges between cells: positive pairs
 	'''
	cell_clus_values = cell_clus['cell_type'].values.astype('str')
	dist_out = pairwise_distances(cell_loc) 
	cell_cell_adj = np.zeros((len(cell_clus), len(cell_clus)), dtype=int)
	for index in range(len(cell_clus)):
			match_int = np.where(cell_clus_values[index] == cell_clus_values)[0]
			sorted_knn = dist_out[index, match_int].argsort()
			cell_cell_adj[index, match_int[sorted_knn[:args.Cell_pos_nos]]] = 1
	pd.DataFrame(cell_cell_adj).to_csv(args.outPath + args.pos_pair, header=None, index=None, sep='\t')




def load_ccc_data(args):
    print("spot location for adjacency")
    spot_loc = pd.read_table(args.spatialLocation, header=0, index_col=0, sep=',')
    
    print("loading cell type annotations")
    cell_type_df = pd.read_csv(args.annoFile, header=0, index_col=0, sep="\t")
    
    print("Calculating pairwise distances between spots")
    dist_loc = pairwise_distances(spot_loc.values, metric=args.locMeasure)
    sorted_knn = dist_loc.argsort(axis=1)

    selected_node = []
    
    if args.selected_cell_type:
        print(f"Selecting nodes for cell type: {args.selected_cell_type}")
        safe_cell_type = args.selected_cell_type.replace('/', '-').replace(' ', '-')
        selected_cell_samples = cell_type_df[cell_type_df['cell_type'] == args.selected_cell_type].index
        selected_row_numbers = [cell_type_df.index.get_loc(idx) for idx in selected_cell_samples]
        # print("selected cell type samples: ", selected_cell_samples)
    
        for target_node in tqdm(range(len(cell_type_df)), desc="Processing nodes"):
            all_neighbors = sorted_knn[target_node, :]
            same_type_neighbors = [n for n in all_neighbors if n in selected_row_numbers]
            top11_same_type = [target_node] + same_type_neighbors[:10]
			
            if len(top11_same_type) < 11:
                print(f"Warning: Only found {len(top11_same_type)} neighbors for cell {target_node}")
			
            selected_node.append(top11_same_type)

        pd.DataFrame(selected_node).to_csv(args.outPath + 'Nei_adj_' + safe_cell_type +'.csv', header=None, index=None, sep='\t')

    else:
        for index in list(range(np.shape(dist_loc)[0])):
            selected_node.append(sorted_knn[index, :11])
        pd.DataFrame(selected_node).to_csv(args.outPath + 'Nei_adj.csv' , header=None, index=None, sep='\t')
    
    selected_node = np.array(selected_node)
    selected_node = torch.LongTensor(selected_node)

    print("spot-ligand data")
    spots_ligand = pd.read_table(args.Ligands_exp, header=0, index_col=0)
    spots_ligand_n = torch.FloatTensor(spots_ligand.values)

    print("spot-receptor data")
    spots_recep = pd.read_table(args.Receptors_exp, header=0, index_col=0)
    spots_recep_n = torch.FloatTensor(spots_recep.values)

    pos = pd.read_table(args.pos_pair, header=None, index_col=None).values
    pos = torch.FloatTensor(pos)

    return selected_node, spots_ligand_n, spots_recep_n, pos, spots_ligand.index, spots_ligand.columns






def perturb_pos_pair_row(row):
    '''get random positive pairs: perturb edge'''
    ones_pos = row[row == 1].index.tolist()
    remaining_pos = [col for col in row.index if col not in ones_pos]
    new_ones_pos = np.random.choice(remaining_pos, size=6, replace=False)
    new_row = row.copy()
    new_row[:] = 0  
    new_row[new_ones_pos] = 1  
    
    return new_row


#------------------------------


def get_CCC_data(adata, latent, args, threthold = 5):

	exp_data    = sp.csr_matrix.toarray(adata.X)
	exp_data_n  = np.zeros( (exp_data.shape[0], exp_data.shape[1]) )
	exp_data_n[ np.where(exp_data > 0) ] = 1
	sum_gene    = np.sum(exp_data_n, axis = 0)

	CCC         = pd.read_table(args.inputPath + args.CCC_file, header=None, index_col=None).values
	ligands     = list(set( adata.var_names[np.where(sum_gene>=threthold)] ) & set(CCC[:,0]))
	receptors   = list(set( adata.var_names[np.where(sum_gene>=threthold)] ) & set(CCC[:,1]))

	lrp_list    = []
	symbol      = '->'
	for index, (lig, rec) in enumerate(CCC):
		if (lig in ligands) and (rec in receptors):
			lrp_list.append( symbol.join( [lig, rec] ) )

	used_ligands_n   = []
	used_receptors_n = []

	for str in list(set(lrp_list)):
		temps = str.split( '->' )
		used_ligands_n.append( temps[0] )
		used_receptors_n.append( temps[1] )

	ligand_int     = [ adata.var_names.tolist().index(item) for item in used_ligands_n  if item in adata.var_names.tolist() ]
	receptor_int   = [ adata.var_names.tolist().index(item) for item in used_receptors_n  if item in adata.var_names.tolist() ]

	exp_data_s     = knn_smoothing(latent, 3, exp_data)
	adata.X        = sp.csr_matrix( exp_data_s )

	sc.pp.normalize_total(adata, inplace=True)
	sc.pp.scale(adata, max_value=10)
	
	ligands_exp    = adata.X[:,ligand_int]
	receptors_exp  = adata.X[:,receptor_int]

	liagand_exps_n = (ligands_exp-ligands_exp.min(axis=0))/(ligands_exp.max(axis=0)-ligands_exp.min(axis=0))
	recep_exps_n   = (receptors_exp-receptors_exp.min(axis=0))/(receptors_exp.max(axis=0)-receptors_exp.min(axis=0))

	pd.DataFrame( liagand_exps_n, index = adata.obs_names.tolist(), columns=list(set(lrp_list)) ).to_csv( args.outPath + args.Ligands_exp, sep='\t' )
	pd.DataFrame( recep_exps_n, index = adata.obs_names.tolist(), columns=list(set(lrp_list)) ).to_csv( args.outPath + args.Receptors_exp, sep='\t' )
	pd.DataFrame( adata.X, index = adata.obs_names.tolist(), columns=adata.var_names.tolist() ).to_csv( args.outPath + args.Denoised_exp, sep='\t' )


def knn_smoothing(latent, k, mat):
    dist = pairwise_distances(latent)
    row = []
    col = []
    sorted_knn = dist.argsort(axis=1)
    for idx in list(range(np.shape(dist)[0])):
        col.extend(sorted_knn[idx, : k].tolist())
        row.extend([idx] * k)

    res = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(col)):
        res[row[i]] += mat[col[i]]

    return res

def save_checkpoint(model, folder='./saved_model/', filename='model_best.pth.tar'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	torch.save(model.state_dict(), os.path.join(folder, filename))

def load_checkpoint(file_path, model, use_cuda=False):

	if use_cuda:
		device = torch.device( "cuda" )
		model.load_state_dict( torch.load(file_path) )
		model.to(device)
		
	else:
		device = torch.device('cpu')
		model.load_state_dict( torch.load(file_path, map_location=device) )

	model.eval()
	return model

def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

	lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr  



 
