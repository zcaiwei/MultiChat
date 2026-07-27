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
from tqdm import tqdm
from pathlib import Path

from .utilities import load_ccc_data, parameter_setting, get_cell_positive_pairs, perturb_pos_pair_row
from .modules import  CCI_model




def get_ccc_tensors_from_adata(mc_adata):
    selected_node = torch.LongTensor(
        mc_adata.uns["CCC"]["Nei_adj"].values.copy()
    )

    spots_ligand = mc_adata.uns["CCC"]["lig_exp"].reindex(mc_adata.obs_names)
    spots_ligand_n = torch.FloatTensor(spots_ligand.values.copy())

    spots_recep = mc_adata.uns["CCC"]["rec_exp"].reindex(mc_adata.obs_names)
    spots_recep_n = torch.FloatTensor(spots_recep.values.copy())

    pos = torch.FloatTensor(
        mc_adata.uns["CCC"]["pos_pair"].values.copy()
    )

    cellName = spots_ligand.index
    LRP_name = spots_ligand.columns

    return selected_node, spots_ligand_n, spots_recep_n, pos, cellName, LRP_name




def Train_CCC_model_parallel(mc_adata, args):
    nei_adj, spots_ligand, spots_recep, pos, cellName, LRP_name = get_ccc_tensors_from_adata(mc_adata)
    args.cci_pairs = spots_ligand.size(1)
    print('Size of CCC pairs: ' + str(args.cci_pairs))
    
    model = CCI_model(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr_cci, weight_decay=args.l2_coef)
    
    print('================================================================')
    print('Contrastive learning for background generation start training')
    print('================================================================')

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
    
    return args.lrp_strength_file
    
    
    
def Train_CCC_model(mc_adata, args):

	mc_adata, nei_adj, spots_ligand, spots_recep, pos, cellName, LRP_name = load_ccc_data(mc_adata, args)

	args.cci_pairs = spots_ligand.size(1)
	print('Size of CCC pairs: ' + str(args.cci_pairs))
	
	model = CCI_model(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
	optim = torch.optim.Adam(model.parameters(), lr=args.lr_cci, weight_decay=args.l2_coef)
	
	print('================================================================')
	print('Contrastive learning start training')
	print('================================================================')
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

	LRI_strength_df = pd.DataFrame(data=LR_activity.data.cpu().numpy(), index = cellName.tolist(), columns = LRP_name.tolist() )
	mc_adata.uns.setdefault("CCC", {})
	mc_adata.uns["CCC"]["LRI_module_strength"] = LRI_strength_df 
     
	return mc_adata
      

def run_contrastive_module(
    mc_adata,
    base_path,
    gpu_id=0,
    selected_cell_type=None
):
    """
    Run CCC contrastive module using data stored in mc_adata.

    Requires:
    - mc_adata.uns["CCC"]["lig_exp"]
    - mc_adata.uns["CCC"]["rec_exp"]
    - mc_adata.obs[cell_type_key]
    - mc_adata.obsm[spatial_key]

    Stores:
    - mc_adata.uns["CCC"]["LRI_module_strength"]
    """

    if "CCC" not in mc_adata.uns:
        raise KeyError("mc_adata.uns['CCC'] does not exist.")

    if "lig_exp" not in mc_adata.uns["CCC"]:
        raise KeyError("mc_adata.uns['CCC']['lig_exp'] does not exist.")

    if "rec_exp" not in mc_adata.uns["CCC"]:
        raise KeyError("mc_adata.uns['CCC']['rec_exp'] does not exist.")


    parser = parameter_setting()
    args, unknown = parser.parse_known_args()

    args.gpu_id = gpu_id
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.selected_cell_type = selected_cell_type

    if args.use_cuda:
        torch.cuda.set_device(args.gpu_id)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if base_path is not None:
        base_path = Path(base_path).expanduser().resolve()
        args.inputPath = str(base_path) + "/"
        args.outPath = str(base_path / "CCC") + "/"

    mc_adata = get_cell_positive_pairs(mc_adata=mc_adata, args=args)

    start = time.time()
    mc_adata = Train_CCC_model(mc_adata, args)
    duration = time.time() - start

    print("Finish training, total time is: " + str(duration) + "s")

    return mc_adata


def run_contrastive_module_for_bg(
    mc_adata,
    args,
    base_path=None,
    n_runs=10,
    n_processes=5,
    seed=42,
    cleanup=True,
):
    import copy
    import glob
    import multiprocessing
    from types import SimpleNamespace
    from ..Analysis import Bg_training
    from ..Analysis.Intra_strength import load_background_inter

    if "CCC" not in mc_adata.uns:
        raise KeyError("mc_adata.uns['CCC'] does not exist.")

    for key in ["lig_exp", "rec_exp", "pos_pair"]:
        if key not in mc_adata.uns["CCC"]:
            raise KeyError(f"mc_adata.uns['CCC']['{key}'] does not exist.")

    if base_path is None:
        base_path = args.inputPath

    base_path = Path(base_path).expanduser().resolve()
    bg_path = base_path / "Bg_CCC"
    bg_path.mkdir(parents=True, exist_ok=True)

    bg_args = copy.deepcopy(args)
    bg_args.inputPath = str(base_path) + "/"
    bg_args.outPath = str(bg_path) + "/"
    bg_args.seed = seed
    bg_args.use_cuda = bg_args.use_cuda and torch.cuda.is_available()

    rng = np.random.default_rng(seed)

    pos_pair = mc_adata.uns["CCC"]["pos_pair"].copy()
    lig_exp = mc_adata.uns["CCC"]["lig_exp"].copy()
    rec_exp = mc_adata.uns["CCC"]["rec_exp"].copy()

    pos_pair_perturb = pos_pair.apply(perturb_pos_pair_row, axis=1)

    lig_exp_shuffled = lig_exp.apply(
        lambda col: rng.permutation(col.to_numpy()),
        axis=0,
    )
    lig_exp_shuffled.index = lig_exp.index
    lig_exp_shuffled.columns = lig_exp.columns

    rec_exp_shuffled = rec_exp.apply(
        lambda col: rng.permutation(col.to_numpy()),
        axis=0,
    )
    rec_exp_shuffled.index = rec_exp.index
    rec_exp_shuffled.columns = rec_exp.columns

    lig_exp_shuffled.to_csv(bg_path / "ligands_expression_shuffled.txt", sep="\t")
    rec_exp_shuffled.to_csv(bg_path / "receptors_expression_shuffled.txt", sep="\t")
    pos_pair_perturb.to_csv(
        bg_path / "Spot_positive_pairs_shuffled.txt",
        sep="\t",
        header=False,
        index=False,
    )

    bg_uns = {
        "CCC": copy.deepcopy(mc_adata.uns["CCC"]),
    }

    bg_uns["CCC"]["lig_exp"] = lig_exp_shuffled
    bg_uns["CCC"]["rec_exp"] = rec_exp_shuffled
    bg_uns["CCC"]["pos_pair"] = pos_pair_perturb

    if "cell_loc" in mc_adata.uns:
        bg_uns["cell_loc"] = mc_adata.uns["cell_loc"]
    elif "spatial" in mc_adata.obsm:
        bg_uns["cell_loc"] = pd.DataFrame(
            mc_adata.obsm["spatial"],
            index=mc_adata.obs_names,
        )
    else:
        raise KeyError("Need mc_adata.uns['cell_loc'] or mc_adata.obsm['spatial'].")

    bg_adata = SimpleNamespace(
        obs=mc_adata.obs.copy(),
        obs_names=mc_adata.obs_names.copy(),
        uns=bg_uns,
    )

    tasks = [
        (run_idx, bg_adata, bg_args)
        for run_idx in range(1, n_runs + 1)
    ]

    # ctx = multiprocessing.get_context("spawn")
    # with ctx.Pool(processes=n_processes) as pool:
    #     output_files = pool.map(Bg_training.run_training, tasks)

    n_processes = max(1, min(n_processes, n_runs))

    if n_processes == 1:
        output_files = [
            Bg_training.run_training(task)
            for task in tasks
        ]
    else:
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(
            processes=n_processes,
            maxtasksperchild=1,
        ) as pool:
            output_files = pool.map(
                Bg_training.run_training,
                tasks,
                chunksize=1,
            )

    print("Background training files:")
    for file in output_files:
        print(file)

    background_inter_df = load_background_inter(
        str(bg_path) + "/",
        file_pattern="LRI_module_strength_run_*.txt",
    )

    concat_file = bg_path / "LRI_module_strength_concat.txt"
    background_inter_df.to_csv(concat_file, sep="\t", index=True)

    if cleanup:
        temp_patterns = [
            "LRI_module_strength_run_*.txt",
            "ligands_expression_shuffled.txt",
            "receptors_expression_shuffled.txt",
            "Spot_positive_pairs_shuffled.txt",
        ]

        for pattern in temp_patterns:
            for file in glob.glob(str(bg_path / pattern)):
                os.remove(file)

    print(f"Concatenated background file saved to: {concat_file}")

    return background_inter_df





