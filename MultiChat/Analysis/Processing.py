import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import os
from tqdm import tqdm

def knn_smoothing(mat, k, latent_matrix):
    '''
    KNN smoothing function: Smooth the input matrix using k-nearest neighbors.
    '''
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(latent_matrix)
    distances, indices = nbrs.kneighbors(latent_matrix)
    smoothed_mat = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        smoothed_mat[:, i] = mat[:, indices[i, :]].sum(axis=1)
    
    return smoothed_mat


def Preprocess_CCC_model(base_path, lr_database, cell_rep, expmatrix):
    '''
    smooth the expression matrix using KNN smoothing and normalize the expression of ligands and receptors.
    '''
    latent_fea = cell_rep
    mat = expmatrix.to_numpy()
    mat_smooth = knn_smoothing(mat, k=3, latent_matrix=latent_fea)
    expmatrix_smooth = pd.DataFrame(mat_smooth, index=expmatrix.index, columns=expmatrix.columns)
    expmatrix_smooth.to_csv(os.path.join(base_path, "CCC/expression_smooth.txt"), sep="\t", index=True, header=True)

    # adata = sc.AnnData(mat_smooth)
    # sc.pp.scale(adata, zero_center=True, max_value=10) 
    # data = adata.X 
    # mat_it = np.sum(data > 0, axis=1) 

    LR_ls = lr_database.apply(lambda row: f"{row['Ligand_Symbol']}->{row['Receptor_Symbol']}", axis=1).tolist()
    ligand_exps = []
    for ligand in lr_database['Ligand_Symbol']:
        if '_' in ligand:
            genes = ligand.split('_')
            mean_expression = expmatrix_smooth.loc[genes, :].mean(axis=0)
            ligand_exps.append(mean_expression)
        else:
            ligand_exps.append(expmatrix_smooth.loc[ligand, :])
    ligand_exps = pd.DataFrame(ligand_exps, index=LR_ls, columns=expmatrix_smooth.columns)
    receptor_exps = []
    for receptor in lr_database['Receptor_Symbol']:
        if '_' in receptor:
            genes = receptor.split('_')
            mean_expression = expmatrix_smooth.loc[genes, :].mean(axis=0)
            receptor_exps.append(mean_expression)
        else:
            receptor_exps.append(expmatrix_smooth.loc[receptor, :])
    receptor_exps = pd.DataFrame(receptor_exps, index=LR_ls, columns=expmatrix_smooth.columns)

    ligand_exps_n = (ligand_exps - ligand_exps.min(axis=1).values[:, None]) / (ligand_exps.max(axis=1).values[:, None] - ligand_exps.min(axis=1).values[:, None]) 
    receptor_exps_n = (receptor_exps - receptor_exps.min(axis=1).values[:, None]) / (receptor_exps.max(axis=1).values[:, None] - receptor_exps.min(axis=1).values[:, None])

    pd.DataFrame(ligand_exps_n.T).to_csv(os.path.join(base_path, "CCC/ligands_expression.txt"), sep="\t", index=True, header=True)
    pd.DataFrame(receptor_exps_n.T).to_csv(os.path.join(base_path, "CCC/receptors_expression.txt"), sep="\t", index=True, header=True)
    
    return ligand_exps_n.T, receptor_exps_n.T


def select_peaks_by_genes_location(gene_info, hvg_genes, peaks_to_filter, scope = 250000):
    filtered_gene_info = preprocess_gene_info(gene_info, scope)
    gene_peaks = gene_peaks_pairs_by_location(filtered_gene_info, hvg_genes, peaks_to_filter)
    filtered_peaks = select_peaks_from_pairs(gene_peaks)
    
    return filtered_peaks, gene_peaks


def preprocess_gene_info(gene_info, scope = 250000):
    filtered_gene_info = []
    columns = ['id', 'chr', 'starts', 'ends', 'forward', 'backward', 'gene']
    print("Preprocessing gene_info:")
    for info in tqdm(gene_info.itertuples()):
        chr = info.chr
        starts = info.starts
        ends = int(info.ends)
        genes = info.genes
        gene_info_id = chr + '-' + str(starts) + '-' + str(ends) + '-' + genes
        forward = max(0, starts - scope)
        backward = starts + scope
        filtered_gene_info.append([gene_info_id, chr, starts, ends, forward, backward, genes])
    filtered_gene_info = pd.DataFrame(filtered_gene_info, columns=columns)
    filtered_gene_info = filtered_gene_info.drop_duplicates(subset=['id'])
    return filtered_gene_info


def gene_peaks_pairs_by_location(filtered_gene_info, hvg_genes, peaks_to_filter):
    gene_peaks = {}
    print("Search the genes-peaks correspondence based on gene_info and scope:")
    for info in tqdm(filtered_gene_info.itertuples()):
        if not info.gene in hvg_genes:
            continue
        id = info.id
        chr = info.chr
        starts = info.starts
        ends = info.ends
        forward = info.forward
        backward = info.backward
        gene = info.gene
        if not gene in gene_peaks:
            gene_peaks[gene] = set()
        for peak in peaks_to_filter:
            peak_chr, coordinates = peak.split(':')
            peak_start, peak_end = coordinates.split('-')
            if peak_chr == chr and int(peak_start) >= forward and int(peak_end) <= backward:
                gene_peaks[gene].add(peak)
    gene_peaks = {gene: peaks for gene, peaks in gene_peaks.items() if len(peaks) > 0}
    return gene_peaks

def select_peaks_from_pairs(gene_peaks):
    filtered_peaks = set()
    print("Search the filtered peaks:")
    for key in tqdm(gene_peaks.keys()):
        filtered_peaks.update(gene_peaks[key])
    filtered_peaks = list(filtered_peaks)
    print("After filtering peaks:", len(filtered_peaks))
    return filtered_peaks

def cicero_peaks_peaks(cicero_conn, peakitems, cicero_cotoff):
    filtered_conn = cicero_conn[abs(cicero_conn['coaccess']) >= cicero_cotoff]
    
    result_dict = {}
    for peakitem in tqdm(peakitems, desc="Processing peakitems"):
        peak2_values = filtered_conn.loc[filtered_conn['Peak1'] == peakitem, 'Peak2'].tolist()
        if peak2_values:
            result_dict[peakitem] = peak2_values
    
    return result_dict

def cicero_peaks_peaks_score(cicero_conn, peakitems, cicero_cotoff):
    filtered_conn = cicero_conn[abs(cicero_conn['coaccess']) >= cicero_cotoff]
    
    relevant_peaks = filtered_conn[filtered_conn['Peak1'].isin(peakitems)]
    
    grouped = relevant_peaks.groupby('Peak1')
    
    peaks_lst = []
    scores_lst = []
    
    for peakitem in tqdm(peakitems, desc="Processing peakitems"):
        try:
            group = grouped.get_group(peakitem)
            peaks_lst.extend(group['Peak2'].tolist())
            scores_lst.extend(group['coaccess'].tolist())
        except KeyError:
            continue  # No matches for this peakitem
    
    return peaks_lst, scores_lst