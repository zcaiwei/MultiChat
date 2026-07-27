import pandas as pd
import numpy as np

import os
from tqdm import tqdm
from anndata import AnnData
from scipy import sparse


def knn_smoothing(mat, k, latent_matrix):
    '''
    KNN smoothing function: Smooth the input matrix using k-nearest neighbors.
    '''
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(latent_matrix)
    distances, indices = nbrs.kneighbors(latent_matrix)
    smoothed_mat = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        smoothed_mat[:, i] = mat[:, indices[i, :]].sum(axis=1)
    
    return smoothed_mat


def Preprocess_CCC_model(mc_adata, base_path, lr_database, cell_rep, expmatrix):
    '''
    smooth the expression matrix using KNN smoothing and normalize the expression of ligands and receptors.
    '''
    ccc_path = os.path.join(base_path, "CCC")
    os.makedirs(ccc_path, exist_ok=True)
    print(f"Directory for CCC inference files: {ccc_path}")

    latent_fea = cell_rep.to_numpy()
    mat = expmatrix.to_numpy()
    mat_smooth = knn_smoothing(mat, k=3, latent_matrix=latent_fea)
    expmatrix_smooth = pd.DataFrame(mat_smooth, index=expmatrix.index, columns=expmatrix.columns)

    mc_adata.uns.setdefault("CCC", {})
    mc_adata.uns['CCC']['smooth_exp'] = expmatrix_smooth

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
  
    return mc_adata, ligand_exps_n.T, receptor_exps_n.T


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
    for peakitem in peakitems:
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
    
    for peakitem in peakitems:
        try:
            group = grouped.get_group(peakitem)
            peaks_lst.extend(group['Peak2'].tolist())
            scores_lst.extend(group['coaccess'].tolist())
        except KeyError:
            continue  # No matches for this peakitem
    
    return peaks_lst, scores_lst


def construct_mc_adata(
    RNA_data,
    cell_clus=None,
    cell_loc=None,
    lr_df=None,
    r_tf_cellcall=None,
    l_r_tf_tg_df=None,
    normalize_log=False,
    celltype_col="cell_type",
    groupby="cell_type",
    marker_top_n=50,
    markers=None,
    if_multi_layer=True,
):
    """
    Construct mc_adata from MultiChat input tables.
    """
    if lr_df is None:
        raise ValueError("lr_df is required.")

    import scanpy as sc

    RNA_data = RNA_data.copy()
    RNA_data.index = RNA_data.index.astype(str)
    RNA_data.columns = RNA_data.columns.astype(str)

    # RNA_data: gene x cell -> adata.X: cell x gene
    adata = AnnData(RNA_data.T)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    if normalize_log:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if cell_clus is not None:
        cell_clus = cell_clus.copy()
        cell_clus.index = cell_clus.index.astype(str)

        if celltype_col not in cell_clus.columns:
            if "celltype" in cell_clus.columns:
                cell_clus = cell_clus.rename(columns={"celltype": celltype_col})
            elif "cell_type" in cell_clus.columns:
                pass
            else:
                raise ValueError("cell_clus must contain `celltype` or `cell_type`.")

        common_cells = adata.obs_names.intersection(cell_clus.index)
        adata.obs.loc[common_cells, cell_clus.columns] = cell_clus.loc[common_cells]
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")
        adata.uns["cell_annotation"] = cell_clus
    
    if cell_loc is not None:
        cell_loc = cell_loc.copy()
        cell_loc.index = cell_loc.index.astype(str)
        adata.obsm["spatial"] = cell_loc.reindex(adata.obs_names).to_numpy()
        adata.uns["cell_loc"] = cell_loc

    # Highly variable genes and marker genes
    if if_multi_layer:
        if markers is not None:
            # gene, cluster
            markers = markers.copy()
            markers.index = markers.index.astype(str)
            adata.uns["marker_genes"] = markers
        else:
            sc.pp.highly_variable_genes(
                adata,
                min_mean=0.0125,
                max_mean=3,
                min_disp=0.5,
            )

            marker_genes = None
            if groupby in adata.obs.columns:
                sc.tl.rank_genes_groups(adata, groupby, method="t-test")

                marker_records = []
                groups = adata.uns["rank_genes_groups"]["names"].dtype.names

                for group in groups:
                    genes = adata.uns["rank_genes_groups"]["names"][group][:marker_top_n]
                    marker_records.append(
                        pd.DataFrame({
                            "gene": genes,
                            "cluster": group,
                        })
                    )

                marker_genes = pd.concat(marker_records, ignore_index=True)
                adata.uns["marker_genes"] = marker_genes

    # perform L-R / L-R-TF / L-R-TF-TG database filtering
    valid_genes = set(adata.var_names)

    def _all_genes_exist(value):
        if pd.isna(value):
            return True
        return all(gene in valid_genes for gene in str(value).split("_"))

    def _check_required_columns(df, required_cols, name):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} is missing required columns: {missing_cols}")

    if l_r_tf_tg_df is not None and not if_multi_layer:
        raise ValueError("l_r_tf_tg_df requires if_multi_layer=True.")

    # lr_df is required and should always be stored
    lr_df = lr_df.copy()

    _check_required_columns(
        lr_df,
        ["Ligand_Symbol", "Receptor_Symbol"],
        "lr_df",
    )

    ligand_check = lr_df["Ligand_Symbol"].apply(_all_genes_exist)
    receptor_check = lr_df["Receptor_Symbol"].apply(_all_genes_exist)

    lr_database_filtered = lr_df.loc[
        ligand_check & receptor_check
    ].copy()

    lr_database_filtered = (
        lr_database_filtered
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if lr_database_filtered.empty:
        raise ValueError(
            "No valid L-R pairs remain after filtering by RNA_data genes."
        )

    adata.uns["L-R_db_used"] = lr_database_filtered

    if if_multi_layer and l_r_tf_tg_df is not None:
        l_r_tf_tg_df = l_r_tf_tg_df.copy()

        required_cols = [
            "Ligand_Symbol",
            "Receptor_Symbol",
            "TF_Symbol",
            "TG_Symbol",
        ]
        _check_required_columns(l_r_tf_tg_df, required_cols, "l_r_tf_tg_df")

        ligand_check = l_r_tf_tg_df["Ligand_Symbol"].apply(_all_genes_exist)
        receptor_check = l_r_tf_tg_df["Receptor_Symbol"].apply(_all_genes_exist)
        tf_check = l_r_tf_tg_df["TF_Symbol"].apply(_all_genes_exist)
        tg_check = l_r_tf_tg_df["TG_Symbol"].apply(_all_genes_exist)

        l_r_tf_tg_df_filtered = l_r_tf_tg_df.loc[
            ligand_check & receptor_check & tf_check & tg_check
        ].copy()

        l_r_tf_tg_df_filtered = (
            l_r_tf_tg_df_filtered
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if l_r_tf_tg_df_filtered.empty:
            raise ValueError(
                "No valid L-R-TF-TG paths remain after filtering by RNA_data genes."
            )

        adata.uns.setdefault("CCC", {})
        adata.uns["CCC"]["L-R-TF-TG_db"] = l_r_tf_tg_df_filtered

    elif if_multi_layer:
        if r_tf_cellcall is not None:
            adata.uns["L-R-TF_CellCall"] = r_tf_cellcall.copy()

    return adata



def construct_mc_adata_for_simulation(
    RNA_data,
    cell_clus=None,
    cell_loc=None,
    lr_df=None,
    l_r_tf_tg_df=None,
    normalize_log=False,
    log1p=False,
    celltype_col="cell_type",
):
    """
    Construct mc_adata for simulation datasets.

    This function is designed for simulated datasets with predefined databases.

    Stores:
    - adata.uns["L-R_db_used"]
    - adata.uns["CCC"]["L-R-TF-TG_db"]
    - adata.uns["cell_annotation"]
    - adata.uns["cell_loc"]
    """

    import scanpy as sc

    RNA_data = RNA_data.copy()
    RNA_data.index = RNA_data.index.astype(str)
    RNA_data.columns = RNA_data.columns.astype(str)

    # RNA_data: gene x cell -> AnnData: cell x gene
    adata = AnnData(RNA_data.T)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    if normalize_log:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if cell_clus is not None:
        cell_clus = cell_clus.copy()
        cell_clus.index = cell_clus.index.astype(str)

        if celltype_col not in cell_clus.columns:
            if "celltype" in cell_clus.columns:
                cell_clus = cell_clus.rename(columns={"celltype": celltype_col})
            elif "cell_type" in cell_clus.columns:
                pass
            else:
                raise ValueError("cell_clus must contain `celltype` or `cell_type`.")

        common_cells = adata.obs_names.intersection(cell_clus.index)
        adata.obs.loc[common_cells, cell_clus.columns] = cell_clus.loc[common_cells]
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")
        adata.uns["cell_type"] = cell_clus

    if cell_loc is not None:
        cell_loc = cell_loc.copy()
        cell_loc.index = cell_loc.index.astype(str)

        adata.obsm["spatial"] = cell_loc.reindex(adata.obs_names).to_numpy()
        adata.uns["cell_loc"] = cell_loc

    valid_genes = set(adata.var_names)

    def _all_genes_exist(value):
        if pd.isna(value):
            return True
        return all(gene in valid_genes for gene in str(value).split("_"))

    # Store/filter L-R database
    if lr_df is not None:
        lr_df = lr_df.copy()

        ligand_check = lr_df["Ligand_Symbol"].apply(_all_genes_exist)
        receptor_check = lr_df["Receptor_Symbol"].apply(_all_genes_exist)

        lr_df_filtered = lr_df.loc[ligand_check & receptor_check].copy()
        lr_df_filtered = lr_df_filtered.drop_duplicates().reset_index(drop=True)

        adata.uns["L-R_db_used"] = lr_df_filtered

    # Store/filter predefined L-R-TF-TG database
    if l_r_tf_tg_df is not None:
        l_r_tf_tg_df = l_r_tf_tg_df.copy()

        required_cols = [
            "Ligand_Symbol",
            "Receptor_Symbol",
            "TF_Symbol",
            "TG_Symbol",
        ]

        missing_cols = [
            col for col in required_cols
            if col not in l_r_tf_tg_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"l_r_tf_tg_df is missing required columns: {missing_cols}"
            )

        ligand_check = l_r_tf_tg_df["Ligand_Symbol"].apply(_all_genes_exist)
        receptor_check = l_r_tf_tg_df["Receptor_Symbol"].apply(_all_genes_exist)
        tf_check = l_r_tf_tg_df["TF_Symbol"].apply(_all_genes_exist)
        tg_check = l_r_tf_tg_df["TG_Symbol"].apply(_all_genes_exist)

        l_r_tf_tg_df_filtered = l_r_tf_tg_df.loc[
            ligand_check & receptor_check & tf_check & tg_check
        ].copy()

        l_r_tf_tg_df_filtered = (
            l_r_tf_tg_df_filtered
            .drop_duplicates()
            .reset_index(drop=True)
        )

        adata.uns.setdefault("CCC", {})
        adata.uns["CCC"]["L-R-TF-TG_db"] = l_r_tf_tg_df_filtered

        # If lr_df is not provided, derive L-R database from predefined paths.
        if lr_df is None:
            lr_df_from_path = (
                l_r_tf_tg_df_filtered[
                    ["Ligand_Symbol", "Receptor_Symbol"]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            adata.uns["L-R_db_used"] = lr_df_from_path

    return adata


def data_preparation(
    mc_adata,
    base_path,
    min_gene_cells=5,
    if_atac=False, 
    gene_info=None,
    scope=2000,
    cicero_cutoff=0.5,
    peak_peak_file=None,
):
    """
    Filter LR database, smooth RNA expression, and store CCC inputs in mc_adata.uns.

    Stores:
    - mc_adata.uns["CCC"]["lig_exp"]
    - mc_adata.uns["CCC"]["rec_exp"]
    - mc_adata.uns["CCC"]["smooth_exp"]
    - mc_adata.uns["CCC"]["L-R_db_filt1"]

    If if_atac=True, also prepares:
    - mc_adata.uns["peak-TG_links"]
    """

    lr_db = mc_adata.uns["L-R_db_used"].copy()
    cell_rep = mc_adata.uns["cell_rep"].copy()
    expmatrix = mc_adata.to_df().T  # gene x cell

    cell_rep.index = cell_rep.index.astype(str)
    expmatrix.columns = expmatrix.columns.astype(str)
    cell_rep = cell_rep.loc[expmatrix.columns]

    non_zero_counts = (expmatrix > 0).sum(axis=1)
    expmatrix_filt = expmatrix.loc[non_zero_counts >= min_gene_cells].copy()

    split_ligand_symbols = lr_db["Ligand_Symbol"].astype(str).str.split("_")
    mask_ligand = split_ligand_symbols.apply(
        lambda symbols: all(symbol in expmatrix_filt.index for symbol in symbols)
    )

    lr_db_filt = lr_db.loc[mask_ligand].copy()

    split_receptor_symbols = lr_db_filt["Receptor_Symbol"].astype(str).str.split("_")
    mask_receptor = split_receptor_symbols.apply(
        lambda symbols: all(symbol in expmatrix_filt.index for symbol in symbols)
    )

    lr_db_filt = lr_db_filt.loc[mask_receptor].copy()

    mc_adata, lig_exp, rec_exp = Preprocess_CCC_model(
        mc_adata=mc_adata,
        base_path=base_path,
        lr_database=lr_db_filt,
        cell_rep=cell_rep,
        expmatrix=expmatrix_filt,
    )

    mc_adata.uns.setdefault("CCC", {})
    mc_adata.uns["CCC"]["lig_exp"] = lig_exp
    mc_adata.uns["CCC"]["rec_exp"] = rec_exp
    mc_adata.uns["CCC"]["L-R_db_filt1"] = lr_db_filt

    if if_atac:
        from .Intra_strength import get_peak_TG_connection
        if gene_info is None:
            raise ValueError("gene_info is required when if_atac=True.")

        if peak_peak_file is not None:

            mc_adata = get_peak_TG_connection(
                mc_adata=mc_adata,
                gene_info=gene_info,
                scope=scope,
                cicero_cutoff=cicero_cutoff,
                peak_peak_file=peak_peak_file,
            )
    

    return mc_adata
