from scipy.stats import pearsonr 
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz, csc_matrix, vstack
from scipy.spatial.distance import cdist
import json
from scipy.stats import norm
import glob
import multiprocessing 
from functools import partial
from statsmodels.robust import mad


def calculate_tf_re(tf_name, re_name, sample_rna, sample_atac, tf_rep, peak_rep, tf_re_ba):
    """
    Get TF-RE score
    """
    tf_expr = sample_rna[tf_name]
    re_access = sample_atac[re_name]
    
    try:
        tf_vec = tf_rep.loc[tf_name].values
        re_vec = peak_rep.loc[re_name].values
        pcc = abs(pearsonr(tf_vec, re_vec)[0])
    except:
        pcc = 0
    
    try:
        ba_score = tf_re_ba.loc[re_name, tf_name]
    except:
        ba_score = 0
    
    return tf_expr * re_access * (pcc + ba_score)


def calculate_re_tg(re_name, tg_name, sample_rna, sample_atac, tg_re_df, gene_rep, peak_rep):
    """
    Get RE-TG score
    """
    tg_expr = sample_rna[tg_name]
    re_access = sample_atac[re_name]
    
    try:
        tg_vec = gene_rep.loc[tg_name].values
        re_vec = peak_rep.loc[re_name].values
        pcc = abs(pearsonr(tg_vec, re_vec)[0])
    except:
        pcc = 0
    
    try:
        re_tg_score = tg_re_df[(tg_re_df['genes'] == tg_name) & 
                              (tg_re_df['peaks'] == re_name)]['scores'].values[0]
    except:
        re_tg_score = 0
    
    return (re_tg_score+pcc) * re_access * tg_expr 


def calculate_tf_re_tg(tf_name, tg_name, sample_rna, sample_atac, tg_re_df, tf_rep, peak_rep, tf_re_ba, gene_rep):
    """
    Get TF-RE-TG score
    """
    associated_res = tg_re_df[tg_re_df['genes'] == tg_name]['peaks'].unique()
    
    total_score = 0
    for re_name in associated_res:
        try:
            if sample_atac[re_name] == 0:
                continue
        except:
            continue

        tf_re_score = calculate_tf_re(tf_name, re_name, sample_rna, sample_atac, tf_rep, peak_rep, tf_re_ba)
        if tf_re_score == 0:
            continue
        
        re_tg_score = calculate_re_tg(re_name, tg_name, sample_rna, sample_atac, tg_re_df, gene_rep, peak_rep)
        
        total_score += tf_re_score * re_tg_score
    
    return total_score


def calculate_all_tf_tg_scores(rna_mat, atac_mat, tg_re_df, tf_rep, peak_rep, tf_re_ba, gene_rep, path, cell_rep=None):
    """
    Get and save all samples TF-TG score
    """
    
    os.makedirs(path, exist_ok=True)
    samples = rna_mat.columns
    tfs = tf_rep.index
    tgs = tg_re_df['genes'].unique()
    
    tf_tg_columns = [f"{tf}->{tg}" for tf in tfs for tg in tgs]
    
    final_results = pd.DataFrame(
        index=samples,
        columns=tf_tg_columns,
        dtype=float
    )
    
    for sample in tqdm(samples, desc="Processing cells"):
        sample_results = pd.DataFrame(index=tfs, columns=tgs, dtype=float)
        
        sample_rna = rna_mat[sample]
        sample_atac = atac_mat.loc[sample]
        
        for tf in tfs:
            try:
                if sample_rna[tf] == 0:
                    sample_results.loc[tf] = 0  
                    continue
            except:
                sample_results.loc[tf] = 0
                continue
            for tg in tgs:
                try:
                    if sample_rna[tg] == 0:
                        sample_results.loc[tf, tg] = 0
                        continue
                except:
                    sample_results.loc[tf, tg] = 0
                    continue
                
                sample_results.loc[tf, tg] = calculate_tf_re_tg(
                    tf, tg,  
                    sample_rna, sample_atac, tg_re_df, 
                    tf_rep, peak_rep, tf_re_ba, gene_rep
                )
        
        sample_results.to_csv(os.path.join(path, f"{sample}.csv"))
        
        final_results.loc[sample] = sample_results.values.ravel()
    
    final_results.to_csv(os.path.join(path, "all_samples_tf_tg_scores.csv"))
    return final_results


def calculate_pcc_rec_tf(gene_rep, tf_rep, receptor_symbol, tf_symbol):
    """
    Get PCC(emb_rec, emb_tf)
    """
    if '_' in receptor_symbol:
        receptor_parts = receptor_symbol.split('_')
        pccs = []
        for part in receptor_parts:
            if part in gene_rep.index and tf_symbol in tf_rep.index:
                pcc, _ = pearsonr(gene_rep.loc[part], tf_rep.loc[tf_symbol])
                pccs.append(abs(pcc))
        return np.mean(pccs) if pccs else np.nan
    else:
        if receptor_symbol in gene_rep.index and tf_symbol in tf_rep.index:
            pcc, _ = pearsonr(gene_rep.loc[receptor_symbol], tf_rep.loc[tf_symbol])
            return abs(pcc)
        else:
            return np.nan                


def build_tf_tg_mapping(tg_re_df, tf_re_ba):
    """
    Build TF-TG dictonary, with TF-TG pair as key and common peaks as values
    """
    result = {}
    
    gene_peak_dict = {}
    for _, row in tg_re_df.iterrows():
        gene = row['genes']
        peak = row['peaks']
        if gene not in gene_peak_dict:
            gene_peak_dict[gene] = []
        gene_peak_dict[gene].append(peak)
    
    tf_peak_dict = {
        tf: tf_re_ba.index[tf_re_ba[tf].notna()].tolist()
        for tf in tf_re_ba.columns
    }
    
    gene_peak_sets = {gene: set(peaks) for gene, peaks in gene_peak_dict.items()}
    
    for tf, tf_peaks in tf_peak_dict.items():
        tf_peak_set = set(tf_peaks)
        for gene, gene_peak_set in gene_peak_sets.items():
            common_peaks = tf_peak_set & gene_peak_set
            if common_peaks:
                key = f"{tf}->{gene}"
                result[key] = list(common_peaks)
    
    return result


def generate_l_r_tf_pairs(l_r_df, r_tf_cellcall):
    '''
    Generate ligand-receptor-TF links based on CellChatDB and CellCallDB
    '''
    sub_r_tf_df = r_tf_cellcall[['Receptor_Symbol', 'TF_Symbol']] 
    sub_r_tf_df = sub_r_tf_df.drop_duplicates()

    r_tf_dict = {}
    for item in sub_r_tf_df['Receptor_Symbol'].tolist():
        if item not in r_tf_dict:
            r_tf_dict[item] = []
    
    for index, row in sub_r_tf_df.iterrows():
        r_tf_dict[row['Receptor_Symbol']].append(row['TF_Symbol'])

    l_lst = []
    r_lst = []
    f_lst = []

    for index, row in l_r_df.iterrows():
        if '_' in row['Receptor_Symbol']:
            r_items = row['Receptor_Symbol'].split('_')
            for item in r_items:
                if item in r_tf_dict:
                    for f_item in r_tf_dict[item]:
                        l_lst.append(row['Ligand_Symbol'])
                        r_lst.append(row['Receptor_Symbol'])
                        f_lst.append(f_item)
        else:
            if row['Receptor_Symbol'] in r_tf_dict:
                for item in r_tf_dict[row['Receptor_Symbol']]:
                    l_lst.append(row['Ligand_Symbol'])
                    r_lst.append(row['Receptor_Symbol'])
                    f_lst.append(item)

    l_r_tf_df = pd.DataFrame({'Ligand_Symbol': l_lst, 'Receptor_Symbol': r_lst, 'TF_Symbol': f_lst})
    l_r_tf_df = l_r_tf_df.drop_duplicates()
    l_r_tf_df = l_r_tf_df.reset_index(drop=True)
    
    return l_r_tf_df


def generate_l_r_tf_tg_pairs(l_r_tf_df, tf_tg_common_peaks):
    '''
    Generate ligand-receptor-TF-TG links
    '''
    l_lst = []
    r_lst = []
    f_lst = []
    g_lst = []
    for index, row in tqdm(l_r_tf_df.iterrows(), desc="Processing L-R-TF-TG pairs"):
        l = row['Ligand_Symbol']
        r = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        
        matching_keys = [key for key in tf_tg_common_peaks.keys() if key.startswith(f"{tf}->")]
        if matching_keys:
            for key in matching_keys:
                tg = key.split('->')[1]
                l_lst.append(l)
                r_lst.append(r)
                f_lst.append(tf)
                g_lst.append(tg)
    L_R_TF_TG_df = pd.DataFrame({
        'Ligand_Symbol': l_lst,
        'Receptor_Symbol': r_lst,
        'TF_Symbol': f_lst,
        'TG_Symbol': g_lst
    })
    return L_R_TF_TG_df       
        

def calculate_r_tf_tg_cor(
    gene_rep: pd.DataFrame,
    tf_rep: pd.DataFrame,
    cell_rep: pd.DataFrame,
    receptors: List[str],
    tf_tg_pairs: List[str],
    reg_dir: str = "R_TF_TG_Reg",  
    output_dir: str = "output_cells_cor"
) -> Tuple[Dict[str, csr_matrix], csr_matrix]:
    """
    Get correlation score
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/global_row_names.json", "w") as f:
        json.dump(receptors, f)
    with open(f"{output_dir}/global_col_names.json", "w") as f:
        json.dump(tf_tg_pairs, f)
    
    cells = cell_rep.index.tolist()
    
    gene_index = {gene: idx for idx, gene in enumerate(gene_rep.index)}
    tf_index = {tf: idx for idx, tf in enumerate(tf_rep.index)}
    
    parsed_pairs = []
    for pair in tf_tg_pairs:
        tf, tg = pair.split("->")
        parsed_pairs.append((tf, tg, gene_index.get(tg, -1)))
    
    print("Precomputing PCC matrices...")
    cell_gene_pcc_matrix = 1 - cdist(cell_rep.values, gene_rep.values, metric='correlation')
    cell_tf_pcc_matrix = 1 - cdist(cell_rep.values, tf_rep.values, metric='correlation')
    cell_gene_pcc_matrix = np.abs(cell_gene_pcc_matrix)  
    cell_tf_pcc_matrix = np.abs(cell_tf_pcc_matrix)      
    
    cell_results = {}
    combined_data = np.zeros((len(cells), len(receptors)*len(tf_tg_pairs)))
    
    for cell_idx, cell in enumerate(tqdm(cells, desc="Processing cells")):
        reg_path = f"{reg_dir}/{cell}.npz"
        if not os.path.exists(reg_path):
            continue
            
        reg_sparse = load_npz(reg_path)
        reg_coo = reg_sparse.tocoo() 
        
        cell_data = np.zeros((len(receptors), len(tf_tg_pairs)))
        
        for rec_idx, pair_idx, reg_val in zip(reg_coo.row, reg_coo.col, reg_coo.data):
            receptor = receptors[rec_idx]
            tf, tg, tg_idx = parsed_pairs[pair_idx]
            
            if '_' in receptor:
                items = receptor.split('_')
                item_pccs = []
                for item in items:
                    if item in gene_index:
                        item_pccs.append(cell_gene_pcc_matrix[cell_idx, gene_index[item]])
                rec_pcc = np.mean(item_pccs) if item_pccs else 0
            else:
                rec_pcc = cell_gene_pcc_matrix[cell_idx, gene_index[receptor]] if receptor in gene_index else 0
            
            tf_pcc = cell_tf_pcc_matrix[cell_idx, tf_index[tf]] if tf in tf_index else 0
            tg_pcc = cell_gene_pcc_matrix[cell_idx, tg_idx] if tg_idx != -1 else 0
            
            cell_data[rec_idx, pair_idx] = rec_pcc * tf_pcc * tg_pcc
        
        cell_sparse = csr_matrix(cell_data)
        cell_results[cell] = cell_sparse
        save_npz(f"{output_dir}/{cell}.npz", cell_sparse)
        
        combined_data[cell_idx, :] = cell_data.ravel()
    
    combined_sparse = csr_matrix(combined_data)
    save_npz(f"{output_dir}/combined_results.npz", combined_sparse)
    
    with open(f"{output_dir}/combined_row_names.json", "w") as f:
        json.dump(cells, f)
    with open(f"{output_dir}/combined_col_names.json", "w") as f:
        json.dump([f"{rec}->{pair}" for rec in receptors for pair in tf_tg_pairs], f)
    
    return cell_results, combined_sparse


def calculate_r_tf_tg_strength(
    cell_rep: pd.DataFrame,
    cor_dir: str,
    reg_dir: str,
    output_dir: str = "output_cells_total"
) -> Tuple[Dict[str, csr_matrix], csr_matrix]:
    """
    Both correlation and regulation score
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{cor_dir}/global_row_names.json", "r") as f:
        receptors = json.load(f)
    with open(f"{cor_dir}/global_col_names.json", "r") as f:
        tf_tg_pairs = json.load(f)
    
    with open(f"{output_dir}/global_row_names.json", "w") as f:
        json.dump(receptors, f)
    with open(f"{output_dir}/global_col_names.json", "w") as f:
        json.dump(tf_tg_pairs, f)
    
    cells = cell_rep.index.tolist()
    
    cell_results = {}
    combined_data = []
    
    for cell in tqdm(cells, desc="Calculating total scores"):
        cor_path = f"{cor_dir}/{cell}.npz"
        cor_mat = load_npz(cor_path)
        
        reg_path = f"{reg_dir}/{cell}.npz"
        reg_mat = load_npz(reg_path)
        
        total_mat = cor_mat.multiply(reg_mat)
        cell_results[cell] = total_mat
        
        save_npz(f"{output_dir}/{cell}.npz", total_mat)
        
        combined_data.append(total_mat.toarray().ravel())
    
    combined_sparse = csr_matrix(np.vstack(combined_data))
    save_npz(f"{output_dir}/combined_results.npz", combined_sparse)
    
    with open(f"{output_dir}/combined_row_names.json", "w") as f:
        json.dump(cells, f)
    with open(f"{output_dir}/combined_col_names.json", "w") as f:
        json.dump([f"{rec}->{pair}" for rec in receptors for pair in tf_tg_pairs], f)
    
    return cell_results, combined_sparse



def calculate_r_tf_tg_reg(
    cell_rep: pd.DataFrame,
    tf_tg_score_df: pd.DataFrame,
    rec_tf_pcc: pd.DataFrame,
    rna_mat: pd.DataFrame,
    output_dir: str = "output_cells"
) -> Tuple[Dict[str, csr_matrix], csr_matrix]:
    """
    Get receptor-tf-tg regulation score for each cell, save as sparse format
    """
    assert tf_tg_score_df.index.isin(cell_rep.index).all(), "Cell names mismatch"
    assert {"Receptor_Symbol", "TF_Symbol"}.issubset(rec_tf_pcc.columns), "rec_tf_pcc Missing necessary columns"
    
    os.makedirs(output_dir, exist_ok=True)
    
    def get_receptor_expression(receptor: str, cell: str) -> float:
        """receptor expression"""
        if '_' in receptor:
            parts = receptor.split('_')
            exprs = [rna_mat.loc[p, cell] if p in rna_mat.index else 0 for p in parts]
            return np.mean(exprs)
        return rna_mat.loc[receptor, cell] if receptor in rna_mat.index else 0
    
    rec_tf_map = rec_tf_pcc.groupby("Receptor_Symbol")["TF_Symbol"].agg(set).to_dict()
    tf_to_pairs = defaultdict(list)
    for pair in tf_tg_score_df.columns:
        tf = pair.split("->")[0]
        tf_to_pairs[tf].append(pair)
    pcc_map = rec_tf_pcc.set_index(["Receptor_Symbol", "TF_Symbol"])["scores"].to_dict()
    
    all_receptors = list(rec_tf_map.keys())
    all_tf_tg_pairs = tf_tg_score_df.columns.tolist()
    cells = tf_tg_score_df.index.tolist()
    
    with open(f"{output_dir}/global_row_names.json", "w") as f:
        json.dump(all_receptors, f)
    with open(f"{output_dir}/global_col_names.json", "w") as f:
        json.dump(all_tf_tg_pairs, f)
    
    cell_results = {}
    combined_data = np.zeros((len(cells), len(all_receptors)*len(all_tf_tg_pairs)))
    
    for i, cell in enumerate(tqdm(cells, desc="Processing cells")):
        cell_scores = tf_tg_score_df.loc[cell]
        cell_data = np.zeros((len(all_receptors), len(all_tf_tg_pairs)))
        
        for rec_idx, receptor in enumerate(all_receptors):
            rec_exp = get_receptor_expression(receptor, cell)
            if rec_exp == 0:
                continue
            for tf in rec_tf_map.get(receptor, set()):
                if tf not in tf_to_pairs:
                    continue
                
                pcc = pcc_map.get((receptor, tf), 0)
                pair_indices = [tf_tg_score_df.columns.get_loc(p) for p in tf_to_pairs[tf]]
                cell_data[rec_idx, pair_indices] = rec_exp * pcc * cell_scores.iloc[pair_indices].values
        
        cell_sparse = csr_matrix(cell_data)
        cell_results[cell] = cell_sparse
        save_npz(f"{output_dir}/{cell}.npz", cell_sparse)
        
        combined_data[i, :] = cell_data.ravel()
    
    combined_sparse = csr_matrix(combined_data)
    save_npz(f"{output_dir}/combined_results.npz", combined_sparse)
    
    with open(f"{output_dir}/combined_row_names.json", "w") as f:
        json.dump(cells, f) 
    with open(f"{output_dir}/combined_col_names.json", "w") as f:
        json.dump([f"{rec}->{pair}" for rec in all_receptors for pair in all_tf_tg_pairs], f)
    
    return cell_results, combined_sparse



def calculate_l_r_tf_tg_strength(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "ligand_cascade_results"
) -> None:
    """
    Get ligand-receptor-tf-tg score, save as signaling transduction response score for ligands
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        combined_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")

    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    ligands = l_r_tf_tg_df['Ligand_Symbol'].unique()
    
    for ligand in tqdm(ligands, desc="Processing ligands"):
        ligand_subdf = l_r_tf_tg_df[l_r_tf_tg_df['Ligand_Symbol'] == ligand]
        result_data = {}
        
        for _, row in ligand_subdf.iterrows():
            receptor = row['Receptor_Symbol']
            tf = row['TF_Symbol']
            tg = row['TG_Symbol']
            l_r_pair = f"{ligand}->{receptor}"
            r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
            
            if l_r_pair in zero_lr_pairs:
                continue
                
            if l_r_pair not in ccc_lrp_df.columns:
                continue
                
            if r_tf_tg_pair not in r_tf_tg_to_col:
                continue
                
            col_idx = r_tf_tg_to_col[r_tf_tg_pair]
            
            inter_signal = ccc_lrp_df[l_r_pair].values  
            intra_signal = combined_sparse[:, col_idx].toarray().flatten()  
            
            total_signal = inter_signal * intra_signal
            pathway = f"{ligand}->{receptor}->{tf}->{tg}"
            result_data[pathway] = total_signal
        
        if result_data:
            result_df = pd.DataFrame(result_data, index=cell_names)
            result_df.to_csv(f"{output_dir}/{ligand}.csv", index=True)
    
    print(f"Processing completed. Results saved to {output_dir}")

    
    
def calculate_l_r_tf_tg_strength_by_tg(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "TG_cascade_results"
) -> None:
    """
    Get ligand-receptor-tf-tg score, save as signaling transduction response score for TGs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    combined_sparse = load_npz(combined_npz_path).tocsc()
    with open(global_row_names_path, 'r') as f:
        cell_names = json.load(f)
    with open(global_col_names_path, 'r') as f:
        r_tf_tg_pairs = json.load(f)
    ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)

    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}
    
    l_r_pairs = ccc_lrp_df.columns

    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    all_tgs = l_r_tf_tg_df["TG_Symbol"].unique()
    for tg in tqdm(all_tgs, desc="Processing TGs"):
        tg_subset = l_r_tf_tg_df[l_r_tf_tg_df["TG_Symbol"] == tg]
        tg_data = {}

        for _, row in tg_subset.iterrows():
            ligand = row["Ligand_Symbol"]
            receptor = row["Receptor_Symbol"]
            tf = row["TF_Symbol"]
            l_r_pair = f"{ligand}->{receptor}"
            r_tf_tg_pair = f"{receptor}->{tf}->{tg}"

            if l_r_pair in zero_lr_pairs:
                continue

            if l_r_pair not in l_r_pairs or r_tf_tg_pair not in r_tf_tg_to_col:
                continue

            l_r_signal = ccc_lrp_df[l_r_pair].values
            r_tf_tg_col = r_tf_tg_to_col[r_tf_tg_pair]
            r_tf_tg_signal = combined_sparse[:, r_tf_tg_col].toarray().flatten()  
            total_signal = l_r_signal * r_tf_tg_signal  

            path = f"{ligand}->{receptor}->{tf}->{tg}"
            tg_data[path] = total_signal

        if tg_data:
            tg_df = pd.DataFrame(tg_data, index=cell_names)
            tg_df.to_csv(os.path.join(output_dir, f"{tg}.csv"), index=True)

    print(f"All TG networks saved to {output_dir}")


    
def scell_to_celltype_mean(
    cell_type_path: str,      
    input_dir: str,           
    output_dir: str,          
    cell_type_col: str = "cell_type",  
    csv_index_col: int = 0,   
    verbose: bool = True      
) -> None:
    """
    将single-cell level的预测强度计算均值得到cell-type level
    """
    # 读取样本类型信息
    cell_type_df = pd.read_csv(cell_type_path, index_col=0, sep="\t")
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            if verbose:
                print(f"Processing: {filename}")
            
            # 读取CSV文件
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath, index_col=csv_index_col)
            
            # 检查样本名是否匹配
            common_samples = df.index.intersection(cell_type_df.index)
            if len(common_samples) == 0:
                if verbose:
                    print(f"  Warning: No matching samples found. Skipping.")
                continue
            
            # 添加样本类型列并分组计算均值
            df = df.loc[common_samples]
            df[cell_type_col] = cell_type_df.loc[common_samples, cell_type_col]
            ct_df = df.groupby(cell_type_col).mean()
            
            # 保存结果
            output_filepath = os.path.join(output_dir, filename)
            ct_df.to_csv(output_filepath)
    
    if verbose:
        print("All files processed!") 



def generate_background_l_r_tf_tg_strength(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "random_ligand_cascade_results",
    random_seed: int = 42
) -> None:
    """
    Generate background data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))
    
    print("Generating background data by row permutation...")
    np.random.seed(random_seed) 
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    background_matrices = []

    for _ in range(10):
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        
        background_matrices.append(background_sparse)

    final_background_sparse = vstack(background_matrices)

    print("Final background sparse matrix shape:", final_background_sparse.shape)

    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    ligands = l_r_tf_tg_df['Ligand_Symbol'].unique()
    
    for ligand in tqdm(ligands, desc="Processing ligands"):
        ligand_subdf = l_r_tf_tg_df[l_r_tf_tg_df['Ligand_Symbol'] == ligand]
        result_data = {}
        
        for _, row in ligand_subdf.iterrows():
            receptor = row['Receptor_Symbol']
            tf = row['TF_Symbol']
            tg = row['TG_Symbol']
            l_r_pair = f"{ligand}->{receptor}"
            r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
            
            if l_r_pair in zero_lr_pairs:
                continue
                
            if l_r_pair not in ccc_lrp_df.columns:
                continue
                
            if r_tf_tg_pair not in r_tf_tg_to_col:
                continue
                
            col_idx = r_tf_tg_to_col[r_tf_tg_pair]

            inter_signal = ccc_lrp_df[l_r_pair].values  
            intra_signal = final_background_sparse[:, col_idx].toarray().flatten()  
            
            total_signal = inter_signal * intra_signal
            pathway = f"{ligand}->{receptor}->{tf}->{tg}"
            result_data[pathway] = total_signal
        
        if result_data:
            result_df = pd.DataFrame(result_data, index=final_cell_names)
            result_df.to_csv(f"{output_dir}/{ligand}.csv", index=True)
    
    print(f"Processing completed. Results saved to {output_dir}")



def generate_background_l_r_tf_tg_strength_parallel(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "random_ligand_cascade_results",
    random_seed: int = 42,
    n_processes: int = None
) -> None:
    """
    Generate background data by parallel
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))
    
    print("Generating background data by row permutation...")
    np.random.seed(random_seed) 
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    background_matrices = []

    for _ in range(10):
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        background_matrices.append(background_sparse)

    final_background_sparse = vstack(background_matrices)
    print("Final background sparse matrix shape:", final_background_sparse.shape)

    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    ligands = l_r_tf_tg_df['Ligand_Symbol'].unique()
    
    process_ligand_partial = partial(
        _process_single_ligand,
        l_r_tf_tg_df=l_r_tf_tg_df,
        ccc_lrp_df=ccc_lrp_df,
        r_tf_tg_to_col=r_tf_tg_to_col,
        zero_lr_pairs=zero_lr_pairs,
        final_background_sparse=final_background_sparse,
        final_cell_names=final_cell_names,
        output_dir=output_dir
    )
    
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1  
    
    print(f"Starting parallel processing with {n_processes} processes...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.map(process_ligand_partial, ligands)
    
    print(f"Processing completed. Results saved to {output_dir}")

def _process_single_ligand(
    ligand: str,
    l_r_tf_tg_df: pd.DataFrame,
    ccc_lrp_df: pd.DataFrame,
    r_tf_tg_to_col: dict,
    zero_lr_pairs: set,
    final_background_sparse: csc_matrix,
    final_cell_names: list,
    output_dir: str
) -> None:
    ligand_subdf = l_r_tf_tg_df[l_r_tf_tg_df['Ligand_Symbol'] == ligand]
    result_data = {}
    
    for _, row in ligand_subdf.iterrows():
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        
        if l_r_pair in zero_lr_pairs:
            continue
        if l_r_pair not in ccc_lrp_df.columns:
            continue
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
            
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        inter_signal = ccc_lrp_df[l_r_pair].values
        intra_signal = final_background_sparse[:, col_idx].toarray().flatten()
        total_signal = inter_signal * intra_signal
        pathway = f"{ligand}->{receptor}->{tf}->{tg}"
        result_data[pathway] = total_signal
    
    if result_data:
        result_df = pd.DataFrame(result_data, index=final_cell_names)
        result_df.to_csv(f"{output_dir}/{ligand}.csv", index=True)



def generate_background_l_r_tf_tg_strength_parallel_with_lexp(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    expression_matrix: pd.DataFrame,  
    output_dir: str = "random_ligand_cascade_results",
    random_seed: int = 42,
    n_processes: int = None
) -> None:

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
        
        # 确保表达矩阵的细胞名与其他数据一致
        expression_matrix = expression_matrix[cell_names]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background data by row permutation...")
    np.random.seed(random_seed)
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    background_matrices = []

    for _ in range(10):
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        background_sparse = csc_matrix(
            (original_sparse.data, 
             permuted_indices[original_sparse.indices], 
             original_sparse.indptr),
            shape=original_sparse.shape
        )
        background_matrices.append(background_sparse)

    final_background_sparse = vstack(background_matrices)
    final_cell_names = cell_names * 10
    
    # 4. 构建受体-TF-TG到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    # 5. 预计算全零的l-r对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 准备多进程处理
    ligands = l_r_tf_tg_df['Ligand_Symbol'].unique()
    
    # 创建partial函数固定共享参数
    process_ligand_partial = partial(
        _process_single_ligand_background,
        l_r_tf_tg_df=l_r_tf_tg_df,
        ccc_lrp_df=ccc_lrp_df,
        r_tf_tg_to_col=r_tf_tg_to_col,
        zero_lr_pairs=zero_lr_pairs,
        final_background_sparse=final_background_sparse,
        final_cell_names=final_cell_names,
        expression_matrix=expression_matrix,  # 传入表达矩阵
        output_dir=output_dir
    )
    
    # 设置进程数
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1
    
    print(f"Starting parallel processing with {n_processes} processes...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.map(process_ligand_partial, ligands)
    
    print(f"Processing completed. Results saved to {output_dir}")

def _process_single_ligand_background(
    ligand: str,
    l_r_tf_tg_df: pd.DataFrame,
    ccc_lrp_df: pd.DataFrame,
    r_tf_tg_to_col: dict,
    zero_lr_pairs: set,
    final_background_sparse: csc_matrix,
    final_cell_names: list,
    expression_matrix: pd.DataFrame,  # 新增表达矩阵参数
    output_dir: str
) -> None:
    """处理单个ligand的辅助函数（背景数据版本）"""
    ligand_subdf = l_r_tf_tg_df[l_r_tf_tg_df['Ligand_Symbol'] == ligand]
    result_data = {}
    
    # 检查ligand是否在表达矩阵中
    if ligand not in expression_matrix.index:
        if '_' in ligand:
            ligand_components = ligand.split('_')
            available_components = [comp for comp in ligand_components if comp in expression_matrix.index]
            if not available_components:
                # print(f"Warning: Ligand {ligand} not found in expression matrix")
                return
            ligand_expression = expression_matrix.loc[available_components].mean(axis=0).values
            # print(f"Using average expression for ligand {ligand}: {available_components}")
        else:
            # print(f"Warning: Ligand {ligand} not found in expression matrix")
            return
    else:
        ligand_expression = expression_matrix.loc[ligand].values
    
    # 获取ligand表达值（重复10次以匹配背景数据）
    # ligand_expression = np.tile(ligand_expression, 10)
    
    for _, row in ligand_subdf.iterrows():
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        
        if l_r_pair in zero_lr_pairs:
            continue
        if l_r_pair not in ccc_lrp_df.columns:
            continue
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
            
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        # inter_signal = np.tile(ccc_lrp_df[l_r_pair].values, 10)  # 重复10次匹配背景数据
        inter_signal = ccc_lrp_df[l_r_pair].values
        intra_signal = final_background_sparse[:, col_idx].toarray().flatten()
        
        # 计算总信号强度（乘以ligand表达值）
        total_signal = inter_signal * intra_signal * ligand_expression
        pathway = f"{ligand}->{receptor}->{tf}->{tg}"
        result_data[pathway] = total_signal
    
    if result_data:
        result_df = pd.DataFrame(result_data, index=final_cell_names)
        result_df.to_csv(f"{output_dir}/{ligand}.csv", index=True)



def generate_background_l_r_tf_tg_strength_by_tg(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "random_TG_cascade_results",
    random_seed: int = 42
) -> None:
    """
    通过行排列生成按TG分组的背景数据
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")

    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background matrix by row permutation...")
    np.random.seed(random_seed)
    
    # 获取非零行索引
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    # 新的非零行应该是一个和原来相同长度的索引列表，但具有不同的行索引
    new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
    # 创建排列索引
    permuted_indices = np.arange(original_sparse.shape[0])
    permuted_indices[nonzero_rows] = new_nonzero_rows
    # permuted_indices[nonzero_rows] = np.random.permutation(nonzero_rows) #仅扰动稀疏矩阵中非零行的顺序
    
    # 应用排列
    background_sparse = csc_matrix(
        (original_sparse.data, 
         permuted_indices[original_sparse.indices], 
         original_sparse.indptr),
        shape=original_sparse.shape
    )

    # 4. 构建映射
    # 4.1 受体-TF-TG 到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}
    
    # 4.2 Ligand-Receptor 到列索引的映射（ccc_lrp_df）
    l_r_pairs = ccc_lrp_df.columns

    # 5. 预计算全零的 L-R 对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 按 TG 分组处理背景数据
    all_tgs = l_r_tf_tg_df["TG_Symbol"].unique()
    
    for tg in tqdm(all_tgs, desc="Processing TGs (background)"):
        # 6.1 获取该 TG 的所有路径
        tg_subset = l_r_tf_tg_df[l_r_tf_tg_df["TG_Symbol"] == tg]
        tg_data = {}

        for _, row in tg_subset.iterrows():
            ligand = row["Ligand_Symbol"]
            receptor = row["Receptor_Symbol"]
            tf = row["TF_Symbol"]
            l_r_pair = f"{ligand}->{receptor}"
            r_tf_tg_pair = f"{receptor}->{tf}->{tg}"

            # 跳过全零的 L-R 对
            if l_r_pair in zero_lr_pairs:
                continue

            # 检查路径是否存在
            if l_r_pair not in l_r_pairs or r_tf_tg_pair not in r_tf_tg_to_col:
                continue

            # 6.2 计算背景信号强度
            l_r_signal = ccc_lrp_df[l_r_pair].values  # L->R 信号（保持不变）
            r_tf_tg_col = r_tf_tg_to_col[r_tf_tg_pair]
            r_tf_tg_signal = background_sparse[:, r_tf_tg_col].toarray().flatten()  # R->TF->TG 信号（背景）
            total_signal = l_r_signal * r_tf_tg_signal  # L->R->TF->TG 信号（背景）

            # 存储路径和信号
            path = f"{ligand}->{receptor}->{tf}->{tg}"
            tg_data[path] = total_signal

        # 6.3 保存该 TG 的背景数据
        if tg_data:
            tg_df = pd.DataFrame(tg_data, index=cell_names)
            tg_df.to_csv(os.path.join(output_dir, f"{tg}.csv"), index=True)

    print(f"Background data generation completed. Results saved to {output_dir}")
    


def generate_background_l_r_tf_tg_strength_by_tg_parallel(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "random_TG_cascade_results",
    random_seed: int = 42,
    n_processes: int = None
) -> None:
    """
    通过行排列生成按TG分组的背景数据（并行版本）
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))

    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background matrix by row permutation...")
    np.random.seed(random_seed)
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    background_matrices = []
    
    for _ in range(10):
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        background_matrices.append(background_sparse)

    final_background_sparse = vstack(background_matrices)
    print("Final background sparse matrix shape:", final_background_sparse.shape)

    # 4. 构建映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}
    l_r_pairs = ccc_lrp_df.columns

    # 5. 预计算全零的 L-R 对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 准备并行处理
    all_tgs = l_r_tf_tg_df["TG_Symbol"].unique()
    
    # 创建partial函数固定共享参数
    process_tg_partial = partial(
        _process_single_tg_background,
        l_r_tf_tg_df=l_r_tf_tg_df,
        ccc_lrp_df=ccc_lrp_df,
        r_tf_tg_to_col=r_tf_tg_to_col,
        zero_lr_pairs=zero_lr_pairs,
        final_background_sparse=final_background_sparse,
        final_cell_names=final_cell_names,
        l_r_pairs=l_r_pairs,
        output_dir=output_dir
    )
    
    # 设置进程数
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1
    
    print(f"Starting parallel processing with {n_processes} processes...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        list(tqdm(
            pool.imap(process_tg_partial, all_tgs),
            total=len(all_tgs),
            desc="Processing TGs (background)"
        ))
        # pool.map(process_tg_partial, all_tgs)
    
    print(f"Background data generation completed. Results saved to {output_dir}")
    

def _process_single_tg_background(
    tg: str,
    l_r_tf_tg_df: pd.DataFrame,
    ccc_lrp_df: pd.DataFrame,
    r_tf_tg_to_col: dict,
    zero_lr_pairs: set,
    final_background_sparse: csc_matrix,
    final_cell_names: list,
    l_r_pairs: list,
    output_dir: str
) -> None:
    """处理单个TG背景数据的辅助函数"""
    tg_subset = l_r_tf_tg_df[l_r_tf_tg_df["TG_Symbol"] == tg]
    tg_data = {}

    for _, row in tg_subset.iterrows():
        ligand = row["Ligand_Symbol"]
        receptor = row["Receptor_Symbol"]
        tf = row["TF_Symbol"]
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"

        if l_r_pair in zero_lr_pairs:
            continue
        if l_r_pair not in l_r_pairs or r_tf_tg_pair not in r_tf_tg_to_col:
            continue

        l_r_signal = ccc_lrp_df[l_r_pair].values
        r_tf_tg_col = r_tf_tg_to_col[r_tf_tg_pair]
        r_tf_tg_signal = final_background_sparse[:, r_tf_tg_col].toarray().flatten()
        total_signal = l_r_signal * r_tf_tg_signal

        path = f"{ligand}->{receptor}->{tf}->{tg}"
        tg_data[path] = total_signal

    if tg_data:
        tg_df = pd.DataFrame(tg_data, index=final_cell_names)
        tg_df.to_csv(os.path.join(output_dir, f"{tg}.csv"), index=True)
    
    
def generate_background_l_r_tf_tg_strength_by_tg_parallel_with_lexp(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    expression_matrix: pd.DataFrame,
    output_dir: str = "random_TG_cascade_results",
    random_seed: int = 42,
    n_processes: int = None
) -> None:
    """
    通过行排列生成按TG分组的背景数据（并行版本）
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
        
        expression_matrix = expression_matrix[cell_names]  
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))

    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background matrix by row permutation...")
    np.random.seed(random_seed)
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    background_matrices = []
    
    for _ in range(10):
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        background_matrices.append(background_sparse)

    final_background_sparse = vstack(background_matrices)
    print("Final background sparse matrix shape:", final_background_sparse.shape)

    # 4. 构建映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}
    l_r_pairs = ccc_lrp_df.columns

    # 5. 预计算全零的 L-R 对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 准备并行处理
    all_tgs = l_r_tf_tg_df["TG_Symbol"].unique()
    
    # 创建partial函数固定共享参数
    process_tg_partial = partial(
        _process_single_tg_background_with_lexp,
        l_r_tf_tg_df=l_r_tf_tg_df,
        ccc_lrp_df=ccc_lrp_df,
        r_tf_tg_to_col=r_tf_tg_to_col,
        zero_lr_pairs=zero_lr_pairs,
        final_background_sparse=final_background_sparse,
        final_cell_names=final_cell_names,
        l_r_pairs=l_r_pairs,
        expression_matrix=expression_matrix,  
        output_dir=output_dir
    )
    
    # 设置进程数
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1
    
    print(f"Starting parallel processing with {n_processes} processes...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        list(tqdm(
            pool.imap(process_tg_partial, all_tgs),
            total=len(all_tgs),
            desc="Processing TGs (background)"
        ))
        # pool.map(process_tg_partial, all_tgs)
    
    print(f"Background data generation completed. Results saved to {output_dir}")    
    
    

def _process_single_tg_background_with_lexp(
    tg: str,
    l_r_tf_tg_df: pd.DataFrame,
    ccc_lrp_df: pd.DataFrame,
    r_tf_tg_to_col: dict,
    zero_lr_pairs: set,
    final_background_sparse: csc_matrix,
    final_cell_names: list,
    l_r_pairs: list,
    expression_matrix: pd.DataFrame,  
    output_dir: str
) -> None:
    """处理单个TG背景数据的辅助函数"""
    tg_subset = l_r_tf_tg_df[l_r_tf_tg_df["TG_Symbol"] == tg]
    tg_data = {}

    for _, row in tg_subset.iterrows():
        ligand = row["Ligand_Symbol"]
        receptor = row["Receptor_Symbol"]
        tf = row["TF_Symbol"]
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        
        if ligand not in expression_matrix.index:
            if '_' in ligand:
                ligand_components = ligand.split('_')
                available_components = [comp for comp in ligand_components if comp in expression_matrix.index]
                if not available_components:
                    # print(f"Warning: Ligand {ligand} not found in expression matrix")
                    continue
                ligand_expression = expression_matrix.loc[available_components].mean(axis=0).values
                # print(f"Using average expression for ligand {ligand}: {available_components}")
            else:
                continue
        else:
            ligand_expression = expression_matrix.loc[ligand].values
        
        if l_r_pair in zero_lr_pairs:
            continue
        if l_r_pair not in l_r_pairs or r_tf_tg_pair not in r_tf_tg_to_col:
            continue

        l_r_signal = ccc_lrp_df[l_r_pair].values
        r_tf_tg_col = r_tf_tg_to_col[r_tf_tg_pair]
        r_tf_tg_signal = final_background_sparse[:, r_tf_tg_col].toarray().flatten()
        total_signal = l_r_signal * r_tf_tg_signal * ligand_expression  

        path = f"{ligand}->{receptor}->{tf}->{tg}"
        tg_data[path] = total_signal

    if tg_data:
        tg_df = pd.DataFrame(tg_data, index=final_cell_names)
        tg_df.to_csv(os.path.join(output_dir, f"{tg}.csv"), index=True)




def get_Inter_Strength(
    att_df: pd.DataFrame,
    base_path: str,
    selected_cell_type: str
) :
    nei_adj = pd.read_csv(f"{base_path}/Nei_adj_{selected_cell_type}.csv", index_col=None, header=None, sep="\t")
    rna_mat = pd.read_csv(f"{base_path}CCC/expression_smooth_{selected_cell_type}.txt", header=0, index_col=0, sep="\t")
    
    lig_mat = pd.DataFrame(index=rna_mat.index, columns=rna_mat.columns)
    for i, row in tqdm(nei_adj.iterrows(), desc="Processing rows", total=nei_adj.shape[0]):
        cell_name = rna_mat.columns[i]
        sender_idxs = row[1:].dropna().astype(int)
        if len(sender_idxs) == 0:
            lig_mat.iloc[i] = 0 
        else:
            sender_idxs = sender_idxs - 1
            sender_expr = rna_mat.iloc[:, sender_idxs]
            mean_expr = sender_expr.mean(axis=1)
            lig_mat[cell_name] = mean_expr.values
    lig_mat_minmax = (lig_mat - lig_mat.min()) / (lig_mat.max() - lig_mat.min())
    rna_mat_minmax = (rna_mat - rna_mat.min()) / (rna_mat.max() - rna_mat.min())
    
    lig_expr_df = pd.DataFrame(index=att_df.index, columns=att_df.columns)
    rec_expr_df = pd.DataFrame(index=att_df.index, columns=att_df.columns)
    
    for pair in att_df.columns:
        lig, rec = pair.split('->')
        
        if '_' in lig:
            valid_lig_components = [gene for gene in lig_components if gene in lig_mat_minmax.index]
            if len(valid_lig_components) > 0:

                lig_expr = lig_mat_minmax.loc[valid_lig_components].mean(axis=0)
                lig_expr_df[pair] = lig_expr.values
            else:
                lig_expr_df[pair] = 0
            lig_components = lig.split('_')
            lig_expr_df[pair] = lig_mat_minmax[lig_components].min(axis=1)
        else:
            lig_expr_df[pair] = lig_mat_minmax.loc[lig] if lig in lig_mat_minmax.index else 0
        
        if '_' in rec:
            rec_components = rec.split('_')
            valid_rec_components = [gene for gene in rec_components if gene in rna_mat_minmax.index]
            if len(valid_rec_components) > 0:
                rec_expr = rna_mat_minmax.loc[valid_rec_components].mean(axis=0)
                rec_expr_df[pair] = rec_expr.values
            else:
                rec_expr_df[pair] = 0
        else:
            rec_expr_df[pair] = rna_mat_minmax.loc[rec] if rec in rna_mat_minmax.index else 0
    
    # result = att_df * lig_expr_df * rec_expr_df
    result = att_df * lig_expr_df
    result.to_csv(f"{base_path}Inter_strength_{selected_cell_type}.txt", sep="\t")
    # weight_df = lig_expr_df * rec_expr_df
    weight_df = lig_expr_df
    weight_df.to_csv(f"{base_path}Inter_strength_weight_{selected_cell_type}.txt", sep="\t")
    
    return result, weight_df

   


def load_background_inter(base_path, file_pattern="CCC_module_LRP_strength_run_*.txt"):
    file_paths = glob.glob(base_path + file_pattern)
    
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file, sep='\t', index_col='Unnamed: 0')
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=False)
    return combined_df




def Identify_significant_lr_pairs(
    background_inter_df: pd.DataFrame,
    sample_inter_df: pd.DataFrame,
    output_path: str = "sc_Significant_LR.csv",
    z_critical: float = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Identify the significant pairs (based on background and obsevered data to calculate Z-score, α=0.05, Z=1.645)
    """
    if z_critical is None:
        z_critical = norm.ppf(1 - alpha)
    
    combined_df = pd.concat([background_inter_df, sample_inter_df], axis=0)
    
    global_mean = combined_df.mean(axis=0)  
    global_std = combined_df.std(axis=0)    
    
    sample_zscore = (sample_inter_df - global_mean) / global_std  
    
    all_sample_results = []
    for sample_name, sample_data in tqdm(sample_inter_df.iterrows(), desc=f"Processing samples"):
        sample_zscore_row = sample_zscore.loc[sample_name]
        significant_mask = sample_zscore_row > z_critical  
        significant_pathways = sample_zscore_row[significant_mask].index.tolist()  
        
        sample_names = [sample_name] * len(significant_pathways)
        sample_zscores = sample_zscore_row[significant_mask].values  
        sample_interactions = sample_inter_df.loc[sample_name, significant_pathways].values  
        sample_results = pd.DataFrame({
            "Sample_Name": sample_names,
            "LR_Symbol": significant_pathways,
            "Inter_Score": sample_interactions,
            "Z_Score": sample_zscores
        })
        sample_results = sample_results.sort_values("Z_Score", ascending=False)  
        all_sample_results.append(sample_results)
    all_sample_results = pd.concat(all_sample_results, ignore_index=True)
    all_sample_results.to_csv(output_path, index=False)
    print(f"Significant_L->R pairs saved to {output_path}")
    
    return all_sample_results


def Identify_significant_paths(
    background_inter_df: pd.DataFrame,
    sample_inter_df: pd.DataFrame,
    output_path: str = "sc_Significant_LR.csv",
    z_critical: float = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Identify the significant paths (based on background and obsevered data to calculate Z-score, α=0.05, Z=1.645)
    """
    if z_critical is None:
        z_critical = norm.ppf(1 - alpha)
    
    combined_df = pd.concat([background_inter_df, sample_inter_df], axis=0)
    
    global_mean = combined_df.mean(axis=0)  
    global_std = combined_df.std(axis=0)    
    
    sample_zscore = (sample_inter_df - global_mean) / global_std  
    
    all_sample_results = []
    for sample_name, sample_data in tqdm(sample_inter_df.iterrows(), desc=f"Processing samples"):
        sample_zscore_row = sample_zscore.loc[sample_name]
        significant_mask = sample_zscore_row > z_critical  
        significant_pathways = sample_zscore_row[significant_mask].index.tolist()  
        
        sample_names = [sample_name] * len(significant_pathways)
        sample_zscores = sample_zscore_row[significant_mask].values  
        sample_interactions = sample_inter_df.loc[sample_name, significant_pathways].values  
        sample_results = pd.DataFrame({
            "Sample_Name": sample_names,
            "Path_Symbol": significant_pathways,
            "Comm_Score": sample_interactions,
            "Z_Score": sample_zscores
        })
        sample_results = sample_results.sort_values("Z_Score", ascending=False)  
        all_sample_results.append(sample_results)
    all_sample_results = pd.concat(all_sample_results, ignore_index=True)
    all_sample_results.to_csv(output_path, index=False)
    print(f"Significant_L->R->TF->TG paths saved to {output_path}")
    
    return all_sample_results



def Identify_significant_lr_pairs_celltype(sif_df, celltype, agg_method='mean'):
    if not isinstance(sif_df, pd.DataFrame) or not isinstance(celltype, pd.DataFrame):
        raise ValueError("inputs need pandas DataFrame")
    
    merged_df = sif_df.merge(
        celltype, 
        left_on='Sample_Name', 
        right_index=True, 
        how='left'  
    )
    
    if isinstance(agg_method, dict):
        result_df = merged_df.groupby(['LR_Symbol', 'celltype']).agg(agg_method)
    else:
        result_df = merged_df.groupby(['LR_Symbol', 'celltype']).agg({
            'Inter_Score': agg_method,
            'Z_Score': agg_method
        })
    
    result_df = result_df.reset_index()
    result_df.rename(columns={'celltype': 'Cell_Type'}, inplace=True)
    
    return result_df



def Identify_significant_paths_celltype(sif_df, celltype, agg_method='mean'):
    if not isinstance(sif_df, pd.DataFrame) or not isinstance(celltype, pd.DataFrame):
        raise ValueError("inputs need pandas DataFrame")
    
    merged_df = sif_df.merge(
        celltype, 
        left_on='Sample_Name', 
        right_index=True, 
        how='left'  
    )
    
    if isinstance(agg_method, dict):
        result_df = merged_df.groupby(['Path_Symbol', 'celltype']).agg(agg_method)
    else:
        result_df = merged_df.groupby(['Path_Symbol', 'celltype']).agg({
            'Comm_Score': agg_method,
            'Z_Score': agg_method
        })
    
    result_df = result_df.reset_index()
    result_df.rename(columns={'celltype': 'Cell_Type'}, inplace=True)
    
    return result_df



def mad_based_score(values):
    eps = 1e-8
    med = np.median(values)
    dev = abs(values - med)
    mad_val = mad(values)
    return dev / (1.4826 * mad_val + eps)

def Identify_volatile_lr_pairs_celltype(data, threshold=None, method='mad'):
    volatility_df = pd.DataFrame(
        index=data.index,
        columns=data.columns,
        dtype=float
    )
    outlier_report = {}

    for path in data.columns:
        values = data[path].values

        if method == 'median':
            scores = np.abs(values - np.median(values)) / np.median(values)
            volatility_df[path] = scores
            path_threshold = 0.3 if threshold is None else threshold
        elif method == 'mad':
            scores = mad_based_score(values)
            volatility_df[path] = scores
            path_threshold = 2.5 if threshold is None else threshold
        elif method == 'ratio':
            total = sum(values)
            if total == 0:
                scores = [0] * len(values)
            else:
                scores = [x/total for x in values]
            volatility_df[path] = scores
            path_threshold = 0.3 if threshold is None else threshold
        else:
            raise ValueError("method must be 'median' or 'mad'")

        outliers = volatility_df[path][volatility_df[path] > path_threshold]
        if not outliers.empty:
            outlier_report[path] = list(zip(outliers.index, outliers.values))
            
    volatility_bin = (volatility_df > path_threshold).astype(int)
    
    rows, cols = np.where(volatility_bin == 1)

    celltype_lst = volatility_bin.index[rows].tolist()
    lr_symbol_lst = volatility_bin.columns[cols].tolist()
    inter_score_lst = data.values[rows, cols].tolist()

    result_df = pd.DataFrame({
        'LR_Symbol': lr_symbol_lst,
        'Cell_Type': celltype_lst,
        'Inter_Score': inter_score_lst,
        'Z_Score': 2  
    })
    result_df = result_df.reset_index(drop=True)

    return result_df, volatility_df, volatility_bin, outlier_report



def Identify_concat_lr_pairs_celltype(sig_LR_pair_celltype,vola_LR_pair_celltype,out_path):
    merged = pd.merge(vola_LR_pair_celltype, 
                 sig_LR_pair_celltype[['LR_Symbol', 'Cell_Type']],
                 on=['LR_Symbol', 'Cell_Type'],
                 how='left',
                 indicator=True)

    to_add = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
    sig_LR_pair_celltype_updated = pd.concat([sig_LR_pair_celltype, to_add], ignore_index=True)
    sig_LR_pair_celltype_updated = sig_LR_pair_celltype_updated.sort_values(['LR_Symbol', 'Cell_Type'])
    sig_LR_pair_celltype_updated = sig_LR_pair_celltype_updated.reset_index(drop=True)
    
    sig_LR_pair_celltype_updated.to_csv(out_path, sep=",", index=False)
    print(f"sig_LR_pair_celltype_updated saved to {out_path}")
    return sig_LR_pair_celltype_updated



def calculate_l_r_tf_tg_strength_cellwise(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = "cellwise_cascade_results"
) -> None:
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        combined_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")

    # 3. 构建受体-TF-TG到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    # 4. 预计算全零的l-r对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 5. 处理每个ligand
    print("Processing all ligand-receptor-TF-TG paths...")
    result_data = {}
        
    for _, row in tqdm(l_r_tf_tg_df.iterrows(), total=len(l_r_tf_tg_df), desc="Calculating paths"):
        ligand = row['Ligand_Symbol']
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        path = f"{ligand}->{receptor}->{tf}->{tg}"
        
        # 跳过全零的l-r对
        if l_r_pair in zero_lr_pairs:
            continue
            
        # 检查l-r对是否存在
        if l_r_pair not in ccc_lrp_df.columns:
            continue
            
        # 检查r-tf-tg对是否存在
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
            
        # 获取列索引
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        
        # 获取信号
        inter_signal = ccc_lrp_df[l_r_pair].values  # l->r信号
        intra_signal = combined_sparse[:, col_idx].toarray().flatten()  # r->tf->tg信号
        
        # 计算总信号强度
        total_signal = inter_signal * intra_signal
        result_data[path] = total_signal
        
    # 保存该ligand的结果
    if result_data:
        result_df = pd.DataFrame(result_data, index=cell_names)
        output_path = os.path.join(output_dir, "cellwise_cascade_results.csv")
        result_df.to_csv(output_path, index=True)
        print(f"Processing completed. Results saved to {output_path}")
    else:
        print("No valid ligand-receptor-TF-TG paths found.")
    


def calculate_l_r_tf_tg_strength_cellwise_with_lexp(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    expression_matrix: pd.DataFrame,  # 新增：表达矩阵 (cells x genes)
    output_dir: str = None,
    output_file: str = "cellwise_cascade_results.csv"
) -> None:

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        combined_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
        expression_matrix = expression_matrix[cell_names]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")

    # 3. 构建受体-TF-TG到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    # 4. 预计算全零的l-r对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 5. 处理所有ligand-receptor-TF-TG路径
    print("Processing all ligand-receptor-TF-TG paths...")
    result_data = {}
    
    for _, row in tqdm(l_r_tf_tg_df.iterrows(), total=len(l_r_tf_tg_df), desc="Calculating paths"):
        ligand = row['Ligand_Symbol']
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        path = f"{ligand}->{receptor}->{tf}->{tg}"
        
        # 跳过全零的l-r对
        if l_r_pair in zero_lr_pairs:
            continue
            
        # 检查l-r对是否存在
        if l_r_pair not in ccc_lrp_df.columns:
            continue
            
        # 检查r-tf-tg对是否存在
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
            
        # 检查ligand是否在表达矩阵中
        if ligand not in expression_matrix.index:
            if '_' in ligand:
                ligand_components = ligand.split('_')
                available_components = [comp for comp in ligand_components if comp in expression_matrix.index]
                if not available_components:
                    print(f"Warning: Ligand {ligand} not found in expression matrix")
                    continue
                ligand_expression = expression_matrix.loc[available_components].mean(axis=0).values
                # print(f"Using average expression for ligand components: {ligand_components}")
            else:
                print(f"Warning: Ligand {ligand} not found in expression matrix")
                continue
        else:
            ligand_expression = expression_matrix.loc[ligand].values
            
        
        # 获取列索引
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        
        # 获取信号
        inter_signal = ccc_lrp_df[l_r_pair].values  # l->r信号
        intra_signal = combined_sparse[:, col_idx].toarray().flatten()  # r->tf->tg信号
        
        # 计算总信号强度（乘以ligand表达值）
        total_signal = inter_signal * intra_signal * ligand_expression
        result_data[path] = total_signal
    
    # 保存结果
    if result_data:
        result_df = pd.DataFrame(result_data, index=cell_names)
        output_path = os.path.join(output_dir+output_file)
        result_df.to_csv(output_path, index=True)
        print(f"Processing completed. Results saved to {output_path}")
    else:
        print("No valid ligand-receptor-TF-TG paths found.")




def generate_background_l_r_tf_tg_strength_cellwise(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    output_dir: str = None,
    random_seed: int = 42
) -> None:

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))
    
    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background data by row permutation...")
    np.random.seed(random_seed) 
    # 获取非零行索引
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    # 创建一个列表存储多个背景稀疏矩阵
    background_matrices = []

    # 重复生成10次
    for _ in range(10):
        # 新的非零行应该是一个和原来相同长度的索引列表，但具有不同的行索引
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        
        # 创建排列索引
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        # 应用排列生成背景稀疏矩阵
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        
        # 将生成的矩阵添加到列表中
        background_matrices.append(background_sparse)

    # 将多个背景稀疏矩阵按行堆叠
    final_background_sparse = vstack(background_matrices)

    # 查看最终生成的稀疏矩阵的形状
    print("Final background sparse matrix shape:", final_background_sparse.shape)

    # 4. 构建受体-TF-TG到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    # 5. 预计算全零的l-r对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 处理每个ligand
    print("Processing all ligand-receptor-TF-TG paths on randomized background...")
    result_data = {}
        
    for _, row in tqdm(l_r_tf_tg_df.iterrows(), total=len(l_r_tf_tg_df), desc="Calculating background paths"):
        ligand = row['Ligand_Symbol']
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        path = f"{ligand}->{receptor}->{tf}->{tg}"
        
        # 跳过全零的l-r对
        if l_r_pair in zero_lr_pairs:
            continue
            
        # 检查l-r对是否存在
        if l_r_pair not in ccc_lrp_df.columns:
            continue
            
        # 检查r-tf-tg对是否存在
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
            
        # 获取列索引
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        
        # 获取信号
        inter_signal = ccc_lrp_df[l_r_pair].values  # l->r信号
        intra_signal = final_background_sparse[:, col_idx].toarray().flatten()  # r->tf->tg信号
        
        # 计算总信号强度
        total_signal = inter_signal * intra_signal
        result_data[path] = total_signal
    
    # 保存该ligand的结果
    if result_data:
        result_df = pd.DataFrame(result_data, index=final_cell_names)
        output_path = os.path.join(output_dir+"background_cellwise_cascade_results.csv")
        result_df.to_csv(output_path, index=True)
        print(f"Unified background result saved to: {output_path}")
    else:
        print("No valid background paths found.")
        
        
        
def generate_background_l_r_tf_tg_strength_cellwise_with_lexp(
    l_r_tf_tg_df: pd.DataFrame,
    combined_npz_path: str,
    global_row_names_path: str,
    global_col_names_path: str,
    ccc_lrp_path: str,
    expression_matrix: pd.DataFrame,
    output_dir: str = None,
    output_file: str = "background_cellwise_cascade_results.csv",
    random_seed: int = 42
) -> None:
    """
    通过行排列生成背景数据，计算每个l-r-tf-tg的strength
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading data...")
    try:
        original_sparse = load_npz(combined_npz_path).tocsc()
        with open(global_row_names_path, 'r') as f:
            cell_names = json.load(f)
        with open(global_col_names_path, 'r') as f:
            r_tf_tg_pairs = json.load(f)
        ccc_lrp_df = pd.read_csv(ccc_lrp_path, sep='\t', index_col=0)
        expression_matrix = expression_matrix[cell_names]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully")
    
    final_cell_names = cell_names * 10
    print("Final cell names length:", len(final_cell_names))
    
    # 3. 生成背景稀疏矩阵（行排列）
    print("Generating background data by row permutation...")
    np.random.seed(random_seed) 
    # 获取非零行索引
    nonzero_rows = np.unique(original_sparse.nonzero()[0])
    # 创建一个列表存储多个背景稀疏矩阵
    background_matrices = []

    # 重复生成10次
    for _ in range(10):
        # 新的非零行应该是一个和原来相同长度的索引列表，但具有不同的行索引
        new_nonzero_rows = np.random.choice(np.arange(original_sparse.shape[0]), size=len(nonzero_rows), replace=False)
        
        # 创建排列索引
        permuted_indices = np.arange(original_sparse.shape[0])
        permuted_indices[nonzero_rows] = new_nonzero_rows
        
        # 应用排列生成背景稀疏矩阵
        background_sparse = csc_matrix(
            (original_sparse.data, 
            permuted_indices[original_sparse.indices], 
            original_sparse.indptr),
            shape=original_sparse.shape
        )
        
        # 将生成的矩阵添加到列表中
        background_matrices.append(background_sparse)

    # 将多个背景稀疏矩阵按行堆叠
    final_background_sparse = vstack(background_matrices)

    # 查看最终生成的稀疏矩阵的形状
    print("Final background sparse matrix shape:", final_background_sparse.shape)

    # 4. 构建受体-TF-TG到列索引的映射
    r_tf_tg_to_col = {pair: idx for idx, pair in enumerate(r_tf_tg_pairs)}

    # 5. 预计算全零的l-r对
    print("Identifying zero L-R pairs...")
    zero_lr_pairs = set()
    for col in ccc_lrp_df.columns:
        if np.all(ccc_lrp_df[col] == 0):
            zero_lr_pairs.add(col)
    print(f"Found {len(zero_lr_pairs)} zero L-R pairs to skip")

    # 6. 处理每个ligand
    print("Processing all ligand-receptor-TF-TG paths on randomized background...")
    result_data = {}
        
    for _, row in tqdm(l_r_tf_tg_df.iterrows(), total=len(l_r_tf_tg_df), desc="Calculating background paths"):
        ligand = row['Ligand_Symbol']
        receptor = row['Receptor_Symbol']
        tf = row['TF_Symbol']
        tg = row['TG_Symbol']
        l_r_pair = f"{ligand}->{receptor}"
        r_tf_tg_pair = f"{receptor}->{tf}->{tg}"
        path = f"{ligand}->{receptor}->{tf}->{tg}"
        
        # 跳过全零的l-r对
        if l_r_pair in zero_lr_pairs:
            continue
            
        # 检查l-r对是否存在
        if l_r_pair not in ccc_lrp_df.columns:
            continue
            
        # 检查r-tf-tg对是否存在
        if r_tf_tg_pair not in r_tf_tg_to_col:
            continue
        
        # 检查ligand是否在表达矩阵中
        if ligand not in expression_matrix.index:
            if '_' in ligand:
                ligand_components = ligand.split('_')
                available_components = [comp for comp in ligand_components if comp in expression_matrix.index]
                if not available_components:
                    print(f"Warning: Ligand {ligand} not found in expression matrix")
                    continue
                ligand_expression = expression_matrix.loc[available_components].mean(axis=0).values
                # print(f"Using average expression for ligand components: {ligand_components}")
            else:
                print(f"Warning: Ligand {ligand} not found in expression matrix")
                continue
        else:
            ligand_expression = expression_matrix.loc[ligand].values   


            
        # 获取列索引
        col_idx = r_tf_tg_to_col[r_tf_tg_pair]
        
        # 获取信号
        inter_signal = ccc_lrp_df[l_r_pair].values  # l->r信号
        intra_signal = final_background_sparse[:, col_idx].toarray().flatten()  # r->tf->tg信号
        
        # 计算总信号强度
        total_signal = inter_signal * intra_signal * ligand_expression
        result_data[path] = total_signal
    
    # 保存该ligand的结果
    if result_data:
        result_df = pd.DataFrame(result_data, index=final_cell_names)
        output_path = os.path.join(output_dir+output_file)
        result_df.to_csv(output_path, index=True)
        print(f"Unified background result saved to: {output_path}")
    else:
        print("No valid background paths found.")   
        
        

# def Indentify_volatile_paths_celltype(data, threshold=None, method='mad'):
#     volatility_df = pd.DataFrame(
#         index=data.index,
#         columns=data.columns,
#         dtype=float
#     )
#     outlier_report = {}

#     for path in data.columns:
#         values = data[path].values

#         if method == 'median':
#             median = np.median(values)
#             adjusted_median = median if median != 0 else 1e-10
#             scores = np.abs(values - median) / adjusted_median
#             # scores = np.abs(values - np.median(values)) / np.median(values)
#             volatility_df[path] = scores
#             path_threshold = 0.3 if threshold is None else threshold
#         elif method == 'mad':
#             scores = mad_based_score(values)
#             volatility_df[path] = scores
#             path_threshold = 2.5 if threshold is None else threshold
#         elif method == 'mean':
#             mean = np.mean(values)
#             adjusted_mean = mean if mean != 0 else 1e-10
#             scores = np.abs(values - mean) / adjusted_mean
#             volatility_df[path] = scores
#             path_threshold = 0.3 if threshold is None else threshold
#         else:
#             raise ValueError("method must be 'median' or 'mad' or 'mean")

#         outliers = volatility_df[path][volatility_df[path] > path_threshold]
#         if not outliers.empty:
#             outlier_report[path] = list(zip(outliers.index, outliers.values))
            
#     volatility_bin = (volatility_df > path_threshold).astype(int)
    
#     rows, cols = np.where(volatility_bin == 1)

#     celltype_lst = volatility_bin.index[rows].tolist()
#     lr_symbol_lst = volatility_bin.columns[cols].tolist()
#     inter_score_lst = data.values[rows, cols].tolist()

#     result_df = pd.DataFrame({
#         'Path_Symbol': lr_symbol_lst,
#         'Cell_Type': celltype_lst,
#         'Comm_Score': inter_score_lst,
#         'Z_Score': 2  
#     })
#     result_df = result_df.reset_index(drop=True)

#     return result_df, volatility_df, volatility_bin, outlier_report


def Identify_volatile_paths_celltype(data, threshold=None, method='mad'):
    volatility_df = pd.DataFrame(
        index=data.index,
        columns=data.columns,
        dtype=float
    )
    outlier_report = {}

    for path in data.columns:
        values = data[path].values

        if method == 'median':
            median = np.median(values)
            adjusted_median = median if median != 0 else 1e-10
            scores = np.abs(values - median) / adjusted_median
            # scores = np.abs(values - np.median(values)) / np.median(values)
            volatility_df[path] = scores
            path_threshold = 0.3 if threshold is None else threshold
        elif method == 'mad':
            scores = mad_based_score(values)
            volatility_df[path] = scores
            path_threshold = 2.5 if threshold is None else threshold
        elif method == 'mean':
            mean = np.mean(values)
            adjusted_mean = mean if mean != 0 else 1e-10
            scores = np.abs(values - mean) / adjusted_mean
            volatility_df[path] = scores
            path_threshold = 0.3 if threshold is None else threshold
        elif method == 'ratio':
            total = sum(values)
            if total == 0:
                scores = [0] * len(values)
            else:
                scores = [x/total for x in values]
            volatility_df[path] = scores
            path_threshold = 0.3 if threshold is None else threshold
        else:
            raise ValueError("method must be 'median' or 'mad' or 'mean" or "ratio")

        outliers = volatility_df[path][volatility_df[path] > path_threshold]
        if not outliers.empty:
            outlier_report[path] = list(zip(outliers.index, outliers.values))
            
    volatility_bin = (volatility_df > path_threshold).astype(int)
    
    rows, cols = np.where(volatility_bin == 1)

    celltype_lst = volatility_bin.index[rows].tolist()
    lr_symbol_lst = volatility_bin.columns[cols].tolist()
    inter_score_lst = data.values[rows, cols].tolist()

    result_df = pd.DataFrame({
        'Path_Symbol': lr_symbol_lst,
        'Cell_Type': celltype_lst,
        'Comm_Score': inter_score_lst,
        'Z_Score': 2  
    })
    result_df = result_df.reset_index(drop=True)

    return result_df, volatility_df, volatility_bin, outlier_report



def Identify_concat_paths_celltype(sig_LR_pair_celltype,vola_LR_pair_celltype,out_path):
    merged = pd.merge(vola_LR_pair_celltype, 
                 sig_LR_pair_celltype[['Path_Symbol', 'Cell_Type']],
                 on=['Path_Symbol', 'Cell_Type'],
                 how='left',
                 indicator=True)

    to_add = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
    sig_LR_pair_celltype_updated = pd.concat([sig_LR_pair_celltype, to_add], ignore_index=True)
    sig_LR_pair_celltype_updated = sig_LR_pair_celltype_updated.sort_values(['Path_Symbol', 'Cell_Type'])
    sig_LR_pair_celltype_updated = sig_LR_pair_celltype_updated.reset_index(drop=True)
    
    sig_LR_pair_celltype_updated.to_csv(out_path, sep=",", index=False)
    print(f"sig_Path_pair_celltype_updated saved to {out_path}")
    return sig_LR_pair_celltype_updated