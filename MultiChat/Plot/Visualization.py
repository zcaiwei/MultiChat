from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from collections import defaultdict
from tqdm import tqdm
from itertools import product
from scipy import stats
import os



   

def get_sender_adj(Nei_adj, cell_type):
    num_cells = len(cell_type)
    adj = np.zeros((num_cells, num_cells))
    for _, row in Nei_adj.iterrows():
        sender = int(row[0])
        neighbors = row[1:].dropna().astype(int).tolist()
        for neighbor in neighbors:
            adj[sender][neighbor] = 1
    return adj


def get_Sig_all_vectors(pathway_name, Sig_LR, cell_type, coord, adj, mode = 'Pathway_Name'):
    if mode == 'Pathway_Name':
        path_df = Sig_LR[Sig_LR['Pathway_Name'] == pathway_name]
    elif mode == 'LR_Symbol':
        path_df = Sig_LR[Sig_LR['LR_Symbol'] == pathway_name]
    current_df = path_df[['Sample_Name', 'Inter_Score', 'Z_Score']].groupby('Sample_Name', as_index=False).mean()
    sample_lst = []
    interccc_lst = []
    zscore_lst = []
    sender_lst = []
    sender_x_lst = []
    sender_y_lst = []
    receiver_x_lst = []
    receiver_y_lst = []
    for index, row in current_df.iterrows():
        sample_name = row['Sample_Name']
        interccc = row['Inter_Score']
        zscore = row['Z_Score']
        idx = cell_type.index.get_loc(sample_name)
        senders = np.where(adj[idx] > 0)[0]
        for sender in senders:
            sample_lst.append(sample_name)
            interccc_lst.append(interccc)
            zscore_lst.append(zscore)
            sender_lst.append(sender)
            sender_x_lst.append(coord.iloc[sender]['x'])
            sender_y_lst.append(coord.iloc[sender]['y'])
            receiver_x_lst.append(coord.iloc[idx]['x'])
            receiver_y_lst.append(coord.iloc[idx]['y'])
    res_df = pd.DataFrame({
        'Sample_Name': sample_lst,
        'Inter_Score': interccc_lst,
        'Z_Score': zscore_lst,
        'Sender': sender_lst,
        'Sender_x': sender_x_lst,
        'Sender_y': sender_y_lst,
        'Receiver_x': receiver_x_lst,
        'Receiver_y': receiver_y_lst
    })
    return res_df


def get_weighted_vector(res):
    x = res['Sender_x'].values[0]
    y = res['Sender_y'].values[0]
    x1 = res['Receiver_x'].values
    y1 = res['Receiver_y'].values
    weights = res['Inter_Score'].values
    total_weight = np.sum(weights)
    ratios = weights / total_weight
    weighted_x = np.sum(ratios * x1)
    weighted_y = np.sum(ratios * y1)
    mean_strength = np.mean(weights)
    return x, y, weighted_x, weighted_y, mean_strength


def get_Sig_weighted_one_vector(res_all_df):
    sample_lst = []
    interccc_lst = []
    zscore_lst = []
    sender_lst = []
    sender_x_lst = []
    sender_y_lst = []
    receiver_x_lst = []
    receiver_y_lst = []
    finded_idx = {}
    for index, row in res_all_df.iterrows():
        sample_name = row['Sample_Name']
        zscore = row['Z_Score']
        sender_idx = row['Sender']
        if sender_idx in finded_idx:
            continue
        finded_idx[sender_idx] = 1
        sub_df = res_all_df[res_all_df['Sender'] == sender_idx]
        x, y, weighted_x, weighted_y, mean_strength = get_weighted_vector(sub_df)
        sample_lst.append(sample_name)
        interccc_lst.append(mean_strength)
        zscore_lst.append(zscore)
        sender_lst.append(sender_idx)
        sender_x_lst.append(x)
        sender_y_lst.append(y)
        receiver_x_lst.append(weighted_x)
        receiver_y_lst.append(weighted_y)
    res_weighted_df = pd.DataFrame({
        'Sample_Name': sample_lst,
        'Inter_Score': interccc_lst,
        'Z_Score': zscore_lst,
        'Sender': sender_lst,
        'Sender_x': sender_x_lst,
        'Sender_y': sender_y_lst,
        'Receiver_x': receiver_x_lst,
        'Receiver_y': receiver_y_lst
    })
    return res_weighted_df


def get_two_hop_cascade_vectors(Sig_path, condition, one_hop_paths):
    focus_sig_path = Sig_path[condition].copy()
    one_hop_sig_path = Sig_path[Sig_path['path_symbol'] == one_hop_paths]
    one_hop_sig_path = Sig_path[Sig_path['path_symbol'] == one_hop_paths]
    one_hop_sig_path = one_hop_sig_path.rename(columns={'from_cell':'X_name',
        'to_cell':'Relay_name', 
        'source':'X_Source',
        'target':'Relay_Target',
        'path_symbol':'X_Path_Symbol', 
        'comm_score':'X_Comm_Score',
        'z_score':'X_Z_Score'})
    merged_df = pd.merge(
        left=one_hop_sig_path,
        right=focus_sig_path,
        left_on='Relay_name',
        right_on='from_cell',
        how='inner'
    )
    x_relay_y = merged_df[['X_name', 'Relay_name', 'to_cell', 'X_Path_Symbol','path_symbol', 'X_Comm_Score', 'comm_score']].copy()
    x_relay_y.rename(columns={
        'to_cell': 'Y_name',
        'X_Path_Symbol': 'XR_Path_Symbol',
        'path_symbol': 'RY_Path_Symbol',
        'X_Comm_Score': 'XR_Comm_Score',
        'comm_score': 'RY_Comm_Score'
    }, inplace=True)
    
    return x_relay_y



def stable_cauchy_combination(pvalues, min_p=1e-16):
    if len(pvalues) == 0:
        return np.nan, np.nan

    pvalues = np.array(pvalues)
    pvalues = np.clip(pvalues, min_p, 1.0)  # First layer of protection: limiting input range

    try:
        t = np.tan((0.5 - pvalues) * np.pi)
        t = np.clip(t, -1e10, 1e10)  # Second layer of protection: limiting extreme tan values to avoid explosion
        combined_stat = np.sum(t)
        combined_p = 0.5 - np.arctan(combined_stat) / np.pi
        combined_p = np.clip(combined_p, min_p, 1.0)  # Third layer of protection: limiting output range
        return combined_p, combined_stat
    except Exception:
        return min_p, np.nan


def get_cauchy_res(ct_target, condition,sig_path,sig_ccc,cell_type,top_n=20):
    focus_sig_ccc_res = sig_path[condition].copy()
    top_20_paths = focus_sig_ccc_res['path_symbol'].value_counts().head(top_n).index.tolist()
    print(f"Fixed top {top_n} paths for {ct_target} target:")
    print(top_20_paths)

    # Create cauchy combination results for all cell types
    cur_df = pd.DataFrame(
        np.random.uniform(0.05, 1, size=(len(cell_type.index), len(top_20_paths))),
        index=cell_type.index.tolist(),
        columns=top_20_paths
    )
    for path in top_20_paths:
        temp = sig_ccc[sig_ccc['Path_Symbol'] == path].copy()
        z_score_lst = temp['Z_Score'].tolist()
        sample_lst = temp['Sample_Name'].tolist()
        p_value_lst = [stats.norm.sf(z_score) for z_score in z_score_lst]
        cur_df.loc[sample_lst, path] = p_value_lst
    matrix_df = cur_df.copy()
    merged_df = matrix_df.merge(cell_type[['cell_type']], left_index=True, right_index=True, how='left')
    res = []    
    for t, sub_df in merged_df.groupby("cell_type"):
        for path in matrix_df.columns:
            pvals = sub_df[path].dropna().values
            p_comb, combined_stat = stable_cauchy_combination(pvals)
            res.append({
                "type": t,
                "pathway": path,
                "combined_pvalue": p_comb,
                "combined_stat": combined_stat
            })
    cauchy_df = pd.DataFrame(res)
    cauchy_df = cauchy_df.sort_values("combined_pvalue")
    cauchy_df['neg_log10_p'] = -np.log10(cauchy_df['combined_pvalue'])
    
    return cauchy_df
    
    
    
def get_Sig_all_vectors_mlpath(pathway_name, Sig_LR, cell_type, coord, adj):
    path_df = Sig_LR[Sig_LR['Path_Symbol'].str.contains(pathway_name)]
    current_df = path_df[['Sample_Name', 'Comm_Score', 'Z_Score']].groupby('Sample_Name', as_index=False).mean()
    sample_lst = []
    interccc_lst = []
    zscore_lst = []
    sender_lst = []
    sender_x_lst = []
    sender_y_lst = []
    receiver_x_lst = []
    receiver_y_lst = []
    for index, row in current_df.iterrows():
        sample_name = row['Sample_Name']
        interccc = row['Comm_Score']
        zscore = row['Z_Score']
        idx = cell_type.index.get_loc(sample_name)
        senders = np.where(adj[idx] > 0)[0]
        for sender in senders:
            sample_lst.append(sample_name)
            interccc_lst.append(interccc)
            zscore_lst.append(zscore)
            sender_lst.append(sender)
            sender_x_lst.append(coord.iloc[sender]['x'])
            sender_y_lst.append(coord.iloc[sender]['y'])
            receiver_x_lst.append(coord.iloc[idx]['x'])
            receiver_y_lst.append(coord.iloc[idx]['y'])
    res_df = pd.DataFrame({
        'Sample_Name': sample_lst,
        'Comm_Score': interccc_lst,
        'Z_Score': zscore_lst,
        'Sender': sender_lst,
        'Sender_x': sender_x_lst,
        'Sender_y': sender_y_lst,
        'Receiver_x': receiver_x_lst,
        'Receiver_y': receiver_y_lst
    })
    return res_df



def get_weighted_vector_mlpath(res):
    x = res['Sender_x'].values[0]
    y = res['Sender_y'].values[0]
    x1 = res['Receiver_x'].values
    y1 = res['Receiver_y'].values
    weights = res['Comm_Score'].values
    total_weight = np.sum(weights)
    ratios = weights / total_weight
    weighted_x = np.sum(ratios * x1)
    weighted_y = np.sum(ratios * y1)
    mean_strength = np.mean(weights)
    return x, y, weighted_x, weighted_y, mean_strength

def get_Sig_weighted_one_vector_mlpath(res_all_df):
    sample_lst = []
    interccc_lst = []
    zscore_lst = []
    sender_lst = []
    sender_x_lst = []
    sender_y_lst = []
    receiver_x_lst = []
    receiver_y_lst = []
    finded_idx = {}
    for index, row in res_all_df.iterrows():
        sample_name = row['Sample_Name']
        zscore = row['Z_Score']
        sender_idx = row['Sender']
        if sender_idx in finded_idx:
            continue
        finded_idx[sender_idx] = 1
        sub_df = res_all_df[res_all_df['Sender'] == sender_idx]
        x, y, weighted_x, weighted_y, mean_strength = get_weighted_vector_mlpath(sub_df)
        sample_lst.append(sample_name)
        interccc_lst.append(mean_strength)
        zscore_lst.append(zscore)
        sender_lst.append(sender_idx)
        sender_x_lst.append(x)
        sender_y_lst.append(y)
        receiver_x_lst.append(weighted_x)
        receiver_y_lst.append(weighted_y)
    res_weighted_df = pd.DataFrame({
        'Sample_Name': sample_lst,
        'Comm_Score': interccc_lst,
        'Z_Score': zscore_lst,
        'Sender': sender_lst,
        'Sender_x': sender_x_lst,
        'Sender_y': sender_y_lst,
        'Receiver_x': receiver_x_lst,
        'Receiver_y': receiver_y_lst
    })
    return res_weighted_df






def plot_communication_double_panel(
    strength_df,          
    coord_df,             
    res_weighted_df,      
    color_map,            
    pathway_name,         
    figpath,              
    strength_scale=10,    
    arrow_rad=0.2,        
    point_size=80,        
    figsize=(24, 10),     
    dpi=300,             
    bbox_inches='tight'   
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.subplots_adjust(wspace=0.3, top=0.88, right=0.8)
    
    cmap_colors = ("#D3D3D3", "#AD2DE9") 
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    
    scatter1 = ax1.scatter(
        strength_df['x'], strength_df['y'], 
        c=strength_df['Comm_Score'],
        cmap=custom_cmap, 
        s=point_size, 
        linewidths=0
    )
    
    for _, row in res_weighted_df.iterrows():
        strength = row['Comm_Score']
        lw = max(0.5, strength * strength_scale)
        arrowstyle = ArrowStyle("Simple", head_length=1, head_width=1.5, tail_width=0.1)
        arrow = FancyArrowPatch(
            (row['Sender_x'], row['Sender_y']),
            (row['Receiver_x'], row['Receiver_y']),
            connectionstyle=f"arc3,rad={arrow_rad}", 
            arrowstyle=arrowstyle, 
            color='black', 
            lw=lw,
            mutation_scale=2
        )
        ax1.add_patch(arrow)
    
    # ax1.set_title('Communication Strength Background')
    fig.colorbar(scatter1, ax=ax1, label='Strength level')
    
    scatter2 = ax2.scatter(
        coord_df['x'], coord_df['y'],
        c=coord_df['color'], 
        alpha=0.9, 
        s=point_size, 
        linewidths=0
    )
    
    for _, row in res_weighted_df.iterrows():
        strength = row['Comm_Score']
        lw = max(0.5, strength * strength_scale)
        arrowstyle = ArrowStyle("Simple", head_length=1, head_width=1.5, tail_width=0.1)
        arrow = FancyArrowPatch(
            (row['Sender_x'], row['Sender_y']),
            (row['Receiver_x'], row['Receiver_y']),
            connectionstyle=f"arc3,rad={arrow_rad}", 
            arrowstyle=arrowstyle,
            color='black', 
            lw=lw,
            mutation_scale=2
        )
        ax2.add_patch(arrow)
    
    handles = [
        plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            label=celltype, 
            markerfacecolor=color, 
            markersize=10
        ) for celltype, color in color_map.items()
    ]
    ax2.legend(
        handles=handles, 
        title='Cell types', 
        bbox_to_anchor=(1.36, 1), 
        loc='upper right'
    )
    # ax2.set_title('Cell Type Communication')
    
    plt.suptitle(f'CCC flow of {pathway_name}', y=1.05)
    
    if figpath is not None:
        pathway_name_new = pathway_name.replace('->', '_')
        filename_base = f'Combined_plot_{pathway_name_new}'
        
        plt.savefig(f"{figpath}{filename_base}.png", dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(f"{figpath}{filename_base}.pdf", bbox_inches=bbox_inches)
    
    plt.show()
    plt.close()
    
    
    
def run_plot_communication_double_panel(
    path_name,
    base_path,
    cell_clus,
    cell_loc,
    meta_df,
    color_map,
    Sig_path_path,
    ccc,
    figpath=None,
    strength_scale=1,
    point_size=30,
    figsize=(10, 4)
):
    """
    Run CCC multi-layer path analysis and plot communication.

    Parameters
    ----------
    path_name : str
        Multi-layer path (e.g., 'geneA->geneB->geneC->geneD')
    base_path : str
        Path to the base directory containing CCC data
    cell_clus : pd.DataFrame
        Cell type annotation
    cell_loc : pd.DataFrame
        Spatial coordinates(must contain 'x', 'y')
    meta_df : pd.DataFrame
        Visualization dataframe (x, y, color)
    color_map : dict
        Cell type color mapping
    Sig_path_path : str
        Path to Significant_paths.csv
    ccc : dict
        Precomputed communication strength dictionary
    figpath : str or None
        Output path (None = do not save)
    """

    nei_adj_path = base_path + 'CCC/Nei_adj.csv'
    if not os.path.exists(nei_adj_path):
        raise FileNotFoundError(f"[ERROR] Nei_adj file not found: {nei_adj_path}")
    print(f"Loading Nei_adj from: {nei_adj_path}")
    Nei_adj = pd.read_csv(base_path + 'CCC/Nei_adj.csv', sep='\t', index_col=None, header=None)

    # ===== Step 1: adjacency =====
    adj = get_sender_adj(Nei_adj, cell_clus)

    # ===== Step 2: load significant paths =====
    Sig_path = pd.read_csv(Sig_path_path)

    # ===== Step 3: compute vectors =====
    res_all_df = get_Sig_all_vectors_mlpath(
        path_name, Sig_path, cell_clus, cell_loc, adj
    )

    res_weighted_df = get_Sig_weighted_one_vector_mlpath(res_all_df)

    # ===== Step 4: construct strength df =====
    Strength_lst = ccc[path_name]

    Strength_df = cell_loc.copy()
    Strength_df['Comm_Score'] = Strength_lst

    # log scaling
    # Strength_df['Comm_Score'] = np.log10(Strength_df['Comm_Score'] * 1e9 + 1e-10)

    # ===== Step 5: plot =====
    plot_communication_double_panel(
        strength_df=Strength_df,
        coord_df=meta_df,
        res_weighted_df=res_weighted_df,
        color_map=color_map,
        pathway_name=path_name,
        figpath=figpath,
        strength_scale=strength_scale,
        point_size=point_size,
        figsize=figsize
    )






def find_signal_paths_with_relay(df):
    # dict：{receiver: [sender1, sender2, ...]}
    receiver_to_senders = defaultdict(list)
    
    for _, row in df.iterrows():
        receiver = row[0] 
        senders = row[1:].dropna().astype(int).tolist() 
        receiver_to_senders[receiver].extend(senders)
    
    # all paths (x, relay, y)
    paths = []
    for relay in receiver_to_senders:
        # 1.relay <-（x）
        senders_to_relay = receiver_to_senders[relay]
        
        # 2.relay ->（y）
        receivers_from_relay = []
        for potential_receiver, senders in receiver_to_senders.items():
            if relay in senders:
                receivers_from_relay.append(potential_receiver)
        
        # 3. (x, relay, y)
        for x in senders_to_relay:
            for y in receivers_from_relay:
                if x != relay and relay != y and x != y:  # Ensure no self-loops
                    paths.append({'x': x, 'relay': relay, 'y': y})
    
    result_df = pd.DataFrame(paths).drop_duplicates()
    
    return result_df



def create_sig_x_relay_y(x_relay_y, sig_path):
    significant_receivers = set(sig_path['Sample_Name'].unique())
    sig_path_dict = {}
    for _, row in sig_path.iterrows():
        sig_path_dict.setdefault(row['Sample_Name'], []).append(
            (row['Path_Symbol'], row['Comm_Score'])
        )
    
    results = []
    
    for _, row in tqdm(x_relay_y.iterrows(), total=x_relay_y.shape[0], desc='Processing paths'):
        x_name = row['x_name']
        relay_name = row['relay_name']
        y_name = row['y_name']
        
        if relay_name in significant_receivers and y_name in significant_receivers:
            relay_paths = sig_path_dict.get(relay_name, [])
            y_paths = sig_path_dict.get(y_name, [])
            
            for (xr_symbol, xr_score), (ry_symbol, ry_score) in product(relay_paths, y_paths):
                results.append({
                    'X_Name': x_name,
                    'Relay_Name': relay_name,
                    'Y_Name': y_name,
                    'XR_Path_Symbol': xr_symbol,
                    'XR_Comm_Score': xr_score,
                    'RY_Path_Symbol': ry_symbol,
                    'RY_Comm_Score': ry_score
                })
    
    return pd.DataFrame(results)



import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_comm_strength(
    comm_item,
    comm_score,
    cell_loc,
    cell_clus,
    scale_factor=1e9,
    log_transform=True,
    eps=1e-10,
    cmap_colors=("#D3D3D3", "#AD2DE9"),
    point_size=20,
    sort_by_expr=True
):
    """
    Plot spatial distribution of a ligand-receptor pair or ligand-receptor-tf-tg path.

    Parameters
    ----------
    comm_item : str
        pair name or path name, e.g., 'Tgfb1->Tgfbr1_Tgfbr2' or 'Pdgfa->Pdgfra->Atf4->Atp1b1'
    comm_score : pd.DataFrame
        DataFrame with communication scores (rows = paths, columns = cells)
    cell_loc : pd.DataFrame
        DataFrame with spatial coordinates (columns should include 'x', 'y')
    cell_clus : pd.DataFrame
        DataFrame with cell type annotations (column 'celltype')
    cmap_colors : tuple
        Color range for expression (low → high)
    point_size : int
        Size of scatter points
    sort_by_expr : bool
        Whether to sort by expression (for better visualization layering)
    """

    data = comm_score[comm_item]
    
    if log_transform:
        plot_data = np.log10(data * scale_factor + eps)
    else:
        plot_data = data.copy()

    coord_cell_type = cell_loc.copy()
    coord_cell_type['cell_type'] = cell_clus['cell_type']
    merged_data = coord_cell_type.join(plot_data.rename('expression'))

    if sort_by_expr:
        merged_data = merged_data.sort_values(by='expression')

    cmap_custom = LinearSegmentedColormap.from_list(
        "custom_cmap", cmap_colors
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        merged_data['x'],
        merged_data['y'],
        c=merged_data['expression'],
        cmap=cmap_custom,
        s=point_size
    )

    # plt.gca().invert_yaxis()
    plt.colorbar(sc, label='Strength level')
    plt.title(f'Communication strength: {comm_item}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_ccc_flow_for_signaling(
    pathway_name,
    Sig_LR_path,
    lr_with_pathway,
    cell_clus,
    cell_loc,
    coord_df,
    color_map,
    base_path=None,
    mode='Pathway_Name',
    alpha=0.6,
    point_size=30,
    arrow_scale=1e1,
    rad=0.2
):
    """
    Plot CCC-driven information flow for a given pathway.

    Parameters
    ----------
    pathway_name : str
        Pathway name (e.g., 'EGF')
    Nei_adj : pd.DataFrame
        Neighbor adjacency matrix
    Sig_LR_path : str
        Path to Significant_LRs.csv
    lr_with_pathway : pd.DataFrame
        Mapping between LR pairs and pathways
    cell_clus : pd.DataFrame
        Cell type annotations
    cell_loc : pd.DataFrame
        Cell spatial coordinates
    coord_df : pd.DataFrame
        DataFrame with x, y, and color info
    color_map : dict
        Cell type color mapping
    mode : str
        Focus mode for significant LR filtering ('Pathway_Name' or 'LR_Symbol')
    alpha : float
        Transparency of background cells
    point_size : int
        Scatter point size
    arrow_scale : float
        Scaling factor for arrow width
    rad : float
        Curvature of arrows
    """
    nei_adj_path = base_path + 'CCC/Nei_adj.csv'     
    if not os.path.exists(nei_adj_path):         
        raise FileNotFoundError(f"[ERROR] Nei_adj file not found: {nei_adj_path}")     
    print(f"Loading Nei_adj from: {nei_adj_path}")     
    Nei_adj = pd.read_csv(base_path + 'CCC/Nei_adj.csv', sep='\t', index_col=None, header=None)

    # ===== Step 1: adjacency =====
    adj = get_sender_adj(Nei_adj, cell_clus)

    # ===== Step 2: load significant LR =====
    Sig_LR = pd.read_csv(Sig_LR_path)

    # ===== Step 3: map pathway =====
    pathway_mapping = lr_with_pathway.set_index('LR_Symbol')['Pathway_Name'].to_dict()
    Sig_LR['Pathway_Name'] = Sig_LR['LR_Symbol'].map(pathway_mapping)

    # ===== Step 4: get vectors =====
    res_all_df = get_Sig_all_vectors(
        pathway_name, Sig_LR, cell_clus, cell_loc, adj, mode=mode
    )

    res_weighted_df = get_Sig_weighted_one_vector(res_all_df)

    # ===== Step 5: plot base scatter =====
    plt.figure(figsize=(8, 6))

    plt.scatter(
        coord_df['x'],
        coord_df['y'],
        c=coord_df['color'],
        alpha=alpha,
        s=point_size,
        linewidths=0
    )

    # plt.gca().invert_yaxis()

    # ===== Step 6: draw arrows =====
    for _, row in res_weighted_df.iterrows():
        strength = row['Inter_Score']
        lw = max(0.5, strength * arrow_scale)

        arrow = FancyArrowPatch(
            (row['Sender_x'], row['Sender_y']),
            (row['Receiver_x'], row['Receiver_y']),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=ArrowStyle(
                "Simple",
                head_length=1.8,
                head_width=2,
                tail_width=0.01
            ),
            color='black',
            lw=lw,
            mutation_scale=2
        )

        plt.gca().add_patch(arrow)

    # ===== Step 7: legend =====
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=celltype,
            markerfacecolor=color,
            markersize=10
        )
        for celltype, color in color_map.items()
    ]

    plt.legend(
        handles=handles,
        title='Cell types',
        bbox_to_anchor=(1.30, 1),
        loc='upper right'
    )

    # ===== Step 8: title =====
    plt.title(f'CCC flow of {pathway_name} signaling')

    plt.tight_layout()
    plt.show()

    return res_weighted_df


def plot_two_hop_signaling(
    pathway_name,
    pair1,
    pair2,
    base_path,
    lr_with_pathway,
    cell_clus,
    cell_loc,
    coord_df,
    color_map,
    arrow_scale=1e1,
    rad=0.2,
    alpha=0.6,
    point_size=30
):
    """
    Visualize potential two-hop (relay) signaling for a given pathway and LR pair combination.

    Parameters
    ----------
    pathway_name : str
        Pathway name (e.g., 'NCAM')
    pair1 : str
        First-hop LR pair (X -> Relay)
    pair2 : str
        Second-hop LR pair (Relay -> Y)
    base_path : str
        Base directory for input files
    cell_clus : pd.DataFrame
        Cell type annotations
    cell_loc : pd.DataFrame
        Cell spatial coordinates
    coord_df : pd.DataFrame
        DataFrame with x, y, and color
    color_map : dict
        Cell type color mapping
    arrow_scale : float
        Scaling factor for arrow width
    rad : float
        Arrow curvature
    alpha : float
        Background transparency
    point_size : int
        Scatter point size
    """

    # ===== Load data =====
    Nei_adj = pd.read_csv(base_path + 'CCC/Nei_adj.csv', sep='\t', header=None, index_col=None)
    Sig_LR = pd.read_csv(base_path + 'CCC/Significant_LRs.csv')
    
    lr_with_pathway['LR_Symbol'] = (
        lr_with_pathway['Ligand_Symbol'] + '->' + lr_with_pathway['Receptor_Symbol']
    )

    # ===== Step 1: relay structure =====
    x_relay_y = find_signal_paths_with_relay(Nei_adj)

    x_relay_y['x_name'] = cell_clus.index[x_relay_y['x']]
    x_relay_y['relay_name'] = cell_clus.index[x_relay_y['relay']]
    x_relay_y['y_name'] = cell_clus.index[x_relay_y['y']]

    # ===== Step 2: prepare significant LR =====
    Sig_path = Sig_LR.copy()
    Sig_path.rename(columns={
        'LR_Symbol': 'Path_Symbol',
        'Inter_Score': 'Comm_Score'
    }, inplace=True)

    # ===== Step 3: filter pathway =====
    sublr = lr_with_pathway[lr_with_pathway['Pathway_Name'] == pathway_name]
    lr_lst = sublr['LR_Symbol']

    sub_sig_path = Sig_path[Sig_path['Path_Symbol'].isin(lr_lst)]

    # ===== Step 4: build relay pairs =====
    sig_x_relay_y = create_sig_x_relay_y(x_relay_y, sub_sig_path)

    # ===== Step 5: filter specific pair combination =====
    cur_df = sig_x_relay_y[
        (sig_x_relay_y['XR_Path_Symbol'] == pair1) &
        (sig_x_relay_y['RY_Path_Symbol'] == pair2)
    ]

    # ===== Step 6: construct edges =====
    cur_xr = pd.DataFrame({
        'sender_x': cur_df['X_Name'].map(cell_loc['x']),
        'sender_y': cur_df['X_Name'].map(cell_loc['y']),
        'receiver_x': cur_df['Relay_Name'].map(cell_loc['x']),
        'receiver_y': cur_df['Relay_Name'].map(cell_loc['y']),
        'score': cur_df['XR_Comm_Score']
    })

    cur_ry = pd.DataFrame({
        'sender_x': cur_df['Relay_Name'].map(cell_loc['x']),
        'sender_y': cur_df['Relay_Name'].map(cell_loc['y']),
        'receiver_x': cur_df['Y_Name'].map(cell_loc['x']),
        'receiver_y': cur_df['Y_Name'].map(cell_loc['y']),
        'score': cur_df['RY_Comm_Score']
    })

    # ===== Step 7: plot =====
    plt.figure(figsize=(8, 6))

    # background cells
    plt.scatter(
        coord_df['x'],
        coord_df['y'],
        c=coord_df['color'],
        alpha=alpha,
        s=point_size,
        linewidths=0
    )

    # plt.gca().invert_yaxis()

    # first hop (X → Relay)
    for _, row in cur_xr.iterrows():
        lw = max(0.5, row['score'] * arrow_scale)

        arrow = FancyArrowPatch(
            (row['sender_x'], row['sender_y']),
            (row['receiver_x'], row['receiver_y']),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=ArrowStyle("Simple", head_length=1.8, head_width=2, tail_width=0.01),
            color='#13393E',
            lw=lw,
            mutation_scale=2
        )
        plt.gca().add_patch(arrow)

    # second hop (Relay → Y)
    for _, row in cur_ry.iterrows():
        lw = max(0.5, row['score'] * arrow_scale)

        arrow = FancyArrowPatch(
            (row['sender_x'], row['sender_y']),
            (row['receiver_x'], row['receiver_y']),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=ArrowStyle("Simple", head_length=1.8, head_width=2, tail_width=0.01),
            color='#B32142',
            lw=lw,
            mutation_scale=2
        )
        plt.gca().add_patch(arrow)

    # legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=celltype,
            markerfacecolor=color,
            markersize=10
        )
        for celltype, color in color_map.items()
    ]

    plt.legend(
        handles=handles,
        title='Cell Types',
        bbox_to_anchor=(1.25, 1),
        loc='upper right'
    )

    # title
    plt.title(f'Potential two-hop signaling: \n{pair1} → {pair2}')

    plt.tight_layout()
    plt.show()

    return cur_df



def plot_effective_two_hop_signaling(
    base_path,
    coord_df,
    color_map,
    pair1,
    pair2,
    can_trans_gene='NCAM1',
    strength_scale=1e1,
    arrow_rad=0.2,
    point_size=10,
    figsize=(8, 8)
):

    # ===== 1. Load data =====
    Sig_path = pd.read_csv(base_path + 'CCC/Significant_paths_res.csv')

    condition = Sig_path['path_symbol'].str.startswith(can_trans_gene)

    sig_x_relay_y = get_two_hop_cascade_vectors(
        Sig_path, condition, pair1
    )

    # ===== 2. Filter specific two-hop path =====
    cur_df = sig_x_relay_y[
        (sig_x_relay_y['XR_Path_Symbol'] == pair1) &
        (sig_x_relay_y['RY_Path_Symbol'] == pair2)
    ]

    # ===== 3. Construct edge data =====
    cur_xr = pd.DataFrame({
        'sender_x': cur_df['X_name'].map(coord_df['x']),
        'sender_y': cur_df['X_name'].map(coord_df['y']),
        'receiver_x': cur_df['Relay_name'].map(coord_df['x']),
        'receiver_y': cur_df['Relay_name'].map(coord_df['y']),
        'score': cur_df['XR_Comm_Score']
    })

    cur_ry = pd.DataFrame({
        'sender_x': cur_df['Relay_name'].map(coord_df['x']),
        'sender_y': cur_df['Relay_name'].map(coord_df['y']),
        'receiver_x': cur_df['Y_name'].map(coord_df['x']),
        'receiver_y': cur_df['Y_name'].map(coord_df['y']),
        'score': cur_df['RY_Comm_Score']
    })

    # ===== 4. Plot =====
    plt.figure(figsize=figsize)

    plt.scatter(
        coord_df['x'], coord_df['y'],
        c=coord_df['color'],
        alpha=0.6,
        s=point_size,
        linewidths=0
    )

    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()

    # ===== 5. Draw XR arrows =====
    for _, row in cur_xr.iterrows():
        lw = max(0.5, row['score'] * strength_scale)

        arrow = FancyArrowPatch(
            (row['sender_x'], row['sender_y']),
            (row['receiver_x'], row['receiver_y']),
            connectionstyle=f"arc3,rad={arrow_rad}",
            arrowstyle=ArrowStyle("Simple", head_length=1.8, head_width=2, tail_width=0.01),
            color='#13393E',
            lw=lw,
            mutation_scale=2
        )
        plt.gca().add_patch(arrow)

    # ===== 6. Draw RY arrows =====
    for _, row in cur_ry.iterrows():
        lw = max(0.5, row['score'] * strength_scale)

        arrow = FancyArrowPatch(
            (row['sender_x'], row['sender_y']),
            (row['receiver_x'], row['receiver_y']),
            connectionstyle=f"arc3,rad={arrow_rad}",
            arrowstyle=ArrowStyle("Simple", head_length=1.8, head_width=2, tail_width=0.01),
            color='#B32142',
            lw=lw,
            mutation_scale=2
        )
        plt.gca().add_patch(arrow)

    # ===== 7. Legend =====
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=celltype,
            markerfacecolor=color,
            markersize=10
        )
        for celltype, color in color_map.items()
    ]

    plt.legend(
        handles=handles,
        title='Cell Types',
        bbox_to_anchor=(1.3, 1),
        loc='upper right'
    )

    # ===== 8. Title =====
    plt.title(f'Effective two-hop signaling: \n{pair1} → {pair2}')

    plt.show()


def build_gene_relay_two_hop_events(sig_path, focus_gene):
    """
    Build two-hop signaling events centered on a focus gene.

    Logic:
    1. First hop:  X -> Relay, where path_symbol ends with focus_gene.
    2. Second hop: Relay -> Y, where path_symbol starts with focus_gene.
    3. Merge by Relay cell.

    Parameters
    ----------
    sig_path : pd.DataFrame
        Significant signaling path table.
        Required columns:
        from_cell, to_cell, source, target, path_symbol, comm_score, z_score

    focus_gene : str
        Focus gene used as relay signaling molecule, e.g. 'Hgf'.

    Returns
    -------
    merged_df : pd.DataFrame
        Two-hop signaling events joined by relay cell.
    """

    # First hop: upstream signal ending with focus_gene
    one_hop_sig_path = sig_path[
        sig_path["path_symbol"].str.endswith(focus_gene, na=False)
    ].copy()

    # Second hop: downstream signal starting with focus_gene
    two_hop_sig_path = sig_path[
        sig_path["path_symbol"].str.startswith(focus_gene, na=False)
    ].copy()

    one_hop_sig_path.rename(
        columns={
            "from_cell": "X_name",
            "to_cell": "Relay_name",
            "source": "X_Source",
            "target": "Relay_Target",
            "path_symbol": "X_Path_Symbol",
            "comm_score": "X_Comm_Score",
            "z_score": "X_Z_Score",
        },
        inplace=True
    )

    merged_df = pd.merge(
        left=one_hop_sig_path,
        right=two_hop_sig_path,
        left_on="Relay_name",
        right_on="from_cell",
        how="inner"
    )

    return merged_df


def plot_two_hop_cell_identity(
    focus_gene,
    base_path=None,
    coord_df=None,
    coord=None,
    cell_type=None,
    color_map=None,
    merged_df=None,
    figsize=(18, 5),
    bg_color="#D3D3D3",
    bg_size=15,
    bg_alpha=1,
    role_size=25,
    role_alpha=0.8
):
    """
    Plot source / relay / target cell identities for two-hop signaling events.

    Parameters
    ----------
    focus_gene : str
        Gene name, e.g. 'Vegfa'.
    base_path : str
        Dataset base path.
    coord_df : pd.DataFrame, optional
        Coordinate dataframe with x, y, color columns.
    coord : pd.DataFrame, optional
        Coordinate dataframe with x, y columns. Used when coord_df is not given.
    cell_type : pd.DataFrame, optional
        Cell type annotation with column 'cell_type'. Used when coord_df is not given.
    color_map : dict, optional
        Cell type color map. Used when coord_df is not given.
    merged_df : pd.DataFrame, optional
        Preloaded merged two-hop event dataframe.
    """

    # ===== Load merged two-hop event data =====
    if merged_df is None:
        if base_path is None:
            raise ValueError("Please provide either merged_df or base_path + focus_gene.")
        else:
            print("Loading merged two-hop event data...")
            merged_file = os.path.join(base_path,  f"{focus_gene}_merged_data.csv")
            merged_df = pd.read_csv(merged_file)

    df = merged_df.copy()

    # ===== Build coord_df if not provided =====
    if coord_df is None:
        if coord is None or cell_type is None or color_map is None:
            raise ValueError("Please provide either coord_df, or coord + cell_type + color_map.")

        coord_df = coord.copy()

        if "cell_type" not in cell_type.columns and "celltype" in cell_type.columns:
            cell_type = cell_type.rename(columns={"celltype": "cell_type"})

        coord_df["color"] = cell_type.loc[coord_df.index, "cell_type"].map(color_map)

    # ===== Compatible column names =====
    source_col = "X_name" if "X_name" in df.columns else "X_Name"
    relay_col = "Relay_name" if "Relay_name" in df.columns else "Relay_Name"

    if "to_cell" in df.columns:
        target_col = "to_cell"
    elif "Y_name" in df.columns:
        target_col = "Y_name"
    else:
        target_col = "Y_Name"

    # ===== Extract cells =====
    source_cells = set(df[source_col].dropna())
    relay_cells = set(df[relay_col].dropna())
    target_cells = set(df[target_col].dropna())

    # ===== Add role flags =====
    plot_df = coord_df.copy()
    plot_df["is_source"] = plot_df.index.isin(source_cells)
    plot_df["is_relay"] = plot_df.index.isin(relay_cells)
    plot_df["is_target"] = plot_df.index.isin(target_cells)

    print("source cells:", plot_df["is_source"].sum())
    print("relay cells:", plot_df["is_relay"].sum())
    print("target cells:", plot_df["is_target"].sum())

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    role_info = [
        ("Source", "is_source"),
        ("Relay", "is_relay"),
        ("Target", "is_target"),
    ]

    for ax, (title, flag) in zip(axes, role_info):
        ax.scatter(
            plot_df["x"], plot_df["y"],
            color=bg_color,
            s=bg_size,
            alpha=bg_alpha
        )

        sub = plot_df[plot_df[flag]]

        ax.scatter(
            sub["x"], sub["y"],
            c=sub["color"],
            s=role_size,
            alpha=role_alpha
        )

        # ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

    return plot_df, fig, axes


def plot_two_hop_cell_identity_combined(
    focus_gene,
    base_path=None,
    coord_df=None,
    coord=None,
    cell_type=None,
    color_map=None,
    merged_df=None,
    figsize=(6, 5),
    bg_color="#D3D3D3",
    bg_size=15,
    bg_alpha=1,
    marker_map=None,
    size_map=None,
    linewidth_map=None,
    role_color_map=None,
    role_alpha=0.9,
    invert_yaxis=False,
    save=False,
    fig_dir=None
):
    """
    Plot source / relay / target cell identities in one combined panel.

    Parameters
    ----------
    focus_gene : str
        Gene name, e.g. 'Vegfa'.
    base_path : str, optional
        Dataset base path. Used to load {focus_gene}_merged_data.csv if merged_df is not provided.
    coord_df : pd.DataFrame, optional
        Coordinate dataframe with x, y, and color columns.
    coord : pd.DataFrame, optional
        Coordinate dataframe with x, y columns. Used when coord_df is not given.
    cell_type : pd.DataFrame, optional
        Cell type annotation with column 'cell_type'. Used when coord_df is not given.
    color_map : dict, optional
        Cell type color map. Used when coord_df is not given.
    merged_df : pd.DataFrame, optional
        Preloaded merged two-hop event dataframe.
    save : bool
        Whether to save figure.
    fig_dir : str, optional
        Directory for saving figure if save=True.
    """

    # ===== Load merged two-hop event data =====
    if merged_df is None:
        if base_path is None:
            raise ValueError("Please provide either merged_df or base_path + focus_gene.")
        print("Loading merged two-hop event data...")
        merged_file = os.path.join(base_path, f"{focus_gene}_merged_data.csv")
        merged_df = pd.read_csv(merged_file)

    df = merged_df.copy()

    # ===== Build coord_df if not provided =====
    if coord_df is None:
        if coord is None or cell_type is None or color_map is None:
            raise ValueError("Please provide either coord_df, or coord + cell_type + color_map.")

        coord_df = coord.copy()

        if "cell_type" not in cell_type.columns and "celltype" in cell_type.columns:
            cell_type = cell_type.rename(columns={"celltype": "cell_type"})

        coord_df["color"] = cell_type.loc[coord_df.index, "cell_type"].map(color_map)

    # ===== Compatible column names =====
    source_col = "X_name" if "X_name" in df.columns else "X_Name"
    relay_col = "Relay_name" if "Relay_name" in df.columns else "Relay_Name"

    if "to_cell" in df.columns:
        target_col = "to_cell"
    elif "Y_name" in df.columns:
        target_col = "Y_name"
    else:
        target_col = "Y_Name"

    # ===== Extract cells =====
    source_cells = set(df[source_col].dropna())
    relay_cells = set(df[relay_col].dropna())
    target_cells = set(df[target_col].dropna())

    # ===== Add role flags =====
    plot_df = coord_df.copy()
    plot_df["is_source"] = plot_df.index.isin(source_cells)
    plot_df["is_relay"] = plot_df.index.isin(relay_cells)
    plot_df["is_target"] = plot_df.index.isin(target_cells)

    # ===== Default plotting style =====
    if marker_map is None:
        marker_map = {
            "source": "^",
            "relay": "+",
            "target": "o"
        }

    if size_map is None:
        size_map = {
            "source": 40,
            "relay": 70,
            "target": 40
        }

    if linewidth_map is None:
        linewidth_map = {
            "source": 0.8,
            "relay": 1.5,
            "target": 0.8
        }

    if role_color_map is None:
        role_color_map = {
            "source": "#ab8adf",
            "relay": "#23bdf5",
            "target": "#fa7d8a"
        }

    # ===== Plot =====
    fig, ax = plt.subplots(figsize=figsize)

    # background
    ax.scatter(
        plot_df["x"], plot_df["y"],
        color=bg_color,
        s=bg_size,
        alpha=bg_alpha
    )

    # overlay source / relay / target
    for role, flag in zip(
        ["source", "relay", "target"],
        ["is_source", "is_relay", "is_target"]
    ):
        sub = plot_df[plot_df[flag]]

        if role == "relay":
            ax.scatter(
                sub["x"], sub["y"],
                c=role_color_map[role],
                s=size_map[role],
                marker=marker_map[role],
                linewidths=linewidth_map[role],
                alpha=role_alpha,
                label=role
            )
        else:
            ax.scatter(
                sub["x"], sub["y"],
                facecolors="none",
                edgecolors=role_color_map[role],
                s=size_map[role],
                marker=marker_map[role],
                linewidths=linewidth_map[role],
                alpha=role_alpha,
                label=role
            )

    if invert_yaxis:
        ax.invert_yaxis()

    ax.legend(title="Role")
    ax.set_title(f"{focus_gene} - Cell Identity (Combined)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

    return plot_df, fig, ax


def plot_top_signal_ranked_abundance(
    focus_cell,
    cell_type,
    sig_pair_res=None,
    sig_lrp=None,
    top_n=20,
    figsize=(12, 6),
    palette="GnBu",
    save=False,
    figpath=None
):
    """
    Plot top abundant significant paths targeting a focus cell type.

    Parameters
    ----------
    focus_cell : str
        Focus cell type, e.g. 'Astro'.
    cell_type : pd.DataFrame
        Cell annotation dataframe with column 'cell_type'.
    sig_pair_res : pd.DataFrame
        Significant path/pair result dataframe.
    sig_lrp : pd.DataFrame, optional
        Alias for sig_pair_res. Kept for compatibility.
    top_n : int
        Number of top paths to show.
    figsize : tuple
        Figure size.
    palette : str
        Seaborn palette.
    save : bool
        Whether to save figure.
    figpath : str
        Save path without extension.
    """

    if sig_pair_res is None:
        if sig_lrp is None:
            raise ValueError("Please provide sig_pair_res or sig_lrp.")
        sig_pair_res = sig_lrp

    if "cell_type" not in cell_type.columns and "celltype" in cell_type.columns:
        cell_type = cell_type.rename(columns={"celltype": "cell_type"})

    focus_samples = cell_type[cell_type["cell_type"] == focus_cell].index.tolist()

    if len(focus_samples) == 0:
        raise ValueError(f"No cells found for focus_cell: {focus_cell}")

    if "to_cell" not in sig_pair_res.columns:
        raise ValueError("sig_pair_res must contain column 'to_cell'.")

    if "path_symbol" not in sig_pair_res.columns:
        raise ValueError("sig_pair_res must contain column 'path_symbol'.")

    focus_sig_lrp = sig_pair_res[
        sig_pair_res["to_cell"].isin(focus_samples)
    ].copy()

    path_counts = focus_sig_lrp["path_symbol"].value_counts()

    if len(path_counts) == 0:
        raise ValueError(f"No significant paths targeting {focus_cell} cells.")

    top_n = min(top_n, len(path_counts))
    plot_counts = path_counts.head(top_n)

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x=plot_counts.index,
        y=plot_counts.values,
        hue=plot_counts.values,
        palette=palette,
        legend=False
    )

    plt.title(
        f"Communication abundance of significant signals targeting {focus_cell}",
        fontsize=14
    )
    plt.xlabel("Communication signals", fontsize=12)
    plt.ylabel("Communication abundance (n)", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points"
        )

    plt.tight_layout()
    plt.show()

    return focus_sig_lrp, path_counts
