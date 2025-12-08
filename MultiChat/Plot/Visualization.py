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
    
    scatter1 = ax1.scatter(
        strength_df['x'], strength_df['y'], 
        c=strength_df['Comm_Score'],
        cmap='viridis', 
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
    
    ax1.set_title('Communication Strength Background')
    fig.colorbar(scatter1, ax=ax1, label='Intensity')
    
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
        title='Cell Types', 
        bbox_to_anchor=(1.35, 1), 
        loc='upper right'
    )
    ax2.set_title('Cell Type Communication')
    
    plt.suptitle(f'Communication Direction under: {pathway_name}', y=1.03)
    
    pathway_name_new = pathway_name.replace('->', '_')
    filename_base = f'Combined_plot_{pathway_name_new}'
    
    plt.savefig(f"{figpath}{filename_base}.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(f"{figpath}{filename_base}.pdf", bbox_inches=bbox_inches)
    
    plt.show()
    plt.close()
    
    
    



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