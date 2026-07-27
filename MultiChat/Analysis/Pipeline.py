from pathlib import Path

import pandas as pd
import torch



def _ensure_base_path(base_path):
    if base_path is None:
        raise ValueError("base_path is required.")
    base_path = str(base_path)
    if not base_path.endswith("/"):
        base_path += "/"
    return base_path


def _load_mc_adata_if_needed(mc_adata=None, adata_path=None):
    if mc_adata is not None:
        return mc_adata

    if adata_path is None:
        raise ValueError("Please provide mc_adata or adata_path.")

    import anndata as ad
    return ad.read_h5ad(adata_path)


def _safe_minmax_col(df):
    denom = df.max() - df.min()
    denom = denom.replace(0, 1)
    return (df - df.min()) / denom


def _safe_minmax_row(df):
    return df.apply(
        lambda x: (x - x.min()) / ((x.max() - x.min()) if x.max() != x.min() else 1),
        axis=1,
    )


def _align_atac_cell_peak(ATAC_data, sample_order):
    ATAC_data = ATAC_data.copy()
    ATAC_data.index = ATAC_data.index.astype(str)
    ATAC_data.columns = ATAC_data.columns.astype(str)

    sample_order = pd.Index(sample_order).astype(str)

    if len(sample_order.intersection(ATAC_data.index)) > 0:
        return ATAC_data.reindex(sample_order).fillna(0)

    if len(sample_order.intersection(ATAC_data.columns)) > 0:
        return ATAC_data.T.reindex(sample_order).fillna(0)

    raise ValueError("ATAC_data must contain mc_adata.obs_names in index or columns.")



def run_multichat_for_LRs(
    mc_adata=None,
    adata_path=None,
    base_path=None,
    background_inter_df=None,
    background_number=10,
    threads=5,
    gpu_id=0,
    celltype_key="cell_type",
    alpha=0.05, 
    min_cells_count=10,
    selected_cell_type=None,
    cleanup_bg=True,
):
    from . import Processing
    from . import Intra_strength as tl
    from ..Model import model_training, utilities

    base_path = _ensure_base_path(base_path)
    mc_adata = _load_mc_adata_if_needed(mc_adata, adata_path)

    mc_adata = model_training.run_contrastive_module(
        mc_adata=mc_adata,
        base_path=base_path,
        gpu_id=gpu_id,
        selected_cell_type=selected_cell_type,
    )

    parser = utilities.parameter_setting()
    args, unknown = parser.parse_known_args()
    args.gpu_id = gpu_id
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    if background_inter_df is None:
        background_inter_df = model_training.run_contrastive_module_for_bg(
            mc_adata=mc_adata,
            args=args,
            base_path=base_path,
            n_runs=background_number,
            n_processes=threads,
            seed=42,
            cleanup=cleanup_bg,
        )


    mc_adata = tl.get_significant_pairs(
        mc_adata=mc_adata,
        background_inter_df=background_inter_df,
        celltype_key=celltype_key,
        alpha=alpha,
        min_cells_count=min_cells_count,
    )

    return mc_adata


def run_multichat_default(
    mc_adata=None,
    adata_path=None,
    base_path=None,
    ATAC_data=None,
    tg_re_df=None,
    tf_re_ba=None,
    r_tf_cellcall=None,
    l_r_df=None,
    L_R_TF_TG_df=None,
    background_inter_df=None,
    background_number=10,
    threads=5,
    gpu_id=0,
    celltype_key="cell_type",
    alpha=0.05, 
    min_cells_count=10,
    ligand_wise=True,
    tg_wise=True,
):
    from . import Intra_strength as tl

    base_path = _ensure_base_path(base_path)

    mc_adata = run_multichat_for_LRs(
        mc_adata=mc_adata,
        adata_path=adata_path,
        base_path=base_path,
        background_inter_df=background_inter_df,
        background_number=background_number,
        threads=threads,
        gpu_id=gpu_id,
        celltype_key=celltype_key,
        alpha=alpha, 
        min_cells_count=min_cells_count,
    )

    print('================================================================')
    print('MultiChat for L-R-TF-TG path score calculation starts ...')
    print('================================================================')

    if ATAC_data is None:
        raise ValueError("ATAC_data is required for default mode.")

    if tg_re_df is None:
        tg_re_df = mc_adata.uns["peak-TG_links"]

    if tf_re_ba is None:
        tf_re_ba = mc_adata.uns["peak-TF_mtx"]

    if l_r_df is None:
        l_r_df = mc_adata.uns["CCC"]["L-R_db_filt1"]

    # if L_R_TF_TG_df is not None:

    if L_R_TF_TG_df is None:
        if "L-R-TF-TG_db" in mc_adata.uns.get("CCC", {}):
            L_R_TF_TG_df = mc_adata.uns["CCC"]["L-R-TF-TG_db"]
        else:
            if r_tf_cellcall is None:
                r_tf_cellcall = mc_adata.uns.get("L-R-TF_CellCall")
            if r_tf_cellcall is None:
                r_tf_cellcall = pd.read_csv(
                    base_path + "inputs/new_ligand_receptor_TFs_homology.txt",
                    sep="\t",
                )

            tf_tg_common_peaks = tl.build_tf_tg_mapping(tg_re_df, tf_re_ba)
            l_r_tf_df = tl.generate_l_r_tf_pairs(l_r_df, r_tf_cellcall)
            L_R_TF_TG_df = tl.generate_l_r_tf_tg_pairs(l_r_tf_df, tf_tg_common_peaks)

            mc_adata.uns["CCC"]["L-R-TF_db"] = l_r_tf_df

    mc_adata.uns["CCC"]["L-R-TF-TG_db"] = L_R_TF_TG_df

    rna_mat = mc_adata.uns["CCC"]["smooth_exp"]
    sample_order = rna_mat.columns.astype(str)

    cell_rep_aligned = mc_adata.uns["cell_rep"].loc[sample_order]
    gene_rep = mc_adata.uns["gene_rep"].copy()
    tf_rep = mc_adata.uns["TF_rep"].copy()
    peak_rep = mc_adata.uns["peak_rep"].copy()

    tf_rep.index = [str(x).replace("M_", "", 1) for x in tf_rep.index]

    atac_mat = _align_atac_cell_peak(ATAC_data, sample_order)

    rna_mat_minmax = _safe_minmax_col(rna_mat)
    atac_mat_minmax = _safe_minmax_row(atac_mat)

    r_tf_tg_results = tl.run_intra_strength_pipeline(
        base_path=base_path,
        rna_mat_minmax=rna_mat_minmax,
        atac_mat_minmax=atac_mat_minmax,
        tg_re_df=tg_re_df,
        tf_rep=tf_rep,
        peak_rep=peak_rep,
        tf_re_ba=tf_re_ba,
        gene_rep=gene_rep,
        cell_rep_aligned=cell_rep_aligned,
        L_R_TF_TG_df=L_R_TF_TG_df,
    )
    del r_tf_tg_results

    ccc_lrp_df = mc_adata.uns["CCC"]["LRI_module_strength"]

    if tg_wise:
        tl.calculate_path_strength_by_tg(
            l_r_tf_tg_df=L_R_TF_TG_df,
            combined_npz_path=base_path + "CCC/R_TF_TG/combined_results.npz",
            global_row_names_path=base_path + "CCC/R_TF_TG/combined_row_names.json",
            global_col_names_path=base_path + "CCC/R_TF_TG/combined_col_names.json",
            ccc_lrp_df=ccc_lrp_df,
            output_dir=base_path + "CCC/L_R_TF_TG/TG_cascade_results",
        )

    if ligand_wise:
        tl.calculate_path_strength(
            l_r_tf_tg_df=L_R_TF_TG_df,
            combined_npz_path=base_path + "CCC/R_TF_TG/combined_results.npz",
            global_row_names_path=base_path + "CCC/R_TF_TG/combined_row_names.json",
            global_col_names_path=base_path + "CCC/R_TF_TG/combined_col_names.json",
            ccc_lrp_df=ccc_lrp_df,
            output_dir=base_path + "CCC/L_R_TF_TG/ligand_cascade_results",
        )

    return mc_adata


def run_multichat_wto_chrom_acc(
    mc_adata=None,
    adata_path=None,
    base_path=None,
    L_R_TF_TG_df=None,
    background_inter_df=None,
    background_number=10,
    threads=5,
    gpu_id=0,
    celltype_key="cell_type",
    alpha=0.05, 
    min_cells_count=10,
    ligand_wise=True,
    tg_wise=True,
):
    from . import Intra_strength as tl

    base_path = _ensure_base_path(base_path)

    mc_adata = run_multichat_for_LRs(
        mc_adata=mc_adata,
        adata_path=adata_path,
        base_path=base_path,
        background_inter_df=background_inter_df,
        background_number=background_number,
        threads=threads,
        gpu_id=gpu_id,
        celltype_key=celltype_key,
        alpha=alpha,
        min_cells_count=min_cells_count
    )

    rna_mat = mc_adata.uns["CCC"]["smooth_exp"]

    if L_R_TF_TG_df is None:
        if "L-R-TF-TG_db" in mc_adata.uns["CCC"]:
            L_R_TF_TG_df = mc_adata.uns["CCC"]["L-R-TF-TG_db"]
        else:
            L_R_TF_TG_df = pd.read_csv(
                base_path + "inputs/Ligand_Receptor_TF_TG_pairs.csv",
                sep="\t",
            )

    cell_rep = rna_mat.T.copy()
    gene_rep = cell_rep.T.copy()

    tg_list = list(set(L_R_TF_TG_df["TG_Symbol"].astype(str)))
    gene_rep = gene_rep.loc[gene_rep.index.intersection(tg_list)]

    tf_rep = cell_rep.T.copy()
    tf_list = list(set(L_R_TF_TG_df["TF_Symbol"].astype(str)))
    tf_rep = tf_rep.loc[tf_rep.index.intersection(tf_list)]

    rna_mat_minmax = _safe_minmax_col(rna_mat)

    r_tf_tg_results = tl.run_intra_strength_ablation_pipeline(
        base_path=base_path,
        rna_mat_minmax=rna_mat_minmax,
        gene_rep=gene_rep,
        tf_rep=tf_rep,
        cell_rep=cell_rep,
        L_R_TF_TG_df=L_R_TF_TG_df,
    )
    del r_tf_tg_results

    mc_adata.uns["CCC"]["L-R-TF-TG_db"] = L_R_TF_TG_df

    ccc_lrp_df = mc_adata.uns["CCC"]["LRI_module_strength"]

    if tg_wise:
        tl.calculate_path_strength_by_tg(
            l_r_tf_tg_df=L_R_TF_TG_df,
            combined_npz_path=base_path + "CCC/R_TF_TG/combined_results.npz",
            global_row_names_path=base_path + "CCC/R_TF_TG/combined_row_names.json",
            global_col_names_path=base_path + "CCC/R_TF_TG/combined_col_names.json",
            ccc_lrp_df=ccc_lrp_df,
            output_dir=base_path + "CCC/L_R_TF_TG/TG_cascade_results",
        )

    if ligand_wise:
        tl.calculate_path_strength(
            l_r_tf_tg_df=L_R_TF_TG_df,
            combined_npz_path=base_path + "CCC/R_TF_TG/combined_results.npz",
            global_row_names_path=base_path + "CCC/R_TF_TG/combined_row_names.json",
            global_col_names_path=base_path + "CCC/R_TF_TG/combined_col_names.json",
            ccc_lrp_df=ccc_lrp_df,
            output_dir=base_path + "CCC/L_R_TF_TG/ligand_cascade_results",
        )

    return mc_adata


def run_multichat(
    mc_adata=None,
    adata_path=None,
    base_path=None,
    ATAC_data=None,
    mode="default",
    if_multi_layer=True,
    if_atac=True,
    background_number=10,
    threads=5,
    gpu_id=0,
    celltype_key="cell_type",
    alpha=0.05,
    min_cells_count=10,
    ligand_wise=True,
    tg_wise=True,
    **kwargs,
):
    """
    Main MultiChat pipeline.

    mode:
    - "default": full pipeline
    - "wto_chrom_acc": without chromatin accessibility
    - "LRs": LR module only
    """

    if not if_multi_layer:
        mode = "LRs"

    if mode == "LRs":
        return run_multichat_for_LRs(
            mc_adata=mc_adata,
            adata_path=adata_path,
            base_path=base_path,
            background_number=background_number,
            threads=threads,
            gpu_id=gpu_id,
            celltype_key=celltype_key,
            alpha=alpha,
            min_cells_count=min_cells_count,
            **kwargs,
        )

    if mode == "default" and if_atac:
        return run_multichat_default(
            mc_adata=mc_adata,
            adata_path=adata_path,
            base_path=base_path,
            ATAC_data=ATAC_data,
            background_number=background_number,
            threads=threads,
            gpu_id=gpu_id,
            celltype_key=celltype_key,
            alpha=alpha,
            min_cells_count=min_cells_count,
            ligand_wise=ligand_wise,
            tg_wise=tg_wise,
            **kwargs,
        )

    if mode == "default" and not if_atac:
        mode = "wto_chrom_acc"

    if mode == "wto_chrom_acc":
        return run_multichat_wto_chrom_acc(
            mc_adata=mc_adata,
            adata_path=adata_path,
            base_path=base_path,
            background_number=background_number,
            threads=threads,
            gpu_id=gpu_id,
            celltype_key=celltype_key,
            alpha=alpha,
            min_cells_count=min_cells_count,
            ligand_wise=ligand_wise,
            tg_wise=tg_wise,
            **kwargs,
        )

    raise ValueError("mode must be one of: 'default', 'wto_chrom_acc', 'LRs'.")