from pathlib import Path
import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from anndata import AnnData
import subprocess
from scipy import sparse

from .Processing import select_peaks_by_genes_location




def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _embedding_to_df(adata_emb, prefix="emb"):
    cols = [f"{prefix}_{i}" for i in range(adata_emb.X.shape[1])]
    return pd.DataFrame(
        _to_dense(adata_emb.X),
        index=adata_emb.obs_names.astype(str),
        columns=cols,
    )


def _standardize_peak_name(x):
    x = str(x)
    if ":" in x and "-" in x:
        return x

    parts = x.split(".")
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        return f"{parts[0]}:{parts[1]}-{parts[2]}"

    return x


def _extract_genes_from_db(*dfs):
    genes = set()

    for df in dfs:
        if df is None:
            continue

        for col in df.columns:
            for value in df[col].dropna().astype(str):
                for gene in value.split("_"):
                    if gene:
                        genes.add(gene)

    return genes


def _make_atac_adata(ATAC_data, cell_names, celltype=None):
    ATAC_data = ATAC_data.copy()
    ATAC_data.index = ATAC_data.index.astype(str)
    ATAC_data.columns = ATAC_data.columns.astype(str)

    cell_names = pd.Index(cell_names.astype(str))

    # Expected orientation: cell x peak
    if len(cell_names.intersection(ATAC_data.index)) > 0:
        atac_cell_peak = ATAC_data.reindex(cell_names).fillna(0)

    # Alternative orientation: peak x cell
    elif len(cell_names.intersection(ATAC_data.columns)) > 0:
        atac_cell_peak = ATAC_data.T.reindex(cell_names).fillna(0)

    else:
        raise ValueError("ATAC_data must contain mc_adata.obs_names in either index or columns.")

    adata_atac = AnnData(atac_cell_peak)
    adata_atac.obs_names = atac_cell_peak.index.astype(str)
    adata_atac.var_names = atac_cell_peak.columns.astype(str)

    if celltype is not None:
        adata_atac.obs["cell_type"] = list(celltype)

    return adata_atac


def _read_motif_matrix(path):
    df = pd.read_csv(path, sep="\t")

    first_col = df.columns[0]
    first_values = df[first_col].astype(str)

    if first_values.str.contains(r"[:.]").any():
        df = df.set_index(first_col)

    df.index = [_standardize_peak_name(x) for x in df.index]
    return df


def _build_peak_tf_matrix(workdir, tf_list):
    workdir = Path(workdir)

    jaspar2016 = _read_motif_matrix(workdir / "jaspar2016_peak_motif_matrix.txt")
    jaspar2018 = _read_motif_matrix(workdir / "jaspar2018_peak_motif_matrix.txt")
    jaspar2020 = _read_motif_matrix(workdir / "jaspar2020_peak_motif_matrix.txt")
    jaspar2024 = _read_motif_matrix(workdir / "jaspar2024_peak_motif_matrix.txt")
    encode = _read_motif_matrix(workdir / "encode_peak_motif_matrix.txt")
    homer = _read_motif_matrix(workdir / "homer_peak_motif_matrix.txt")
    cisbp = _read_motif_matrix(workdir / "cisbp_peak_motif_matrix.txt")

    cisbp.columns = cisbp.columns.astype(str).str.replace(".", "_", regex=False)
    homer.columns = homer.columns.astype(str).str.replace(r"\(.*\)", "", regex=True)
    encode.columns = [str(col).capitalize() for col in encode.columns]

    motif_dfs = [jaspar2016, jaspar2018, jaspar2020, jaspar2024, cisbp, encode, homer]

    tf_set = set(pd.Series(tf_list).dropna().astype(str))
    common_tfs = set()

    for df in motif_dfs:
        common_tfs |= tf_set.intersection(set(df.columns.astype(str)))

    averaged_data = {}

    for tf in common_tfs:
        tf_values = []

        for df in motif_dfs:
            if tf in df.columns:
                tf_values.append(df[tf])

        if tf_values:
            averaged_data[tf] = pd.concat(tf_values, axis=1).mean(axis=1, skipna=True)

    peak_tf_mtx = pd.DataFrame(averaged_data)
    peak_tf_mtx.index = [_standardize_peak_name(x) for x in peak_tf_mtx.index]

    return peak_tf_mtx


def get_joint_embedding(
    mc_adata,
    ATAC_data,
    gene_tss,
    workdir,
    motif_r_script,
    lr_df=None,
    marker_genes=None,
    celltype_key="cell_type",
    gene_tss_chr_col="chr",
    gene_tss_start_col="start",
    gene_tss_end_col="end",
    gene_tss_gene_col="Gene",
    min_gene_cells=3,
    min_peak_cells=3,
    n_bins=5,
    gene_peak_scope=250000,
    graph_dirname="graph0",
    pbg_workers=12,
    pbg_params=None,
    run_motif=True,
    overwrite_motif=False,
    peak_tf_mtx_file=None,
    genome="mm10"
):
    """
    Generate HGE joint embedding and store the outputs in mc_adata.uns.
    """

    import MultiChat.Heterogeneous_g_emb as hge

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    motif_r_script = Path(motif_r_script)

    lr_df = lr_df if lr_df is not None else mc_adata.uns.get("L-R_db_used")
    r_tf_cellcall = mc_adata.uns.get("L-R-TF_CellCall")
    marker_genes = marker_genes if marker_genes is not None else mc_adata.uns.get("marker_genes")

    if lr_df is None:
        raise ValueError("lr_df is required or mc_adata.uns['L-R_db_used'] must exist.")

    if r_tf_cellcall is not None:
        tf_list = r_tf_cellcall["TF_Symbol"].dropna().astype(str).tolist()
    else:
        raise ValueError(
            "TF information is required to build the peak-TF matrix."
            " Please ensure mc_adata.uns['L-R-TF_CellCall'] exists or provide r_tf_cellcall when constructing mc_adata."
        )
    
    hge.settings.set_workdir(str(workdir))
    hge.settings.set_figure_params(
        dpi=80,
        style="white",
        fig_size=[5, 5],
        rc={"image.cmap": "viridis"},
    )

    celltype = None
    if celltype_key in mc_adata.obs.columns:
        celltype = mc_adata.obs[celltype_key].astype(str).values

    # RNA AnnData: cell x gene
    adata_CG = mc_adata.copy()
    if celltype is not None:
        adata_CG.obs["cell_type"] = celltype

    # ATAC AnnData: cell x peak
    adata_CP = _make_atac_adata(
        ATAC_data=ATAC_data,
        cell_names=mc_adata.obs_names,
        celltype=celltype,
    )

    # Store peak names for the following pipeline
    mc_adata.uns['peak_lst'] = adata_CP.var_names.astype(str).tolist()

    # RNA processing
    hge.pp.filter_genes(adata_CG, min_n_cells=min_gene_cells)
    hge.pp.select_variable_genes(adata_CG, layer=None)
    hge.tl.discretize(adata_CG, n_bins=n_bins)

    db_genes = _extract_genes_from_db(lr_df)

    if marker_genes is not None and "gene" in marker_genes.columns:
        db_genes |= set(marker_genes["gene"].dropna().astype(str))

    hvg_genes = set(adata_CG.var_names[adata_CG.var["highly_variable"]])
    selected_genes = list(hvg_genes | db_genes)

    adata_CG_fltr = adata_CG[:, adata_CG.var_names.isin(selected_genes)].copy()

    # ATAC peak filtering
    hge.pp.filter_peaks(adata_CP, min_n_cells=min_peak_cells)

    gene_info = gene_tss.copy()
    gene_info = gene_info.rename(
        columns={
            gene_tss_start_col: "starts",
            gene_tss_end_col: "ends",
            gene_tss_gene_col: "genes",
        }
    )

    if gene_tss_chr_col != "chr" and gene_tss_chr_col in gene_info.columns:
        gene_info = gene_info.rename(columns={gene_tss_chr_col: "chr"})

    hvg_genes = list(adata_CG_fltr.var_names)
    peaks_to_filter = list(adata_CP.var_names)

    filtered_peaks, genes_peaks = select_peaks_by_genes_location(
        gene_info=gene_info,
        hvg_genes=hvg_genes,
        peaks_to_filter=peaks_to_filter,
        scope=gene_peak_scope,
    )

    hge.pp.binarize(adata_CP)
    adata_CP_fltr = adata_CP[:, filtered_peaks].copy()

    if peak_tf_mtx_file is not None:
        peak_tf_mtx_file = Path(peak_tf_mtx_file).expanduser().resolve()

        if not peak_tf_mtx_file.exists():
            raise FileNotFoundError(f"peak_tf_mtx_file does not exist: {peak_tf_mtx_file}")

        peak_tf_mtx = pd.read_csv(peak_tf_mtx_file, sep=",", index_col=0)
        peak_tf_mtx.index = [_standardize_peak_name(x) for x in peak_tf_mtx.index]

    else:
        # Motif matrix generation
        cell_peak_count_path = workdir / "CellPeak_fltr_count.csv"

        peak_count_df = pd.DataFrame(
            _to_dense(adata_CP_fltr.X),
            index=adata_CP_fltr.obs_names,
            columns=adata_CP_fltr.var_names,
        )
        peak_count_df.to_csv(cell_peak_count_path)

        motif_files = [
            workdir / "jaspar2016_peak_motif_matrix.txt",
            workdir / "jaspar2018_peak_motif_matrix.txt",
            workdir / "jaspar2020_peak_motif_matrix.txt",
            workdir / "jaspar2024_peak_motif_matrix.txt",
            workdir / "encode_peak_motif_matrix.txt",
            workdir / "homer_peak_motif_matrix.txt",
            workdir / "cisbp_peak_motif_matrix.txt",
        ]

        motif_ready = all(path.exists() for path in motif_files)

        if run_motif and (overwrite_motif or not motif_ready):
            subprocess.run(
                [
                    "Rscript",
                    str(motif_r_script),
                    str(cell_peak_count_path),
                    str(workdir),
                    str(genome),
                ],
                check=True,
            )

        peak_tf_mtx = _build_peak_tf_matrix(
            workdir=workdir,
            tf_list=tf_list,
        )

    peak_tf_mtx = peak_tf_mtx.loc[
        peak_tf_mtx.index.intersection(filtered_peaks)
    ].copy()

    adata_PM = AnnData(peak_tf_mtx)
    adata_PM.obs_names = peak_tf_mtx.index.astype(str)
    adata_PM.var_names = peak_tf_mtx.columns.astype(str)

    hge.pp.binarize(adata_PM)

    filtered_peaks = list(peak_tf_mtx.index)
    adata_CP_fltr = adata_CP[:, filtered_peaks].copy()

    print('================================================================')
    print('Heterogenous graph contrastive learning start training')
    print('================================================================')

    # Generate HGE graph and train
    hge.tl.gen_graph(
        list_CP=[adata_CP_fltr],
        list_CG=[adata_CG_fltr],
        list_PM=[adata_PM],
        copy=False,
        use_highly_variable=False,
        use_top_pcs=False,
        dirname=graph_dirname,
    )

    dict_config = hge.settings.pbg_params.copy()
    dict_config["workers"] = pbg_workers

    if pbg_params is not None:
        dict_config.update(pbg_params)

    hge.tl.pbg_train(
        pbg_params=dict_config,
        auto_wd=True,
        save_wd=True,
        output="model",
    )

    hge.settings.pbg_params = dict_config

    dict_adata = hge.read_embedding()

    adata_C = dict_adata["C"]
    adata_G = dict_adata["G"]
    adata_P = dict_adata["P"]
    adata_M = dict_adata["M"]

    adata_M.obs.index = "M_" + adata_M.obs.index.astype(str)

    adata_C.obs["entity_anno"] = "Cell"
    adata_G.obs["entity_anno"] = "Gene"
    adata_P.obs["entity_anno"] = "Peak"
    adata_M.obs["entity_anno"] = "Motif"

    if "cell_type" in adata_CG.obs.columns:
        adata_C.obs["cell_type"] = adata_CG[adata_C.obs_names, :].obs["cell_type"].copy()

    adata_all = hge.tl.embed(
        adata_ref=adata_C,
        list_adata_query=[adata_G, adata_P, adata_M],
    )

    adata_all.obs["entity_anno"] = ""
    adata_all.obs.loc[adata_C.obs_names, "entity_anno"] = "Cell"
    adata_all.obs.loc[adata_G.obs_names, "entity_anno"] = "Gene"
    adata_all.obs.loc[adata_P.obs_names, "entity_anno"] = "Peak"
    adata_all.obs.loc[adata_M.obs_names, "entity_anno"] = "Motif"

    # Store requested outputs
    mc_adata.uns["peak-TF_mtx"] = peak_tf_mtx
    cell_rep = _embedding_to_df(adata_C, prefix="cell_emb")
    cell_rep.index = cell_rep.index.astype(str) 
    cell_order = pd.Index(mc_adata.obs_names.astype(str))
    cell_rep = cell_rep.loc[cell_order]
    mc_adata.uns["cell_rep"] = cell_rep
    mc_adata.uns["gene_rep"] = _embedding_to_df(adata_G, prefix="gene_emb")
    mc_adata.uns["peak_rep"] = _embedding_to_df(adata_P, prefix="peak_emb")
    mc_adata.uns["TF_rep"] = _embedding_to_df(adata_M, prefix="tf_emb")

    # Store useful intermediate outputs
    mc_adata.uns["joint_embedding"] = {
        "cell_rep": adata_C,
        "gene_rep": adata_G,
        "peak_rep": adata_P,
        "TF_rep": adata_M,
        "filtered_peaks": filtered_peaks,
        "HGEworkdir": str(workdir),
    }

    return mc_adata




def get_joint_embedding_for_simulation(
    mc_adata,
    ATAC_data,
    peak_tf_mtx_file,
    workdir,
    l_r_tf_tg_df=None,
    celltype_key="cell_type",
    min_gene_cells=3,
    min_peak_cells=3,
    n_top_genes=20,
    n_bins=5,
    graph_dirname="graph0",
    pbg_workers=12,
    pbg_params=None,
):
    """
    Generate HGE joint embedding for simulation data.

    This simplified version is designed for simulated datasets where:
    - peak-TF matrix is already provided;
    - gene TSS annotation is not required;
    - motif matching does not need to be rerun;
    - L-R-TF-TG database is predefined.

    Stores:
    - mc_adata.uns["peak-TF_mtx"]
    - mc_adata.uns["cell_rep"]
    - mc_adata.uns["gene_rep"]
    - mc_adata.uns["peak_rep"]
    - mc_adata.uns["TF_rep"]
    - mc_adata.uns["joint_embedding"]
    """

    import MultiChat.Heterogeneous_g_emb as hge
    from scipy import sparse

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    peak_tf_mtx_file = Path(peak_tf_mtx_file).expanduser().resolve()

    if not peak_tf_mtx_file.exists():
        raise FileNotFoundError(f"peak_tf_mtx_file does not exist: {peak_tf_mtx_file}")

    if l_r_tf_tg_df is None:
        l_r_tf_tg_df = mc_adata.uns.get("CCC", {}).get("L-R-TF-TG_db")

    if l_r_tf_tg_df is None:
        l_r_tf_tg_df = mc_adata.uns.get("L-R-TF-TG_db")

    if l_r_tf_tg_df is None:
        raise ValueError(
            "l_r_tf_tg_df is required or must exist in "
            "mc_adata.uns['CCC']['L-R-TF-TG_db'] / mc_adata.uns['L-R-TF-TG_db']."
        )

    hge.settings.set_workdir(str(workdir))
    hge.settings.set_figure_params(
        dpi=80,
        style="white",
        fig_size=[5, 5],
        rc={"image.cmap": "viridis"},
    )

    celltype = None
    if celltype_key in mc_adata.obs.columns:
        celltype = mc_adata.obs[celltype_key].astype(str).values

    # RNA AnnData: cell x gene
    adata_CG = mc_adata.copy()
    if not sparse.issparse(adata_CG.X):
        adata_CG.X = sparse.csr_matrix(adata_CG.X)
    if celltype is not None:
        adata_CG.obs["cell_type"] = celltype

    # ATAC AnnData: cell x peak
    adata_CP = _make_atac_adata(
        ATAC_data=ATAC_data,
        cell_names=mc_adata.obs_names,
        celltype=celltype,
    )
    if not sparse.issparse(adata_CP.X):
        adata_CP.X = sparse.csr_matrix(adata_CP.X)

    mc_adata.uns["peak_lst"] = adata_CP.var_names.astype(str).tolist()

    # RNA preprocessing
    # hge.pp.filter_genes(adata_CG, min_n_cells=min_gene_cells)
    hge.pp.select_variable_genes(
        adata_CG,
        layer=None,
        n_top_genes=n_top_genes,
    )
    hge.tl.discretize(adata_CG, n_bins=n_bins)

    # Select genes used in the simulation database plus HVGs
    db_genes = _extract_genes_from_db(l_r_tf_tg_df)
    hvg_genes = set(adata_CG.var_names[adata_CG.var["highly_variable"]])
    selected_genes = list(hvg_genes | db_genes)

    adata_CG_fltr = adata_CG[:, adata_CG.var_names.isin(selected_genes)].copy()

    # ATAC preprocessing
    hge.pp.filter_peaks(adata_CP, min_n_cells=min_peak_cells)

    # Read provided peak-TF matrix
    peak_tf_mtx = pd.read_csv(peak_tf_mtx_file, sep="\t", index_col=0)
    peak_tf_mtx.index = [_standardize_peak_name(x) for x in peak_tf_mtx.index]

    # Keep only peaks available after ATAC filtering
    filtered_peaks = list(
        peak_tf_mtx.index.intersection(adata_CP.var_names.astype(str))
    )

    if len(filtered_peaks) == 0:
        raise ValueError(
            "No common peaks between ATAC_data and peak_tf_mtx_file. "
            "Please check peak naming format."
        )

    peak_tf_mtx = peak_tf_mtx.loc[filtered_peaks].copy()

    adata_CP_fltr = adata_CP[:, filtered_peaks].copy()

    adata_PM = AnnData(peak_tf_mtx)
    adata_PM.obs_names = peak_tf_mtx.index.astype(str)
    adata_PM.var_names = peak_tf_mtx.columns.astype(str)
    if not sparse.issparse(adata_PM.X):
        adata_PM.X = sparse.csr_matrix(adata_PM.X)

    hge.pp.binarize(adata_CP_fltr)
    hge.pp.binarize(adata_PM)

    print("================================================================")
    print("Heterogenous graph contrastive learning start training")
    print("================================================================")

    hge.tl.gen_graph(
        list_CP=[adata_CP_fltr],
        list_CG=[adata_CG_fltr],
        list_PM=[adata_PM],
        copy=False,
        use_highly_variable=False,
        use_top_pcs=False,
        dirname=graph_dirname,
    )

    dict_config = hge.settings.pbg_params.copy()
    dict_config["workers"] = pbg_workers

    if pbg_params is not None:
        dict_config.update(pbg_params)

    hge.tl.pbg_train(
        pbg_params=dict_config,
        auto_wd=True,
        save_wd=True,
        output="model",
    )

    hge.settings.pbg_params = dict_config

    dict_adata = hge.read_embedding()

    adata_C = dict_adata["C"]
    adata_G = dict_adata["G"]
    adata_P = dict_adata["P"]
    adata_M = dict_adata["M"]

    adata_M.obs.index = "M_" + adata_M.obs.index.astype(str)

    adata_C.obs["entity_anno"] = "Cell"
    adata_G.obs["entity_anno"] = "Gene"
    adata_P.obs["entity_anno"] = "Peak"
    adata_M.obs["entity_anno"] = "Motif"

    if "cell_type" in adata_CG.obs.columns:
        adata_C.obs["cell_type"] = adata_CG[adata_C.obs_names, :].obs["cell_type"].copy()

    adata_all = hge.tl.embed(
        adata_ref=adata_C,
        list_adata_query=[adata_G, adata_P, adata_M],
    )

    adata_all.obs["entity_anno"] = ""
    adata_all.obs.loc[adata_C.obs_names, "entity_anno"] = "Cell"
    adata_all.obs.loc[adata_G.obs_names, "entity_anno"] = "Gene"
    adata_all.obs.loc[adata_P.obs_names, "entity_anno"] = "Peak"
    adata_all.obs.loc[adata_M.obs_names, "entity_anno"] = "Motif"

    # Store outputs in mc_adata
    mc_adata.uns["peak-TF_mtx"] = peak_tf_mtx
    cell_rep = _embedding_to_df(adata_C, prefix="cell_emb")
    cell_rep.index = cell_rep.index.astype(str)
    cell_order = pd.Index(mc_adata.obs_names.astype(str))
    cell_rep = cell_rep.loc[cell_order]
    mc_adata.uns["cell_rep"] = cell_rep
    mc_adata.uns["gene_rep"] = _embedding_to_df(adata_G, prefix="gene_emb")
    mc_adata.uns["peak_rep"] = _embedding_to_df(adata_P, prefix="peak_emb")
    mc_adata.uns["TF_rep"] = _embedding_to_df(adata_M, prefix="tf_emb")

    mc_adata.uns["joint_embedding"] = {
        "cell_rep": adata_C,
        "gene_rep": adata_G,
        "peak_rep": adata_P,
        "TF_rep": adata_M,
        "filtered_peaks": filtered_peaks,
        "HGEworkdir": str(workdir)
    }

    return mc_adata