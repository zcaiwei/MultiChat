rm(list=ls())

suppressPackageStartupMessages({
  library(chromVAR)
  library(motifmatchr)
  library(SummarizedExperiment)
  library(Matrix)
  library(JASPAR2020)
  library(GenomicRanges)
  library(tidyr)
  library(dplyr)
  library(TFBSTools)
  library(BSgenome.Mmusculus.UCSC.mm10)
})

# ----------------------- Command-line parameters----------------------- #
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_motif_matrix.R <input_count_csv> <output_dir>")
}
input_file <- args[1]
output_dir <- args[2]

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ----------------------- step1: read peak*cell count matrix ----------------------- #
counts <- t(read.csv(input_file, header = TRUE, row.names = 1))
peak_list <- rownames(counts)

# 转换为 GRanges
peaks_gr <- GRanges(
  seqnames = sub("\\..*", "", peak_list),
  ranges = IRanges(
    start = as.numeric(sub(".*\\.(\\d+)\\..*", "\\1", peak_list)),
    end   = as.numeric(sub(".*\\.(\\d+)\\.(\\d+)", "\\2", peak_list))
  )
)

# ----------------------- step2: build RangedSummarizedExperiment ----------------------- #
genome <- BSgenome.Mmusculus.UCSC.mm10
se <- SummarizedExperiment(assays = list(counts = as.matrix(counts)), rowRanges = peaks_gr)
se_gc <- addGCBias(se, genome = genome)

# ----------------------- step3: motif databases ----------------------- #
source("/Applications/WHUer/notebooks/MultiChat-main/MultiChat/Data_preprocessing/get_motif_list.R")  

species <- "Mus musculus"
collection <- "CORE"
version <- 1

jaspar_2020 <- getMotifList("jaspar2020", species, collection, version)
jaspar_2016 <- getMotifList("jaspar2016", species, collection, version)
jaspar_2018 <- getMotifList("jaspar2018", species, collection, version)
jaspar_2024 <- getMotifList("jaspar2024", species, collection, version)
# vierstra_motifs <- getMotifList("vierstra", species, "individual", version)
encode_motifs  <- getMotifList("encode", species, collection, version)
homer_motifs   <- getMotifList("homer", species, collection, version)
cisbp_motifs   <- getMotifList("cisbp", species, collection, version)

# ----------------------- step4: motif matching ----------------------- #
motif_sets <- list(
  jaspar2016 = jaspar_2016,
  jaspar2018 = jaspar_2018,
  jaspar2020 = jaspar_2020,
  jaspar2024 = jaspar_2024,
  # vierstra   = vierstra_motifs,
  encode     = encode_motifs,
  homer      = homer_motifs,
  cisbp      = cisbp_motifs
)

for (db_name in names(motif_sets)) {
  cat("Processing:", db_name, "\n")
  motif_set <- motif_sets[[db_name]]
  
  ix_scores <- motifmatchr::matchMotifs(motif_set, se_gc, genome = genome, out = "scores")
  freq_motif <- motifCounts(ix_scores)
  motif_names <- sapply(names(motif_set), function(x) motif_set[[x]]@name)
  colnames(freq_motif) <- motif_names
  
  outfile <- file.path(output_dir, paste0(db_name, "_peak_motif_matrix.txt"))
  write.table(as.matrix(freq_motif), file = outfile, sep = "\t", quote = FALSE)
}
