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
  # library(BSgenome.Mmusculus.UCSC.mm10)
})

# ----------------------- Command-line parameters----------------------- #
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_motif_matrix.R <input_count_csv> <output_dir> <genome>")
}
input_file <- args[1]
output_dir <- args[2]
genome_name <- tolower(args[3])

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

genome_config <- switch(
  genome_name,
  "hg38" = list(package = "BSgenome.Hsapiens.UCSC.hg38", species = "Homo sapiens"),
  "mm10" = list(package = "BSgenome.Mmusculus.UCSC.mm10", species = "Mus musculus"),
  stop("Unsupported genome: ", genome_name, ". Supported genomes: hg38, mm10.")
)

if (!requireNamespace(genome_config$package, quietly = TRUE)) {
  stop(
    "Required package is not installed: ", genome_config$package,
    ". Please install it before running motif matching."
  )
}

suppressPackageStartupMessages(
  library(genome_config$package, character.only = TRUE)
)

genome <- get(genome_config$package)
species <- genome_config$species

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ----------------------- step1: read peak*cell count matrix ----------------------- #
counts <- t(read.csv(input_file, header = TRUE, row.names = 1))
peak_list <- rownames(counts)

# convert to GRanges
peaks_gr <- GRanges(
  seqnames = sub("\\..*", "", peak_list),
  ranges = IRanges(
    start = as.numeric(sub(".*\\.(\\d+)\\..*", "\\1", peak_list)),
    end   = as.numeric(sub(".*\\.(\\d+)\\.(\\d+)", "\\2", peak_list))
  )
)

# ----------------------- step2: build RangedSummarizedExperiment ----------------------- #
se <- SummarizedExperiment(assays = list(counts = as.matrix(counts)), rowRanges = peaks_gr)
se_gc <- addGCBias(se, genome = genome)

# ----------------------- step3: motif databases ----------------------- #
args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- normalizePath(sub(file_arg, "", args_all[grep(file_arg, args_all)]))
script_dir <- dirname(script_path)

source(file.path(script_dir, "get_motif_list.R"))

species <- genome_config$species
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
