#!/usr/bin/env Rscript

#======================#
# Get command arguments
#======================#
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript run_cicero.R <input_csv> <genome> <output_txt> <output_rdata>")
}

input_csv   <- args[1]   # e.g. "ATACmatrix.csv"
genome_name <- args[2]   # e.g. "mm10"
output_txt  <- args[3]   # e.g. "ISSAAC_cicero_connections_mm10.txt"
output_rdata<- args[4]   # e.g. "ISSAAC_cicero_connections_mm10.RData"

cat("Running Cicero pipeline with parameters:\n")
cat("Input CSV: ", input_csv, "\n")
cat("Genome: ", genome_name, "\n")
cat("Output txt: ", output_txt, "\n")
cat("Output RData: ", output_rdata, "\n")

#======================#
# Load libraries
#======================#
suppressPackageStartupMessages({
  library(cicero)
  library(Matrix)
  library(data.table)
  library(Seurat)
  library(Signac)
  library(EnsDb.Mmusculus.v79)
  library(GenomicRanges)
  library(rtracklayer)
  library(reshape2)
  library(magrittr)
  library(R.utils)
  library(monocle3)
})

options(stringsAsFactors = FALSE)

#======================#
# Step1: Load ATAC data
#======================#
atac_counts <- fread(input_csv)
atac_counts <- as.data.frame(atac_counts)
rownames(atac_counts) <- atac_counts[[1]]
atac_counts <- atac_counts[, -1]
counts <- t(atac_counts)

#======================#
# Step2: Transform for Cicero
#======================#
chunk_size <- 1000
peaks <- rownames(counts)
cells <- colnames(counts)
result <- list()

for (i in seq(1, nrow(counts), chunk_size)) {
  chunk <- counts[i:min(i+chunk_size-1, nrow(counts)), ]
  result[[i]] <- data.table(
    peak = rep(peaks[i:min(i+chunk_size-1, nrow(counts))], each = ncol(counts)),
    cell = rep(cells, times = nrow(chunk)),
    count = as.vector(as.matrix(chunk))
  )[count > 0]  
}

counts_frag <- rbindlist(result)
colnames(counts_frag) <- c("Var1", "Var2", "Freq")
counts_frag <- as.data.frame(counts_frag)
input_cds <- make_atac_cds(counts_frag, binarize = TRUE)

#======================#
# Step3: Run Cicero
#======================#
set.seed(2017)
peaks_genome <- rownames(atac_counts)
rm(atac_counts)

input_cds <- monocle3::detect_genes(input_cds)
input_cds <- preprocess_cds(
  input_cds,
  method = "PCA",  
  num_dim = 50,   
  norm_method = "none"  
)
input_cds <- reduce_dimension(input_cds, max_components = 2, num_dim=6,
                             reduction_method = 'tSNE', norm_method = "none", 
                             preprocess_method = 'PCA')

tsne_coords <- reducedDims(input_cds)[["tSNE"]]
cicero_cds <- cicero::make_cicero_cds(input_cds, reduced_coordinates = tsne_coords)

# Load genome info
mm_genome <- getChromInfoFromUCSC(genome_name)
mm_genome <- mm_genome[!grepl("_", mm_genome$chrom), c("chrom", "size")]

conns_cicero <- run_cicero(cicero_cds, mm_genome)
conns_cicero$coaccess[is.na(conns_cicero$coaccess)] <- 0

#======================#
# Save results
#======================#
write.table(conns_cicero, file = output_txt, sep = "\t", quote = FALSE, row.names = FALSE)
save(conns_cicero, file = output_rdata)

cat("Cicero run completed successfully!\n")
