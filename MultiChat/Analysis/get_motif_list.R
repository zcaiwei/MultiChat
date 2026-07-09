# motif_functions.R

getMotifList <- function(motifSet, species, collection, version) {
  if (tolower(motifSet) == "jaspar2024") {
    if (!requireNamespace("JASPAR2024", quietly = TRUE)) {
      stop("Package 'JASPAR2024' is not installed.")
    }
    library(JASPAR2024)
    motifs <- getJasparMotifs(species)  # 修改此处以使用 getJasparMotifs
  } else if (tolower(motifSet) == "jaspar2020") {
    if (!requireNamespace("JASPAR2020", quietly = TRUE)) {
      stop("Package 'JASPAR2020' is not installed.")
    }
    library(JASPAR2020)
    motifs <- TFBSTools::getMatrixSet(JASPAR2020, list(species = species, collection = collection))
  } else if (tolower(motifSet) == "jaspar2018") {
    if (!requireNamespace("JASPAR2018", quietly = TRUE)) {
      stop("Package 'JASPAR2018' is not installed.")
    }
    library(JASPAR2018)
    motifs <- TFBSTools::getMatrixSet(JASPAR2018, list(species = species, collection = collection))
  } else if (tolower(motifSet) == "jaspar2016") {
    if (!requireNamespace("JASPAR2016", quietly = TRUE)) {
      stop("Package 'JASPAR2016' is not installed.")
    }
    library(JASPAR2016)
    motifs <- TFBSTools::getMatrixSet(JASPAR2016, list(species = species, collection = collection))
  } else if (tolower(motifSet) == "cisbp") {
    if (!requireNamespace("chromVARmotifs", quietly = TRUE)) {
      stop("Package 'chromVARmotifs' is not installed.")
    }
    library(chromVARmotifs)
    if (tolower(species) == "homo sapiens") {
      if (version == 1) {
        data("human_pwms_v1", package = "chromVARmotifs")
        motifs <- human_pwms_v1        
      } else if (version == 2) {
        data("human_pwms_v2", package = "chromVARmotifs")
        motifs <- human_pwms_v2
      } else {
        stop("Only versions 1 and 2 exist for CISBP!")
      }
    } else if (tolower(species) == "mus musculus") {
      if (version == 1) {
        data("mouse_pwms_v1", package = "chromVARmotifs")
        motifs <- mouse_pwms_v1        
      } else if (version == 2) {
        data("mouse_pwms_v2", package = "chromVARmotifs")
        motifs <- mouse_pwms_v2
      } else {
        stop("Only versions 1 and 2 exist for CISBP!")
      }
    } else {
      stop("Species not recognized. Supported: 'homo sapiens', 'mus musculus' for CISBP!")
    }
  } else if (tolower(motifSet) == "encode") {
    if (!requireNamespace("chromVARmotifs", quietly = TRUE)) {
      stop("Package 'chromVARmotifs' is not installed.")
    }
    library(chromVARmotifs)
    data("encode_pwms", package = "chromVARmotifs")
    motifs <- encode_pwms
  } else if (tolower(motifSet) == "homer") {
    if (!requireNamespace("chromVARmotifs", quietly = TRUE)) {
      stop("Package 'chromVARmotifs' is not installed.")
    }
    library(chromVARmotifs)
    data("homer_pwms", package = "chromVARmotifs")
    motifs <- homer_pwms
  } else if (tolower(motifSet) == "vierstra") {
    if (tolower(collection) == "individual") {
      url <- "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/Vierstra_Individual_Motifs.rds"
      motifs <- readRDS(url(url))
    } else if (tolower(collection) == "archetype") {
      url <- "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/Vierstra_Archetype_Motifs_v2.1.rds"
      motifs <- readRDS(url(url))
    } else {
      stop("Collection not recognized for Vierstra. Accepted: 'individual', 'archetype'.")
    }
  } else {
    stop("MotifSet not recognized!")
  }
  return(motifs)
}

# get_peak_TF_links <- function(peaks_bed, species, genome, motifSet = "jaspar2024", collection = "CORE", version = 2) {
#   motifs <- getMotifList(motifSet, species, collection, version)
#   peaks_new <- GRanges(seqnames = peaks_bed$R.chrom,
#                        ranges = IRanges(start = peaks_bed$R.start, end = peaks_bed$R.end))
#   motif_ix <- matchMotifs(motifs, peaks_new, genome, out = "scores",p.cutoff=1e-10)
#   S <- as.matrix(motif_ix@assays@data$motifScores)
#   M <- as.matrix(motif_ix@assays@data$motifMatches)
#   TF <- motif_ix@colData$name
  
#   L_TF_list <- list()
#   for (j in 1:nrow(M)) {
#     if (sum(M[j,]) > 0) {
#       p <- paste0(peaks_bed$R.chrom[j], "_", peaks_bed$R.start[j], "_", peaks_bed$R.end[j])
#       TF_j = TF[M[j,]]
#       if (length(TF_j) > 0) {
#         L_TF_list[[j]] <- data.frame(loci = p, TF = TF_j)
#       }
#     }
#   }
#   L_TF_record <- do.call(rbind, L_TF_list)
#   return(L_TF_record)
# }

get_peak_TF_links <- function(peaks_bed, species, genome, motifSet = "jaspar2024", collection = "CORE", version = 2) {
  motifs <- getMotifList(motifSet, species, collection, version)
  peaks_new <- GRanges(seqnames = peaks_bed$R.chrom,
                       ranges = IRanges(start = peaks_bed$R.start, end = peaks_bed$R.end))
  motif_ix <- matchMotifs(motifs, peaks_new, genome, out = "scores",p.cutoff=1e-15)
  S <- as.matrix(motif_ix@assays@data$motifScores)
  M <- as.matrix(motif_ix@assays@data$motifMatches)
  TF <- motif_ix@colData$name
  
  L_TF_list <- list()
  for (j in 1:nrow(M)) {
    if (sum(M[j,]) > 0) {
      p <- paste0(peaks_bed$R.chrom[j], "_", peaks_bed$R.start[j], "_", peaks_bed$R.end[j])
      TF_j = TF[M[j,]]
      S_j = S[j, M[j,]]  # 获取得分
      if (length(TF_j) > 0) {
        L_TF_list[[j]] <- data.frame(loci = p, TF = TF_j, score = S_j)  # 添加得分到数据框中
      }
    }
  }
  L_TF_record <- do.call(rbind, L_TF_list)
  return(L_TF_record)
}
