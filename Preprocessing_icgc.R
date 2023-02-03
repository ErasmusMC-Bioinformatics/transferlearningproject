# Imports
library(textshape)
library(clusterProfiler)
library(org.Hs.eg.db)
library(rstudioapi)
library(tidyverse)
library(stringr)
library(RegParallel)

# Set working directory to where script is (dependent on using Rstudio as API)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load raw RNA-seq data
raw_data <- read.csv("PDAC_ICGC/exp_seq.tsv",sep="\t",header=TRUE)

# Keep only important values
subset_rna_df <- raw_data[, c("icgc_donor_id","gene_id","normalized_read_count")]


# Pivot dataframe to correct format (make it similar to TCGA data)
rna_df <- subset_rna_df %>%
  pivot_wider(id_cols = icgc_donor_id,
              names_from = gene_id,
              values_from = normalized_read_count)

rna_df <- column_to_rownames(rna_df,"icgc_donor_id")

rna_df <- rna_df[!(row.names(rna_df) %in% c('DO33168')),]

rna_df <- sapply(rna_df, unlist)

rna_df <- as.data.frame(rna_df)

donor_ids <- unique(subset_rna_df$icgc_donor_id)

donor_ids <- donor_ids[donor_ids !='DO33168']

rownames(rna_df) <- donor_ids

# Load clinical data

raw_clinical <- read.csv("PDAC_ICGC/donor.tsv",sep="\t",header=TRUE)
raw_sample <- read.csv("PDAC_ICGC/sample.tsv",sep="\t",header=TRUE)
raw_specimen  <- read.csv("PDAC_ICGC/specimen.tsv",sep="\t",header=TRUE)

# Get ID's of only PDAC patients

pdac_only_ids <- raw_specimen[raw_specimen$tumour_histological_type == "Pancreatic Ductal Adenocarcinoma",]
pdac_only_ids <- pdac_only_ids[pdac_only_ids$specimen_type == "Primary tumour - solid tissue",]
pdac_only_ids <- unique(pdac_only_ids$icgc_donor_id)

# Keep only PDAC patients no other subtypes

pdac_clinical <- subset(raw_clinical, icgc_donor_id %in% pdac_only_ids)
rna_df <- subset(rna_df, rownames(rna_df) %in% pdac_only_ids)

#Set ensembl ID to gene symbol

rna_df_symbol <- bitr(colnames(rna_df), fromType = "ENSEMBL",toType = c("SYMBOL"),OrgDb = org.Hs.eg.db)

rna_df <- rna_df[names(rna_df) %in% rna_df_symbol$ENSEMBL]

names(rna_df) <- rna_df_symbol$SYMBOL[match(names(rna_df), rna_df_symbol$ENSEMBL)]

#Remove duplicated genes

rna_df <- rna_df[!duplicated(colnames(rna_df))]

#Load pre-processed PDAC data to get genes to match to

tcga_pdac <- read.csv("C:/Users/julia/PycharmProjects/CoxTnnet/UnscaledMrna_Correct/PAAD.csv",sep=",")

gene_symbols <- as.data.frame(str_split_fixed(tcga_pdac$X, ' / ',2))

gene_symbols <- gene_symbols$V1

#Match PDAC TCGA genes to ICGC genes

rna_df <- rna_df[colnames(rna_df) %in% gene_symbols]

#Keep features that have less than 20% 0-counts across patients
zerocountpercentage <- colSums(rna_df == 0)/(dim(rna_df)[1])*100
rna_df <- rna_df[,zerocountpercentage < 20]

#Log 2 transform
rna_norm <- log2(rna_df + 0.01)

#Export file
path_export_file <- paste("C:/Users/julia/PycharmProjects/CoxTnnet/UnscaledMrna_Correct/PDAC_ICGC_2.csv",sep="")
write.csv(rna_norm,path_export_file,quote=F,row.names = T)

#Export clinical file
pdac_clinical <- pdac_clinical[pdac_clinical$icgc_donor_id !="DO33168",]
path_export_file <- paste("C:/Users/julia/PycharmProjects/CoxTnnet/ToScaleFinal/PDAC_ICGC_clinical.csv",sep="")
write.csv(pdac_clinical,path_export_file,quote=F,row.names = T)
