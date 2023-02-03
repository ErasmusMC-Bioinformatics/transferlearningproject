# Imports
library(TCGAbiolinks)
library(DESeq2)
library(biomaRt)
library(stringr)
library(bestNormalize)
library(rstudioapi)

# Set working directory to where script is (dependent on using Rstudio as API)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Source TCGA-Assembler
source("TCGA-Assembler/Module_A.R")
source("TCGA-Assembler/Module_B.R")


sCancers <- list("LAML","GBM","OV","HNSC","LUSC","LUAD","BLCA","STAD","BRCA","SKCM","LIHC",
                 "LGG","COAD","UCEC","ESCA","KIRC","SARC","CESC","MESO","KIRP","UCS","PAAD",
                 "ACC","READ","CHOL","UVM","THCA","KICH","DLBC","PRAD","THYM","PCPG","TGCT")

# For every cancer in sCancers download clinical and RNA-seq data

for(sCancer in sCancers) {
  print(sCancer)
  filename_d = paste(sCancer,"_unnormalized",sep="")
  filename_RNASeq	<- DownloadRNASeqData(cancerType	=	sCancer, outputFileName = filename_d,	assayPlatform	=	"gene_RNAseq",
                                             saveFolderName	=	"./TCGA_mrna/", tissueType = "TB")  
  output_name = paste(sCancer,"__un_GeneExp",sep="")
    gene_exp_df <- ProcessRNASeqData(inputFilePath	=	filename_RNASeq[1],	outputFileName	= output_name,	
                                     outputFileFolder	=	"./TCGA_mrna_unnormalized/",	
                                     dataType	=	"geneExp",	
                                     verType	=	"RNASeqV2")
  
  path_clinical <- DownloadBiospecimenClinicalData(sCancer, saveFolderName = sPath1)
}


# PDAC specific section

# Load clinical data
datapath_clinical = paste("Data/nationwidechildrens.org_clinical_patient_paad.txt",sep="")
clinical = read.table(datapath_clinical, header = TRUE,sep = "\t")


# Load RNA-seq data
sCancer = "PAAD"

filename = paste("TCGA_mrna_unnormalized/",sCancer,"__un_geneExp.txt",sep="")

mrna_df <- read.csv(filename,sep="\t")

filename_clinical = paste("Data/nationwidechildrens.org_clinical_patient_",tolower(sCancer),".txt",sep="")

clinical = read.csv(filename_clinical, header = TRUE,sep = "\t")

# Transform gene expression values to TPM
mrna_df_tpm <- mrna_df %>% dplyr::select(ends_with(".1"))
mrna_df_tpm <- mrna_df_tpm * 1e6
mrna_df_tpm$GeneSymbol <- mrna_df$GeneSymbol
mrna_df_tpm$EntrezID <- mrna_df$EntrezID

mrna_df <- mrna_df_tpm

#Keep only samples that have mrna and clinical data
mrna_id<-substr(colnames(mrna_df),1,12)
mrna_id<-gsub("\\.", '-', mrna_id)
colnames(mrna_df)<-substr(colnames(mrna_df),1,12)
colnames(mrna_df)<- gsub("\\.", '-', colnames(mrna_df))
clinical_id = clinical$bcr_patient_barcode[3:nrow(clinical)]
common_id = intersect(mrna_id,clinical_id)
common_id <- c("GeneSymbol","EntrezID",common_id)
mrna_df <- mrna_df[common_id]

#Keep only PDAC samples
clinical_pdac <- clinical[clinical$histologic_diagnosis == "Pancreas-Adenocarcinoma Ductal Type",]
clinical_pdac_ids <- c("GeneSymbol","EntrezID",clinical_pdac$bcr_patient_barcode)
common_pdac_id <- intersect(mrna_id,clinical_pdac_ids)
mrna_df <- mrna_df[common_pdac_id]

#Remove genes with no Gene symbol
mrna_df <- mrna_df[mrna_df$GeneSymbol != "?",]

#Remove duplicate gene symbols
mrna_df$GeneEntrez <- paste(mrna_df$GeneSymbol,"/",mrna_df$EntrezID)

#Set GeneSymbol as row names, remove EntrezId
rownames(mrna_df) <- mrna_df$GeneEntrez
mrna_df <- subset(mrna_df,select=-c(GeneSymbol,EntrezID,GeneEntrez))

#Keep features that have less than 20% 0-counts across patients
zerocountpercentage <- rowSums(mrna_df == 0)/(dim(mrna_df)[2])*100
mrna_new <- mrna_df[zerocountpercentage < 20, ]

#Normalize data with log2(x+1)
mrna_norm <- log2(mrna_new + 0.01)
mrna_norm2 <- log2(mrna_trans + 1)

#Export processed mrna data
path_export_file <- paste("C:/Users/julia/PycharmProjects/CoxTnnet/UnscaledMrna/", sCancer, ".csv",sep="")
write.csv(mrna_norm,path_export_file,quote=F,row.names = T)

for (sCancer in sCancers){
  print(sCancer)
  
  # Load RNA-seq and clinical data
  filename = paste("TCGA_mrna_unnormalized/",sCancer,"__un_geneExp.txt",sep="")
  mrna_df <- read.csv(filename,sep="\t")
  
  filename_clinical = paste("Data/nationwidechildrens.org_clinical_patient_",tolower(sCancer),".txt",sep="")
  clinical = read.csv(filename_clinical, header = TRUE,sep = "\t")
  
  # Transform gene expression values to TPM 
  mrna_df_tpm <- mrna_df %>% dplyr::select(ends_with(".1"))
  mrna_df_tpm <- mrna_df_tpm * 1e6
  mrna_df_tpm$GeneSymbol <- mrna_df$GeneSymbol
  mrna_df_tpm$EntrezID <- mrna_df$EntrezID
  
  mrna_df <- mrna_df_tpm
  
  #Keep only samples that have mrna and clinical data
  mrna_id<-substr(colnames(mrna_df),1,12)
  mrna_id<-gsub("\\.", '-', mrna_id)
  colnames(mrna_df)<-substr(colnames(mrna_df),1,12)
  colnames(mrna_df)<- gsub("\\.", '-', colnames(mrna_df))
  clinical_id = clinical$bcr_patient_barcode[3:nrow(clinical)]
  common_id = intersect(mrna_id,clinical_id)
  common_id <- c("GeneSymbol","EntrezID",common_id)
  mrna_df <- mrna_df[common_id]
  print(dim(mrna_df))
  
  #Remove samples with no Gene symbol
  mrna_df <- mrna_df[mrna_df$GeneSymbol != "?",]
  
  #Remove duplicate gene symbols
  mrna_df$GeneEntrez <- paste(mrna_df$GeneSymbol,"/",mrna_df$EntrezID)
  
  #Set GeneSymbol as row names, remove EntrezId
  rownames(mrna_df) <- mrna_df$GeneEntrez
  mrna_df <- subset(mrna_df,select=-c(GeneSymbol,EntrezID,GeneEntrez))
  
  #Keep features that have less than 20% 0-counts across patients
  zerocountpercentage <- rowSums(mrna_df == 0)/(dim(mrna_df)[2]-1)*100
  mrna_new <- mrna_df[zerocountpercentage < 20, ]
  print(dim(mrna_new))
  
  #Normalize data with bestNorm package, which selects the most appropriate transformation
  mrna_norm <- log2(mrna_new + 0.01)
  
  #Add back row names
  rownames(mrna_norm) <- rownames(mrna_new)
  
  #Export file to be scaled in python
  path_export_file <- paste("C:/Users/julia/PycharmProjects/CoxTnnet/UnscaledMrna_Correct/", sCancer, ".csv",sep="")
  write.csv(mrna_norm,path_export_file,quote=F,row.names = T)
}
