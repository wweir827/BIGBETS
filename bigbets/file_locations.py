import os

homedir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/TCGA/PMEC"
tcga_dir=os.path.join(homedir,'pan_can_maf')

#clinical dataset directories
other_dataset_dir=os.path.join(homedir,"Other_studies")
samstein_dir=os.path.join(other_dataset_dir,"Samstein_2019/")
imvigor_dir=os.path.join(other_dataset_dir,"IMVigor210/")
cbioportal_data_dir=os.path.join(other_dataset_dir,'combined_cbioportal')
rose_dir=os.path.join(cbioportal_data_dir,"rose_2020")
braun_dir=os.path.join(other_dataset_dir,"Braun_2020")
miao_dir=os.path.join(other_dataset_dir,"Miao_2018")

gsea_dir= os.path.join(tcga_dir, 'GSEA_files')
gsea_sig_file = os.path.join(gsea_dir,'gene_signatures/GSEA_gene_sinatures.gmt')

#intermediate datafiles BiG-BET scores
bigbet_scores_file=os.path.join(tcga_dir,'tcga_18kgenes_wpolyphen_rewiring_zscores.csv')
