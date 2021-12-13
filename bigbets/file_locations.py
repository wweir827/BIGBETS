import os
import pathlib

#data directory expected to be in directory containing module.
homedir=pathlib.Path(__file__).parent.parent.resolve()
homedir=os.path.join(homedir,"data")

tcga_dir=os.path.join(homedir,'pan_can_maf')

#clinical dataset directories
other_dataset_dir=os.path.join(homedir,"Other_studies")
samstein_dir=os.path.join(other_dataset_dir,"Samstein_2019/")
imvigor_dir=os.path.join(other_dataset_dir,"IMVigor210/")
cbioportal_data_dir=os.path.join(other_dataset_dir,'combined_cbioportal')
rose_dir=os.path.join(other_dataset_dir,"Rose_2020")
braun_dir=os.path.join(other_dataset_dir,"Braun_2020")
miao_dir=os.path.join(other_dataset_dir,"Miao_2018")

gsea_dir= os.path.join(tcga_dir, 'GSEA_files')
gsea_sig_file = os.path.join(gsea_dir,'gene_signatures/GSEA_gene_sinatures.gmt')

#intermediate datafiles BiG-BET scores
bipartite_samples_dir=os.path.join(homedir,'bipartite_sampling_data')
# bigbet_scores_file=os.path.join(bipartite_samples_dir,'tcga_18kgenes_wpolyphen_rewiring_zscores.csv')
bigbet_scores_file=os.path.join(bipartite_samples_dir,'tcga_allgenes_high_wpolyphen_rewiring_zscores.csv')
bigbet_moderate_included_scores_file=os.path.join(bipartite_samples_dir,'tcga_allgenes_high_moderate_wpolyphen_rewiring_zscores.csv')
bigbet_scores_samstein_file=os.path.join(bipartite_samples_dir,'samstein_rewiring_zscores.csv')
bigbet_scores_tcga_filt_samstein_file=os.path.join(bipartite_samples_dir,'tcga_filter_samstein_cancertypes_rewiring_zscores.csv')