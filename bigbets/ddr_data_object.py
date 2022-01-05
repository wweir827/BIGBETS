import os,sys,re
import numpy as np
import pandas as pd
import gzip,pickle
import scipy.stats as stats
from .name_matching_scripts import match_names_to_symbols
import mygene
from .file_locations import tcga_dir
import logging
logger=logging.getLogger(__name__)
#we define this as a singleton meta-class so that there is only one instannce of class
#throughout


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DDR_Data_object(object,metaclass=Singleton):

    def __init__(self,load_chromatin=False):
        logger.info("Creating DDR Object")
        self.load_chromatin=load_chromatin
        self.data_dir=tcga_dir
        self.load_ddr()


    def load_ddr(self):
        self.protein_domain_df = pd.read_csv(os.path.join(self.data_dir, "allgene_domains_uniprot.csv"), index_col=0)
        self.prot_lengths = self.protein_domain_df.groupby(['gene'])[
            'totlen'].max()  # just take any of the values in the groups should all be the same

        # these had to look up
        miss_len = {"CDKN2A": 132, "GNAS": 394, "WISP3": 354,
                    "FAM46C": 391, "CUX1": 1505, 'U2AF1': 240,
                    "HLA-A": 365, 'HLA-B': 362}
        for k, val in miss_len.items():
            self.prot_lengths.loc[k] = val

        # Load core gene pathwasy
        self.ddr_gene_df = pd.read_excel(os.path.join(self.data_dir, "knijnenburg_core_ddr_list.xlsx"), skiprows=1, index_col=0)
        # use the pathway abbreviations for the column names
        self.ddr_gene_df.columns = map(lambda x: re.search("(?<=\()\w+(?=\))", x).group(), self.ddr_gene_df.columns)
        # print('before dropping DR and TLS', self.ddr_gene_df.shape)

        self.ddr_gene_df.drop(['DR', 'TLS'], axis=1, inplace=True)
        self.ddr_gene_df.dropna(how='all', axis=0, inplace=True)
        self.ddr_gene_df[~self.ddr_gene_df.isnull()] = 'X'
        self.ddr_gene_df.index = match_names_to_symbols(self.ddr_gene_df.index)
        # print('ddr_genes_df.shape', self.ddr_gene_df.shape)

        self.ddr_all_non_core = pd.read_excel(os.path.join(self.data_dir, "knijnenburg_ddr_all.xlsx"), index_col=0)
        self.ddr_all_non_core.columns = map(lambda x: re.search("(?<=\()\w+(?=\))", x).group(), self.ddr_all_non_core.columns)
        self.ddr_all_non_core.dropna(how='all', axis=0, inplace=True)
        self.ddr_all_non_core[~self.ddr_all_non_core.isnull()] = 'X'
        self.ddr_all_non_core.index=match_names_to_symbols(self.ddr_all_non_core.index)
        # print('ddr_genes_df.shape', self.ddr_gene_df.shape)

        self.gene_2_path_dict = {}
        self.gene_2_path_noncore_dict = {}
        self.path_2_path_noncore_dict = {}
        self.path_2_genes_dict = {}

        for gene in self.ddr_gene_df.index.values:
            paths = self.ddr_gene_df.columns.values[np.where(self.ddr_gene_df.loc[gene, :] == 'X')[0]]
            self.gene_2_path_dict[gene] = paths
        for gene in self.ddr_all_non_core.index.values:
            paths = self.ddr_all_non_core.columns.values[np.where(self.ddr_all_non_core.loc[gene, :] == 'X')[0]]
            self.gene_2_path_noncore_dict[gene] = paths

        for path in self.ddr_all_non_core.columns.values:
            genes = self.ddr_all_non_core.index.values[np.where(self.ddr_all_non_core.loc[:, path] == 'X')[0]]
            self.path_2_path_noncore_dict[path] = genes

        for path in self.ddr_gene_df.columns.values:
            genes = self.ddr_gene_df.index.values[np.where(self.ddr_gene_df.loc[:, path] == 'X')[0]]
            self.path_2_genes_dict[path] = genes

        histone_modification_file=os.path.join(self.data_dir,"olga_1638108885328.xls")
        self.histone_modification_genes_df=pd.read_excel(histone_modification_file,sheet_name='Genes')
        self.path_2_genes_dict['histone_modification_pathway']=self.histone_modification_genes_df['Symbol'].values

        #this queries go database for chromatin remodeling genes
        if self.load_chromatin:
            self.load_chromatin_remodel_genes()

        self.path_of_interest = self.path_2_genes_dict.keys()
        self.genes_of_interest = self.gene_2_path_dict.keys()


    def load_chromatin_remodel_genes(self):
        logger.info("Loading Chromatin remodelling genes from GO")

        from goatools.base import download_go_basic_obo
        obo_fname = download_go_basic_obo()
        from goatools.base import download_ncbi_associations
        gene2go = download_ncbi_associations()
        from goatools.anno.genetogo_reader import Gene2GoReader

        # get all associations in reverse
        objanno = Gene2GoReader("gene2go", taxids=[9606])  # use humans
        go2geneids_human = objanno.get_id2gos(namespace='BP', go2geneids=True)  # reverse associations
        print("{N:} GO terms associated with human NCBI Entrez GeneIDs".format(N=len(go2geneids_human)))

        from goatools.go_search import GoSearch
        srchhelp = GoSearch("go-basic.obo", go2items=go2geneids_human)



        chromat_remod_parent=['GO:0006338']
        #Just looked up the id for chromatin remodeling rather than search
        # chromat_remod_search = re.compile(r'chromatin remodeling', flags=re.IGNORECASE)
        # chromat_remod_parent = srchhelp.get_matching_gos(chromat_remod_search)
        gos_all = srchhelp.add_children_gos(chromat_remod_parent)
        # Get Entrez GeneIDs for cell cycle GOs
        geneids = srchhelp.get_items(gos_all)


        mg = mygene.MyGeneInfo()
        genetble = mg.querymany(list(geneids),
                                scopes=['entrezgene'],
                                field=['symbol'],
                                species='human',
                                as_dataframe=True)

        #     #drop missing
        #     genetble = genetble.iloc[np.where(genetble['notfound']!=True)[0],:]
        #     genetble=genetble.iloc[np.where(np.logical_not(pd.isnull(genetble['entrezgene'])))[0],:]
        #     genetble.to_csv(genetable_out)
        chromatin_remod_genes = np.sort(genetble['symbol'].values)
        # chromatin_remod_genes = chromatin_remod_genes[np.isin(chromatin_remod_genes, features)]
        self.path_2_genes_dict['chromatin_remodel'] = chromatin_remod_genes

#this gets imported and used throughout the other classes
myddr_obj=DDR_Data_object(load_chromatin=True)