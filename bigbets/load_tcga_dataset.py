import os,sys,re
import numpy as np
import pandas as pd
import gzip,pickle
import scipy.stats as stats
import mygene
import sklearn.preprocessing as skp
from .name_matching_scripts import match_names_to_symbols
from .file_locations import tcga_dir
from .ddr_data_object import myddr_obj
from .load_clinical_datasets import ClinicalDataSet,load_GSEA_gene_signatures
from .bipartite_helper_functions import get_gene_signature
import logging
logger=logging.getLogger(__name__)

class TCGA_Data(ClinicalDataSet):

    def __init__(self,genelist=None,cut=5000):
        super(TCGA_Data,self).__init__()
        self.data_dir = tcga_dir
        self.cut=cut #maximun TMB before sample is excluded
        self.genelist=genelist #typically this would be the list of PMEC genes to restrict analysis to. 
        self.load_tcga()
        self.filter_tcga()
        self.load_tcga_deletions()

    def get_tcga_bar(self,x):
        return "-".join(x.split("-")[:3])

    def load_tcga(self):
        logger.info("Loading TCGA dataset")
        #only keep a subset of the columns in the dataset
        cols2keep = ['Hugo_Symbol', 'Entrez_Gene_Id', 'Center', 'NCBI_Build', 'Chromosome',
                     'Start_Position', 'End_Position', 'Strand', 'Variant_Classification',
                     'Variant_Type', 'COSMIC', 'dbSNP_RS', 'Tumor_Sample_Barcode', 'HGVSp',
                     'Consequence', 'Protein_position', 'VARIANT_CLASS', 'HGVSp_Short',
                     'all_effects', 'PolyPhen', 'SIFT', 'IMPACT', 'TSL', 'PHENO', 'FILTER']

        self.alteration_data_tcga = pd.read_table(os.path.join(self.data_dir, "mc3.v0.2.8.PUBLIC.maf"), usecols=cols2keep)

        #make sure names are HUGO symbols
        mygenes_df_file=os.path.join(self.data_dir, 'tcga_all_mg_query_df.pickle')
        if not os.path.exists(mygenes_df_file):
            self.create_tcga_gene_names_map()
        with gzip.open(mygenes_df_file, 'rb') as fh:
            my_genes_df=pickle.load(fh)


        #map over the symobols based on presearched names in the gene map dataset.
        self.alteration_data_tcga['Hugo_Symbol']=self.alteration_data_tcga['Hugo_Symbol'].map(lambda x:my_genes_df['symbol'].get(x,x) )

        # self.alteration_data_tcga = pd.read_table(os.path.join(self.data_dir_pc, "TCGA_pancancer_maf.txt"))

        logger.info('alteration_data_tcga.shape: %s',str(self.alteration_data_tcga.shape))
        self.tcga_sample_data = pd.read_table(os.path.join(self.data_dir, 'all_samples_tumor_types.csv'), index_col=0)

        # We use the TSS site to map to the id's  (There are still some though that just aren't mappable from this data (cBioportal is missing ~150 of the samples))
        tss2tumorname = dict(zip(map(lambda x: x.split('-')[1], self.tcga_sample_data.index),
                                 self.tcga_sample_data['TCGA PanCanAtlas Cancer Type Acronym']))
        self.alteration_data_tcga['analysis_group'] = self.alteration_data_tcga['Tumor_Sample_Barcode'].map(\
            lambda x: tss2tumorname[x.split('-')[1]] if x.split('-')[1] in tss2tumorname else 'None')
        self.alteration_data_tcga['pathway'] = self.alteration_data_tcga['Hugo_Symbol'].map(lambda x: myddr_obj.gene_2_path_dict[x][0] \
            if x in myddr_obj.gene_2_path_dict else 'None')

        self.alteration_data_tcga['PolyPhen_comb'] = self.alteration_data_tcga['PolyPhen'].apply(lambda x: re.sub("\(\d+\.?\d*\)", "", x))


        self.tcga_id2subtype = {}
        for k, v in self.alteration_data_tcga.groupby(['Tumor_Sample_Barcode', 'analysis_group']).groups.items():
            self.tcga_id2subtype[k[0]] = k[1]
        tcga_shortid2subtype = dict([("-".join(k.split("-")[:3]), val) for k, val in self.tcga_id2subtype.items()])


        snv_data = self.alteration_data_tcga.iloc[np.where(self.alteration_data_tcga.loc[:, 'VARIANT_CLASS'] == 'SNV')[0], :]
        indel_data = self.alteration_data_tcga.iloc[np.where(self.alteration_data_tcga.loc[:, 'VARIANT_CLASS'] != 'SNV')[0], :]


        indel_cnts = indel_data.groupby(by='Tumor_Sample_Barcode').size()
        snv_cnts = snv_data.groupby(by='Tumor_Sample_Barcode').size()
        self.tcga_snv_indel_df = pd.DataFrame(
            {'snv_cnts': snv_cnts, 'indel_cnts': indel_cnts, 'total': snv_cnts.add(indel_cnts, fill_value=0)})
        self.tcga_snv_indel_df[self.tcga_snv_indel_df.isnull()] = 0

        # READ in mutational loads
        tcga_mut_loads = pd.read_table(os.path.join(self.data_dir, 'mutation-load_updated.txt'))
        tcga_mut_loads['tmb'] = tcga_mut_loads['Non-silent per Mb']  +tcga_mut_loads['Silent per Mb']




        short2longer = dict(
            zip(map(lambda x: self.get_tcga_bar(x), self.tcga_snv_indel_df.index.values), self.tcga_snv_indel_df.index.values))
        tcga_mut_loads.index = map(lambda x: short2longer.get(self.get_tcga_bar(x), 'None'),
                                   tcga_mut_loads['Tumor_Sample_ID'].values)
        tcga_mut_loads = tcga_mut_loads.iloc[np.where(~tcga_mut_loads.index.isin(['None']))[0], :]
        self.tcga_snv_indel_df.loc[tcga_mut_loads.index, 'tmb'] = tcga_mut_loads['tmb']
        # Calculate SNV and Indel Counts for each of the tumors
        # filter out the samples that have an extreme number of SNP
        self.alteration_data_tcga = self.alteration_data_tcga.loc[self.alteration_data_tcga['Tumor_Sample_Barcode'].isin(
        self.tcga_snv_indel_df.index[np.where(self.tcga_snv_indel_df['snv_cnts'] < self.cut)[0]]), :]
        #some of the TCGA samples have Nans.  Use total snv+indel cnt to predict

        temp = self.tcga_snv_indel_df.loc[:, ['total', 'tmb']].dropna(axis=0, how='any')
        slope, inter, rval, pval, stderr = stats.linregress(temp)

        #getting missing indices and use mx+b to "predict" tmb
        missing_inds=np.where(self.tcga_snv_indel_df['tmb'].isna())[0]
        self.tcga_snv_indel_df.iloc[missing_inds,self.tcga_snv_indel_df.columns.get_loc('tmb')]=self.tcga_snv_indel_df['total'].iloc[missing_inds].apply(lambda x : slope*x+inter)


    def filter_tcga(self):
        logger.info("Filtering TCGA Dataset")
        polyphen2keep=['.','possibly_damaging','probably_damaging','unknown']

        high_consequence = ['stop_lost', 'stop_gained', 'transcript_amplification', 'transcript_ablation', 'start_lost','frameshift_variant','splice_site','translation_start_site']
        moderate_consequence = ['inframe_insertion', 'inframe_deletion', 'missense_variant', 'protein_altering_variant']

        if self.genelist is not None:
            self.tcga_alt_filt = self.alteration_data_tcga.iloc[np.where(self.alteration_data_tcga['Hugo_Symbol'].isin(self.genelist))[0], :]
        else:
            self.tcga_alt_filt= self.alteration_data_tcga.copy() #use all genes

        self.tcga_alt_filt_wmod = self.tcga_alt_filt.iloc[
                             np.where(self.tcga_alt_filt['Consequence'].isin(high_consequence + moderate_consequence))[0], :]


        self.tcga_alt_filt = self.tcga_alt_filt.iloc[np.where(self.tcga_alt_filt['Consequence'].isin(high_consequence))[0], :]
        #FILTER BY POLYPHEN
        self.tcga_alt_filt = self.tcga_alt_filt.iloc[
                                  np.where(self.tcga_alt_filt['PolyPhen_comb'].isin(polyphen2keep))[0], :]
        self.tcga_alt_filt_wmod = self.tcga_alt_filt_wmod.iloc[
                                  np.where(self.tcga_alt_filt_wmod['PolyPhen_comb'].isin(polyphen2keep))[0], :]

        print("tcga_alt_filt.shape",self.tcga_alt_filt.shape)
        self.tcga_spec_by_all_genes = self.tcga_alt_filt.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack(
            fill_value=0)
        self.tcga_spec_by_all_genes_wmod = self.tcga_alt_filt_wmod.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack(
            fill_value=0)
        self.tcga_spec_by_all_genes = (self.tcga_spec_by_all_genes > 0).astype(int)
        self.tcga_spec_by_all_genes_wmod = (self.tcga_spec_by_all_genes_wmod > 0).astype(int)

    def create_tcga_gene_names_map(self):
        """This only needs to be run once and then we can just load the file above """
        genenames = self.alteration_data_tcga['Hugo_Symbol'].unique()
        mg = mygene.MyGeneInfo()
        ressyms = mg.querymany(genenames, scopes='symbol', fields='symbol,entrezgene',
                               species='human', returnall=True)
        cnt = 0
        my_res_df = pd.DataFrame()
        for qvals in ressyms['out']:
            for k, val in qvals.items():
                my_res_df.loc[cnt, k] = val
            cnt += 1

        # drop missing items
        if 'notfound' in my_res_df.columns:
            my_res_df = my_res_df.loc[my_res_df['notfound'] != True, :]

        missing_genes = ressyms['missing']

        res_missing = mg.querymany(missing_genes, scopes='name,alias,retired,other_names', fields='symbol,entrezgene',
                                   species='human', returnall=True)

        for qvals in res_missing['out']:
            for k, val in qvals.items():
                my_res_df.loc[cnt, k] = val
            cnt += 1

        # should correspond to best gene name. probobably other ways to do the matchign as well
        # or you could try lining up with entrez id but this seems to work decently well.
        my_res_df = my_res_df.loc[my_res_df.groupby('query')['_score'].idxmax().dropna(), :]
        my_res_df.index = my_res_df['query']

        # map gene names to symbols found
        tcga_dir = "/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/TCGA/PMEC/pan_can_maf/"
        with gzip.open(os.path.join(tcga_dir, 'tcga_all_mg_query_df.pickle'), 'wb') as fh:
            pickle.dump(my_res_df, fh)

    def load_tcga_deletions(self,read=True):
        tcga_del_file=os.path.join(self.data_dir, 'tcga_cn_ddr_genes_only.csv')
        if not read or not os.path.exists(tcga_del_file):
            print('loading whitelisted gistic values.  Filtering on DDR only')
            tcga_copy_number_gistic = os.path.join(self.data_dir, 'all_thresholded.by_genes_whitelisted.tsv')
            i = 0
            for df in pd.read_table(tcga_copy_number_gistic, chunksize=5000, index_col=0):
                if i == 0:
                    tcga_cn = pd.DataFrame(columns=df.columns)
                row2keep = np.where(df.index.isin(myddr_obj.ddr_all_non_core.index))[0]
                if len(row2keep) > 0:
                    #         print(df.iloc[row2keep,0])
                    tcga_cn = pd.concat([tcga_cn, df.iloc[row2keep, :]])
                i += 1
                print(i)
            self.cn_samp_by_gene_tcga = tcga_cn.iloc[:, 2:].T
            self.cn_samp_by_gene_tcga.to_csv(tcga_del_file)
        else:
            self.cn_samp_by_gene_tcga = pd.read_csv(tcga_del_file, index_col=0)
            self.cn_samp_by_gene_tcga.columns = match_names_to_symbols(self.cn_samp_by_gene_tcga.columns) #map over genes that have different names in TCGA
        barshorttened = map(lambda x: "-".join(x.split('-')[:3]), self.cn_samp_by_gene_tcga.index)
        self.cn_samp_by_gene_tcga.index = barshorttened
        tcga_shortid2subtype = dict([("-".join(k.split("-")[:3]), val) for k, val in self.tcga_id2subtype.items()])

        allpaths = [x for x in set(tcga_shortid2subtype.values()) if str(x) not in ['nan', 'None']]

        self.cn_subtype_by_gene_tcga = pd.DataFrame(0, index=allpaths,
                                               columns=myddr_obj.ddr_all_non_core.index)

        self.cn_subtype_by_path_tcga = pd.DataFrame(0, index=allpaths,
                                               columns=myddr_obj.ddr_all_non_core.columns)
        cnt = 0
        nnz_inds = np.where(self.cn_samp_by_gene_tcga < -1)
        for i in range(len(nnz_inds[0])):
            ix = nnz_inds[0][i]
            iy = nnz_inds[1][i]
            csamp = self.cn_samp_by_gene_tcga.index[ix]
            cgene = self.cn_samp_by_gene_tcga.columns[iy]
            try:
                csub = tcga_shortid2subtype[csamp]
            except KeyError:
                continue
            if str(csub) == 'nan':
                continue
            cnt += 1
            self.cn_subtype_by_gene_tcga.loc[csub, cgene] += 1
            cpaths = myddr_obj.gene_2_path_noncore_dict[cgene]
            for cpath in cpaths:
                self.cn_subtype_by_path_tcga.loc[csub, cpath] += 1

        self.del_samp_by_gene_tcga = self.cn_samp_by_gene_tcga.copy()
        self.del_samp_by_gene_tcga[self.del_samp_by_gene_tcga >= -1] = 0
        self.del_samp_by_gene_tcga[self.del_samp_by_gene_tcga < -1] = 1

    def load_clinical_data(self):
        logger.info("Loading TCGA Clinical data")

        self.tcga_survive_file = os.path.join(self.data_dir, 'TCGA-CDR-SupplementalTableS1.xlsx')
        clinical_data_raw = pd.read_excel(self.tcga_survive_file, sheet_name='TCGA-CDR', index_col=0)
        clinical_data_raw.index = clinical_data_raw['bcr_patient_barcode']
        #There isn't a one-to-one mapping here so we have to combine some of the tumors.
        short_ids = list(map(lambda x: "-".join(x.split("-")[:3]), self.tcga_spec_by_all_genes.index))
        conver_dict = dict(zip(self.tcga_spec_by_all_genes.index, short_ids))
        rev_conv_dict = {}
        for k, val in conver_dict.items():
            rev_conv_dict[val] = rev_conv_dict.get(val, []) + [k]

        self.clinical_data = pd.DataFrame()
        for k, vals in rev_conv_dict.items():
            for val in vals:
                if k in clinical_data_raw.index:
                    self.clinical_data.loc[val, 'type'] = clinical_data_raw.loc[k, 'type']
                    self.clinical_data.loc[val, 'race'] = clinical_data_raw.loc[k, 'race']
                    self.clinical_data.loc[val, 'os'] = clinical_data_raw.loc[k, 'OS.time'] / 30
                    self.clinical_data.loc[val, 'censOS'] = clinical_data_raw.loc[k, 'OS']
                    self.clinical_data.loc[val, 'os_d'] = clinical_data_raw.loc[k, 'DSS.time'] / 30
                    self.clinical_data.loc[val, 'censOS_d'] = clinical_data_raw.loc[k, 'DSS']

                    if clinical_data_raw.loc[k, 'OS.time'] < 5 * 365:
                        self.clinical_data.loc[val, 'os_5yr'] = min(clinical_data_raw.loc[k, 'OS.time'], 5 * 365) / 30.
                        self.clinical_data.loc[val, 'censOS_5yr'] = clinical_data_raw.loc[k, 'OS']
                    else:
                        self.clinical_data.loc[val, 'os_5yr'] = (5 * 365) / 30
                        self.clinical_data.loc[val, 'censOS_5yr'] = 0

                else:
                    self.clinical_data.loc[val, 'censOS'] = np.NaN
                    self.clinical_data.loc[val, 'os'] = np.NaN
                    self.clinical_data.loc[val, 'censOS_d'] = np.NaN
                    self.clinical_data.loc[val, 'os_d'] = np.NaN

        self.clinical_data['TMB'] = self.tcga_snv_indel_df.loc[self.clinical_data.index, 'tmb']
        self.clinical_data['high_TMB'] = self.clinical_data['TMB'].apply(lambda x: "TMB-H" if x > 15 else "TMB-L")

        #for marking clincial data wth inherited method
        self.spec_by_genes=self.tcga_spec_by_all_genes
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_rnaseq_data(self):
        logger.info("Loading TCGA RNAseq data")
        tcga_rna_file = os.path.join(self.data_dir, "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv")
        self.rnaseq_df = pd.read_table(tcga_rna_file)

        gene_entrez = self.rnaseq_df['gene_id'].apply(lambda x: x.split("|")[1])
        entrez2symbolsfile=os.path.join(self.data_dir,'tcga_rnaseq_entrez_to_symbols_table.csv')
        if os.path.exists(entrez2symbolsfile):
            convtable=pd.read_csv(entrez2symbolsfile,index_col=0)
        else:
            mg = mygene.MyGeneInfo()
            res = mg.querymany(gene_entrez, scopes='entrezgene', fields='symbol', returnall=True, as_dataframe=True)
            convtable=res['out']
            convtable.to_csv(entrez2symbolsfile)

        symbols = np.array(list(map(lambda x: convtable['symbol'].get(x, 'Not_found'), gene_entrez)))
        self.rnaseq_df['gene_id'] = symbols

        self.rnaseq_df = self.rnaseq_df.set_index('gene_id')
        self.rnaseq_df = self.rnaseq_df.loc[self.rnaseq_df.index.isin(self.tcga_spec_by_all_genes.columns), :]


        short_ids = list(map(self.get_tcga_bar, self.tcga_spec_by_all_genes.index))
        id2longer = dict(zip(short_ids, self.tcga_spec_by_all_genes.index))
        self.rnaseq_df.columns = list(map(self.get_tcga_bar, self.rnaseq_df.columns))
        self.rnaseq_df = self.rnaseq_df.loc[:, self.rnaseq_df.columns.isin(id2longer)]
        self.rnaseq_df.columns = list(map(lambda x: id2longer[x], self.rnaseq_df.columns))

        #log2(1+p) and do robust scaling across the different samples.
        self.log_all_rna_exp = np.log2(self.rnaseq_df + 1)
        scaler = skp.RobustScaler().fit(self.log_all_rna_exp)
        self.log_all_rna_exp_scaled = pd.DataFrame(scaler.transform(self.log_all_rna_exp), index=self.log_all_rna_exp.index,
                                              columns=self.log_all_rna_exp.columns)
        self.log_all_rna_exp_scaled = self.log_all_rna_exp_scaled.T

        #construct GSEA signatures as well.
        self.get_GSEA_signatures()

    def get_GSEA_signatures(self):
        gene_sigs=load_GSEA_gene_signatures()
        self.ig_signatures = pd.DataFrame(index=self.log_all_rna_exp_scaled.index)
        for sig, genes in gene_sigs.items():
            cgenes = np.array(genes)[np.where(np.isin(genes, self.log_all_rna_exp_scaled.columns))[0]]
            self.ig_signatures.loc[:, sig] = get_gene_signature(self.log_all_rna_exp_scaled, cgenes)