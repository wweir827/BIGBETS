import os,sys,re
import gzip,pickle
import scipy.stats as stats
import numpy as np
import pandas as pd
import logging
logger=logging.getLogger(__name__)
import sklearn.preprocessing as skp
from .name_matching_scripts import match_names_to_symbols
from .ddr_data_object import myddr_obj
from .file_locations import *


def load_GSEA_gene_signatures():
    gene_sigs = {}
    with open(gsea_sig_file, 'r') as fh:
        for line in fh.readlines():
            ls = line.strip().split('\t')
            gene_sigs[ls[0]] = ls[1:]
    gene2ig_sig = {}
    for k, val in gene_sigs.items():
        for gene in val:
            gene2ig_sig[gene] = k

    return gene_sigs

class ClinicalDataSet():
    if os.path.exists(bigbet_scores_file):
        # bigbet_scores_df = pd.read_csv(bigbet_scores_file,index_col=0)
        bigbet_scores_df = pd.read_csv(bigbet_moderate_included_scores_file,index_col=0)

    else:
        bigbet_scores_df=None
        logger.info("BiG-BETS scores not detected.  Will not be able to annotate clinical datasets.")
    def __init__(self):
        pass
    
    def add_big_bets_category(self,genes,label):

        gene_mutated_samples = self.spec_by_genes.index[
            np.where(np.sum(self.spec_by_genes.loc[:, genes], axis=1) > 0)[0]]
        # print('gene_mutated_samples, genes, samples', len(genes), len(gene_mutated_samples))
        self.clinical_bigbets_df[label] = 'WT'
        self.clinical_bigbets_df.loc[self.clinical_bigbets_df.index.intersection(gene_mutated_samples), label] = 'MUT'
        self.clinical_bigbets_df['{:}_TMB'.format(label)] = list(
            map(lambda x: "_".join([str(val) for val in self.clinical_bigbets_df.loc[x, ['high_TMB', label]]]),
                self.clinical_bigbets_df.index))
        self.clinical_bigbets_df['{:}:highTMB'.format(re.sub("_","",label))] = (self.clinical_bigbets_df['high_TMB'] == 'TMB-H') * (
                self.clinical_bigbets_df[label] == 'MUT').astype(int)
        return gene_mutated_samples

    def add_all_big_bets_categories(self,bigbets_df,drop_intersection=True):
        """

        :param bigbets_df: dataframe with BiG-BET scores.
        :return:
        """
        if bigbets_df is None:
            logger.info("Unable to annotate clinical dataset.")
            return

        bigbets_df['path'] = list(map(lambda x: myddr_obj.gene_2_path_dict.get(x, ['None'])[0], bigbets_df.index))

        bigbets_filt_df = bigbets_df.iloc[np.where(bigbets_df.index.isin(self.spec_by_genes.columns))[0], :]
        
        ddr_bigbets_filt_df = bigbets_filt_df.iloc[np.where(~bigbets_filt_df['path'].isin(['None',np.nan]))  [0], :]
        ddr_bigbets_filt_df.sort_values(by='bigbets', inplace=True)
        ddr_genes_present=ddr_bigbets_filt_df.index
        self.clinical_bigbets_df=self.clinical_data.copy()
        self.add_big_bets_category(ddr_genes_present,'DDR')

        cut = 0
        high_ddr_genes = ddr_bigbets_filt_df.index[np.where(ddr_bigbets_filt_df['bigbets'] > np.abs(cut))]
        ddr_high_gene_mutated=self.add_big_bets_category(high_ddr_genes,'DDR_high_z')

        #take into account intersection.
        low_ddr_genes = ddr_bigbets_filt_df.index[np.where(ddr_bigbets_filt_df['bigbets'] < cut)]
        ddr_low_gene_mutated=self.add_big_bets_category(low_ddr_genes,'DDR_low_z')
        if drop_intersection:
            self.clinical_bigbets_df.loc[set(ddr_low_gene_mutated).intersection(ddr_high_gene_mutated),'DDR_low_z'] =  'WT'
            self.clinical_bigbets_df['DDR_low_z_TMB'] = list(
                map(lambda x: "_".join([str(val) for val in self.clinical_bigbets_df.loc[x, ['high_TMB', 'DDR_low_z']]]),self.clinical_bigbets_df.index))
            self.clinical_bigbets_df['DDRlowz:highTMB'] = (self.clinical_bigbets_df['high_TMB'] == 'TMB-H') * (self.clinical_bigbets_df['DDR_low_z'] == 'MUT').astype(int)


        if "mwu_tcga" in ddr_bigbets_filt_df.columns:
            low_mwu_genes = ddr_bigbets_filt_df['mwu_tcga'].sort_values(ascending=False)[:len(low_ddr_genes)].index
            self.add_big_bets_category(low_mwu_genes, 'DDR_low_mwu')



        if 'histone_modification_pathway' in myddr_obj.path_2_genes_dict.keys():
            # select for histone modification and chromatin remodelling genes
            histone_genes = myddr_obj.path_2_genes_dict['histone_modification_pathway']
            histone_genes = histone_genes[np.isin(histone_genes, bigbets_filt_df.index)]
            hist_zscores = bigbets_filt_df.loc[histone_genes, :]
            low_hist_genes = hist_zscores.index[np.where(hist_zscores['bigbets'] < cut)]
            self.add_big_bets_category(low_hist_genes, 'hist_lowz')

        if 'chromatin_remodel' in myddr_obj.path_2_genes_dict.keys():
            chromatin_genes = myddr_obj.path_2_genes_dict['chromatin_remodel']
            chromatin_genes = chromatin_genes[np.isin(chromatin_genes, bigbets_filt_df.index)]
            chromatin_zscores = bigbets_filt_df.loc[chromatin_genes, :]
            low_chromatin_genes = chromatin_zscores.index[np.where(chromatin_zscores['bigbets'] < cut)]
            self.add_big_bets_category(low_chromatin_genes, 'chromatin_lowz')



        # set individual DDR path information
        for path, genes in myddr_obj.path_2_genes_dict.items():
            cgenes = genes[np.isin(genes, self.spec_by_genes.columns)]
            self.clinical_bigbets_df['{}'.format(path)] = 'WT'
            cmut_samps = self.spec_by_genes.index[np.where(np.sum(self.spec_by_genes.loc[:, cgenes], axis=1) > 0)[0]]
            logging.info("%s %d",path, len(cmut_samps))
            self.clinical_bigbets_df.loc[cmut_samps, '{}'.format(path)] = 'MUT'



class SamsteinData(ClinicalDataSet):
    def __init__(self):
        logging.info("Loading Samstein clinical data")
        super(SamsteinData,self).__init__()
        #load mutation data
        self.data_dir=samstein_dir
        self.load_genomic_data()
        self.load_clinical_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)
        self.clinical_bigbets_df = self.clinical_bigbets_df.iloc[
                                   np.where(self.clinical_bigbets_df['DRUG_TYPE'].isin(['PD-1/PDL-1']))[0],:]

    def load_clinical_data(self,tmb_thresh=10):
        logging.info("Loading Samstein clinical data")

        self.clinical_data = pd.read_table(os.path.join(self.data_dir, 'data_clinical_patient.txt'), skiprows=4)
        
        clinical_data2 = pd.read_table(os.path.join(self.data_dir, 'data_clinical_sample.txt'), skiprows=4)

        self.clinical_data = self.clinical_data.merge(clinical_data2, on='PATIENT_ID')
        self.clinical_data.set_index('PATIENT_ID', inplace=True)
        self.clinical_data['TMB'] = self.clinical_data['TMB_SCORE']


        # self.clinical_data['high_TMB']=self.clinical_data.loc[:,['CANCER_TYPE',"TMB"]].apply(get_TMB_status,axis=1)
        self.clinical_data['high_TMB'] = self.clinical_data['TMB'].apply(lambda x: 'TMB-H' if x > tmb_thresh else 'TMB-L')

        self.clinical_data['os'] = self.clinical_data['OS_MONTHS']
        self.clinical_data['censOS'] = (self.clinical_data['OS_STATUS'] == 'DECEASED').astype(int)

        self.clinical_data['nframe_shifts'] = 0
        patient_frame_shifts = self.frame_shift_muts.groupby(['PATIENT_ID']).size()
        self.clinical_data.loc[patient_frame_shifts.index, 'nframe_shifts'] = patient_frame_shifts.values
        self.clinical_data['cohort']='samstein'

    def load_genomic_data(self):
        self.all_mutations = pd.read_table(os.path.join(self.data_dir, 'data_mutations_extended.txt'))
        col2keep = ['Hugo_Symbol', 'Consequence', 'Start_Position', 'End_Position',
                    'Variant_Classification', 'Variant_Type', 'HGVSp', 'IMPACT', 'Tumor_Sample_Barcode']
        self.all_mutations = self.all_mutations.loc[:, col2keep]

        orig_names = self.all_mutations['Hugo_Symbol'].unique()
        new_names = match_names_to_symbols(orig_names)
        gene_name_map = dict(zip(orig_names, new_names))

        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Hugo_Symbol'].apply(
            lambda x: gene_name_map.get(x, x))

        def get_sample_id(x):
            return "-".join(x.split('-')[:2])

        self.all_mutations['PATIENT_ID'] = self.all_mutations['Tumor_Sample_Barcode'].apply(get_sample_id)

        consequence_to_keep = ['missense_mutation', 'nonsense_mutation',
                               'frame_shift_del', 'frame_shift_ins', 'translation_start_site', 'nonstop_mutation']

        self.all_mutations['Variant_Classification'] = self.all_mutations['Variant_Classification'].apply(
            lambda x: x.lower())
        self.all_mutations = self.all_mutations.iloc[
                             np.where(self.all_mutations['Variant_Classification'].isin(consequence_to_keep))[0], :]

        self.spec_by_genes = (self.all_mutations.groupby(['PATIENT_ID', 'Hugo_Symbol']).size().unstack(
            fill_value=0) > 0).astype(
            int)

        self.frame_shift_muts = self.all_mutations.loc[
                                self.all_mutations['Variant_Classification'].isin(
                                    ['frame_shift_del', 'frame_shift_ins']), :]


class IMVigorData(ClinicalDataSet):

    def __init__(self):
        # load mutation data
        super(IMVigorData, self).__init__()
        self.data_dir = imvigor_dir
        self.load_clinical_data()
        self.load_genonomic_data()
        self.load_rnaseq_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_clinical_data(self, tmb_thresh=10):
        logging.info("Loading IMVIigor clinical data")
        def get_response(val):
            if val in ['PD', 'SD']:
                return 'SD/PD'
            elif val in ['CR', 'PR']:
                return 'CR/PR'
            else:
                return None

        self.data_dir = imvigor_dir
        # load sample phenotype data
        self.clinical_data_file = os.path.join(self.data_dir, "fone_pheno_data.csv")
        self.clinical_data = pd.read_csv(self.clinical_data_file, doublequote=True, index_col=0)

        self.s1ind2anon_id = dict(zip(self.clinical_data.index, self.clinical_data['ANONPT_ID']))

        clinical_data_file = os.path.join(self.data_dir, "additional_pheno_data.csv")
        add_samle_data = pd.read_csv(clinical_data_file, doublequote=True, index_col=0)

        self.clinical_data = pd.merge(self.clinical_data, add_samle_data, on=u'ANONPT_ID', how='inner', suffixes=["", "_cdf"])
        self.clinical_data.index = self.clinical_data['ANONPT_ID']
        self.clinical_data = self.clinical_data.iloc[np.where(~self.clinical_data.index.duplicated())[0], :]
        self.clinical_data['best_response'] = self.clinical_data['Best Confirmed Overall Response']

        self.clinical_data['binaryResponse']=self.clinical_data['best_response'].apply(get_response)
        # sample genomic data


        self.clinical_data['high_TMB'] = self.clinical_data["FMOne mutation burden per MB"].apply(
            lambda x: 'TMB-H' if x > tmb_thresh else 'TMB-L')
        self.clinical_data['TMB'] = self.clinical_data["FMOne mutation burden per MB"]
        self.clinical_data['cohort'] = 'IMVigor210'

    def load_genonomic_data(self):
        logging.info("Loading IMVIigor genomic data")
        short_vars_file = os.path.join(self.data_dir, "fone_assay_known_short.csv")
        short_vars_likely_file = os.path.join(self.data_dir, "fone_assay_likely_short.csv")
        dels_file = os.path.join(self.data_dir, "fone_assay_deletion.csv")
        amps_file = os.path.join(self.data_dir, "fone_assay_amplification.csv")

        short_var_df = pd.read_csv(short_vars_file, index_col=0, doublequote=True)
        short_var_likely_df = pd.read_csv(short_vars_likely_file, index_col=0, doublequote=True)
        deletions_df = pd.read_csv(dels_file, index_col=0, doublequote=True)
        amplifications_df = pd.read_csv(amps_file, index_col=0, doublequote=True)

        # change over the sample names here
        short_var_df.columns = map(lambda x: self.s1ind2anon_id[x], short_var_df.columns)
        short_var_likely_df.columns = map(lambda x: self.s1ind2anon_id[x], short_var_likely_df.columns)
        deletions_df.columns = map(lambda x: self.s1ind2anon_id[x], deletions_df.columns)
        amplifications_df.columns = map(lambda x: self.s1ind2anon_id[x], amplifications_df.columns)

        self.all_mutations = short_var_df.unstack().dropna().to_frame(name='description').reset_index()


        self.all_mutations.columns = ['Specimen ID', 'Hugo_Symbol', 'description']
        self.all_mutations['functional_status'] = 'known'

        self.all_mutations_likely = short_var_likely_df.unstack().dropna().to_frame(name='description').reset_index()
        self.all_mutations_likely.columns = ['Specimen ID', 'Hugo_Symbol', 'description']
        self.all_mutations_likely['functional_status'] = 'likely'

        self.all_mutations = pd.concat([self.all_mutations, self.all_mutations_likely])
        self.all_mutations.index = np.arange(self.all_mutations.shape[0])

        orig_names = self.all_mutations['Hugo_Symbol'].unique()
        new_names = match_names_to_symbols(orig_names)
        gene_name_map = dict(zip(orig_names, new_names))
        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Hugo_Symbol'].apply(lambda x: gene_name_map.get(x, x))

        self.spec_by_genes = (self.all_mutations.groupby(['Specimen ID', 'Hugo_Symbol']).size() > 0).astype(int).unstack(fill_value=0)

        # Drop the samples with no phenotype data
        self.spec_by_genes = self.spec_by_genes.iloc[np.where(self.spec_by_genes.index.isin(self.clinical_data.index))[0], :]
        self.spec_by_genes = self.spec_by_genes.iloc[:, np.where(np.sum(self.spec_by_genes > 0, axis=0) > 0)[0]]

    def load_rnaseq_data(self):
        logging.info("Loading IMVigor RNAseqdata")
        self.gene_sigs=load_GSEA_gene_signatures()
        all_ig_signatures = np.array([gene for genes in self.gene_sigs.values() for gene in genes])
        rnaseq_file = os.path.join(imvigor_dir, "imvigor_rnaseq_counts.csv")
        raw_rnaseq_df = pd.read_csv(rnaseq_file, index_col=0)
        entrez2symbol_table = pd.read_csv(os.path.join(imvigor_dir, 'entrez2symbols.csv'), index_col=0)
        entr2symbol = dict(zip(entrez2symbol_table['entrez_id'], entrez2symbol_table['symbol']))
        # convert back to symbol
        raw_rnaseq_df.index = list(map(lambda x: entr2symbol[x], raw_rnaseq_df.index))

        # change sample IDs
        sample_df = pd.read_csv(os.path.join(imvigor_dir, 'imvigor_sample_info.csv'), index_col=0)
        sampid2anoptid = dict(zip(sample_df.index, sample_df['ANONPT_ID']))
        raw_rnaseq_df.columns = list(map(lambda x: sampid2anoptid[x], raw_rnaseq_df.columns))

        raw_rnaseq_df = raw_rnaseq_df.iloc[np.where(np.logical_not(raw_rnaseq_df.index.isna()))[0], :]

        genecounts = np.sum(raw_rnaseq_df > 0, axis=1)
        genetotcounts = np.sum(raw_rnaseq_df, axis=1)

        #filter criteria for low expressed genes
        num_samps_required = 5
        tot_count_required = 10

        gene2remove = genecounts.index[genecounts < num_samps_required]
        gene2remove2 = genetotcounts.index[genetotcounts < tot_count_required]


        gene2remove = np.array(list(set(gene2remove).union(gene2remove2)))

        # take 3 min to run
        #mapping to the gene symbols
        genetable_file = os.path.join(imvigor_dir, "mygene_symbol2coding.csv")
        genetble = pd.read_csv(genetable_file)

        symbol2proteincoding = dict(zip(genetble['query'], genetble['ensembl.type_of_gene']))
        symbol2entrez = dict(
            [(k, int(val)) for k, val in dict(zip(genetble['query'], genetble['entrezgene'])).items() if
             str(val) != 'nan'])
        entrez2symbol = dict([(val, k) for k, val in symbol2entrez.items()])

        #remove low count genes
        self.rnaseq_filter_untransformed = raw_rnaseq_df.drop(gene2remove, axis=0)
        #remove non-coding genes
        non_coding = self.rnaseq_filter_untransformed.index[
            list(map(lambda x: symbol2proteincoding.get(x, 'None') != 'protein_coding',
                     self.rnaseq_filter_untransformed.index))]

        # keep in markers for signature genes
        non_coding = non_coding[np.where(np.logical_not(non_coding.isin(all_ig_signatures)))[0]]
        self.rnaseq_filter_untransformed.drop(non_coding, axis=0, inplace=True)
        self.rnaseq_filter_untransformed = self.rnaseq_filter_untransformed.iloc[np.where(np.logical_not(self.rnaseq_filter_untransformed.index.duplicated()))[0],:]

        log_all_rna_exp = np.log2(self.rnaseq_filter_untransformed + 1)
        log_all_rna_exp.to_csv(os.path.join(imvigor_dir, 'logtrans_filtered_imvigor_rnaseq.csv'))
        scaler = skp.RobustScaler().fit(log_all_rna_exp)
        self.rnaseq_log_robust_filtered = pd.DataFrame(scaler.transform(log_all_rna_exp), index=log_all_rna_exp.index,columns = log_all_rna_exp.columns)
        self.rnaseq_log_robust_filtered = self.rnaseq_log_robust_filtered.T  # take transpose

class CBioportalCombined(ClinicalDataSet):

    def __init__(self):
        super(CBioportalCombined, self).__init__()
        self.data_dir = cbioportal_data_dir
        self.all_data_dirs = []
        self.load_genomic_data()
        self.load_clinical_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_clinical_data(self):
        self.clinical_data = pd.DataFrame()
        #align columns as best as possible between different datasets
        clinical_cols = {'skcm_dfci_2015': {'OS_MONTHS': 'os', 'OS_STATUS': 'censOS',
                                            'DURABLE_CLINICAL_BENEFIT': "RECIST", "TREATMENT": "TREATMENT"},
                         'skcm_mskcc_2014': {'OS_MONTHS': 'os', 'OS_STATUS': 'censOS',
                                             'TREATMENT_RESPONSE': "RECIST", "TREATMENT": 'TREATMENT'},
                         'ccrcc_dfci_2019': {'OS_MONTHS': 'os', 'OS_CENSOR': 'censOS',
                                             'RECIST': "RECIST", "DRUG": "TREATMENT"},
                         'mel_ucla_2016': {'OS_MONTHS': 'os', 'OS_STATUS': 'censOS',
                                           'DURABLE_CLINICAL_BENEFIT': "RECIST", "TREATMENT": "TREATMENT"},
                         'nsclc_mskcc_2018': {'PFS_MONTHS': 'os', 'PFS_STATUS': 'censOS',
                                              'BEST_OVERALL_RESPONSE': "RECIST"}, #have to use PFS for these last two as they did not have OS
                         'nsclc_mskcc_2015': {'PFS_MONTHS': 'os', 'PFS_STATUS': 'censOS',
                                              'OVERALL_RESPONSE': "RECIST"}}

        for dirs in os.listdir(self.data_dir):
            cdir = os.path.join(self.data_dir, dirs)
            if os.path.isdir(cdir):
                for datafile in os.listdir(cdir):
                    cdatafile = os.path.join(cdir, datafile)
                    if re.search("data_clinical_patient", datafile):

                        ccombined_clinical_data = pd.read_table(cdatafile, skiprows=4)
                        ccombined_clinical_data['cohort'] = dirs

                        for k, val in clinical_cols[dirs].items():
                            if k == 'OS_STATUS':
                                ccombined_clinical_data[val] = ccombined_clinical_data[k].apply(
                                    lambda x: 1 if re.search("1", x) else 0)
                            else:
                                ccombined_clinical_data[val] = ccombined_clinical_data[k]
                        if self.clinical_data.shape[0] == 0:
                            self.clinical_data = ccombined_clinical_data
                        else:
                            self.clinical_data = pd.concat([self.clinical_data, ccombined_clinical_data], axis=0,
                                                             join='inner')

        self.clinical_data['PATIENT_ID'] = self.clinical_data['PATIENT_ID'].apply(
            lambda x: re.sub('(?<=nsclc_mskcc_2018)p', 's', x))
        self.clinical_data.index = self.clinical_data['PATIENT_ID']
        self.clinical_data['binaryResponse'] = self.clinical_data['RECIST'].apply(
            lambda x: 'CR/PR' if x in ['PR', 'CR', 'Yes', 'response', 'Partial Response',
                                       'Stable Response'] else 'SD/PD')
        self.clinical_data['censOS'] = self.clinical_data['censOS'].apply(
            lambda x: 1 if re.search("1", str(x)) else 0)

        cancer_types = {'skcm_dfci_2015': 'Melanoma', 'skcm_mskcc_2014': 'Melanoma',
                        'ccrcc_dfci_2019': 'Renal Cell Carcinoma',
                        'mel_ucla_2016': 'Melanoma', 'nsclc_mskcc_2018': 'Non-Small Cell Lung Cancer',
                        'nsclc_mskcc_2015': 'Non-Small Cell Lung Cancer'}

        self.clinical_data['CANCER_TYPE'] = self.clinical_data['cohort'].apply(lambda x: cancer_types[x])

        def get_TMB_status(x):
            if x[1] >= cancer_specific_TMB_cuts[x[0]]:
                return "TMB-H"
            else:
                return "TMB-L"

        self.clinical_data['TMB'] = self.total_mut_counts['count'].reindex(self.clinical_data.index)

        #We take the upper half of samples by the total number of mutations
        cancer_specific_TMB_cuts = self.total_mut_counts.groupby('cohort')['count'].apply(
            lambda x: np.quantile(x, .51)).to_dict()
        self.total_mut_counts['high_TMB'] = self.total_mut_counts.loc[:, ['cohort', "count"]].apply(get_TMB_status, axis=1)
        self.clinical_data['high_TMB'] = self.total_mut_counts['high_TMB'].reindex(self.clinical_data.index)

        self.clinical_data = self.clinical_data.iloc[
                               np.where(~self.clinical_data['RECIST'].isin([np.nan, 'NE', "X"]))[0], :]

        self.clinical_data = self.clinical_data.loc[self.clinical_data.index.isin(self.spec_by_genes.index),:]
        self.spec_by_genes = self.spec_by_genes.loc[
                                 self.spec_by_genes.index.isin(self.clinical_data.index), :]
    
    def load_genomic_data(self):
        mutations_data = {}
        combined_mutations = []
        self.combined_all_mutations = pd.DataFrame()

        for dirs in os.listdir(self.data_dir):
            cdir = os.path.join(self.data_dir, dirs)
            if os.path.isdir(cdir):
                self.all_data_dirs.append(cdir)
                for datafile in os.listdir(cdir):
                    cdatafile = os.path.join(cdir, datafile)
                    if re.search("data_mutations", datafile):
                        logging.info("loading %s %s",str(dirs), str(datafile))
                        mutations_data[dirs] = mutations_data.get(dirs, []) + [cdatafile]
                        cmutations = pd.read_table(cdatafile)
                        cmutations['cohort'] = dirs
                        combined_mutations.append(cmutations['Hugo_Symbol'].unique())

                        if self.combined_all_mutations.shape[0] == 0:
                            self.combined_all_mutations = cmutations
                        else:
                            self.combined_all_mutations = pd.concat([self.combined_all_mutations, cmutations], axis=0,join='inner')

        self.combined_all_mutations['Tumor_Sample_Barcode'] = self.combined_all_mutations['Tumor_Sample_Barcode'].apply(
            lambda x: re.sub('-Tumor-SM-\w+', "", x))
        self.total_mut_counts = self.combined_all_mutations.groupby(['cohort', 'Tumor_Sample_Barcode']).size().reset_index()
        self.total_mut_counts['count'] = self.total_mut_counts[0]
        self.total_mut_counts.index = self.total_mut_counts['Tumor_Sample_Barcode']

        # filter down to unique mutations
        self.combined_all_mutations = self.combined_all_mutations.loc[self.combined_all_mutations.duplicated(
            subset=['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Chromosome',
                    'Start_Position', 'End_Position'], keep='first'), :]
        # common set of genes present in all.
        muts2keep = set.intersection(*[set(muts) for muts in combined_mutations])
        self.combined_all_mutations = self.combined_all_mutations.loc[self.combined_all_mutations['Hugo_Symbol'].isin(muts2keep), :]


        consequence_to_keep = ['nonsense_mutation',
                               'missense', 'missense_mutation',
                               'frameshift', 'nonsense',
                               'frame_shift_del', 'frame_shift_ins', 'translation_start_site', 'nonstop_mutation']
        orig_names = self.combined_all_mutations['Hugo_Symbol'].unique()
        new_names = match_names_to_symbols(orig_names)
        gene_name_map = dict(zip(orig_names, new_names))
        self.combined_all_mutations['Hugo_Symbol'] = self.combined_all_mutations['Hugo_Symbol'].apply(
            lambda x: gene_name_map.get(x, x))
        # filter based on consequence 
        self.combined_all_mutations['Variant_Classification'] = self.combined_all_mutations['Variant_Classification'].apply(
            lambda x: str(x).lower())
        # print(self.combined_all_mutations['Variant_Classification'].value_counts().index)

        # self.combined_all_mutations=self.combined_all_mutations.iloc[np.where(self.combined_all_mutations['Variant_Classification'].isin(consequence_to_keep))[0],:]
        # print('vars2keep', np.sum(self.combined_all_mutations['Variant_Classification'].isin(consequence_to_keep)))

        self.spec_by_genes = (
                    self.combined_all_mutations.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack(
                        fill_value=0) > 0).astype(int)
        # add back in samples that didn't have any variants meeting filters
        for samp in self.total_mut_counts.index[~self.total_mut_counts.index.isin(self.spec_by_genes.index)]:
            self.spec_by_genes.loc[samp, :] = 0

class Rose2021(ClinicalDataSet):

    def __init__(self):
        super(Rose2021, self).__init__()
        self.data_dir=rose_dir
        self.load_genomic_data()
        self.load_clinical_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_genomic_data(self):
        logging.info("Loading Rose et al genomic data")
        save_muts_file = os.path.join(self.data_dir, 'fgfr3_proj_all_snv.csv')
        #line up sample IDs with 
        conversation_table = os.path.join(self.data_dir, "BACI_Omniseq_Sample_Name_Key.xlsx")
        samp_cov_table = pd.read_excel(conversation_table)
        sample_cov_dict = {}
        sample_rnaseq_cov_dict = {}
        for ind in samp_cov_table.index:
            csamp_id = samp_cov_table.loc[ind, 'Omniseq_BAC01_ID']
            csamp_id2 = samp_cov_table.loc[ind, 'Sample ID']
            if str(csamp_id) != 'nan':
                for val in csamp_id.split(','):
                    sample_cov_dict[val] = csamp_id2

        for ind in samp_cov_table.index:
            csamp_id = samp_cov_table.loc[ind, 'Omniseq_RS_ID (RNAseq)']
            csamp_id2 = samp_cov_table.loc[ind, 'Sample ID']
            if str(csamp_id) != 'nan':
                for val in csamp_id.split(','):
                    sample_rnaseq_cov_dict[val] = csamp_id2

        self.all_mutations = pd.read_csv(save_muts_file, index_col=0)
        self.all_mutations['Sample'] = list(map(lambda x: sample_cov_dict[x], self.all_mutations['Sample']))
        self.all_mutations['Tumor_Sample_Barcode']=self.all_mutations['Sample']
        # cnv_df['Sample']=list(map(lambda x: sample_cov_dict[x],cnv_df['Sample']))
        self.all_mutations = self.all_mutations.iloc[np.where(self.all_mutations['Qual'] > 100)[0], :]
        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Gene']
        orig_names = self.all_mutations['Hugo_Symbol'].unique()
        new_names = match_names_to_symbols(orig_names)
        gene_name_map = dict(zip(orig_names, new_names))
        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Hugo_Symbol'].apply(
            lambda x: gene_name_map.get(x, x))

        self.all_mutations = self.all_mutations.iloc[np.where(self.all_mutations['impact'].isin(['HIGH', "MODERATE", "LOW"]))[0], :]
        clin_var2remove = ['Benign', "Benign/Likely benign", "Benign, other", 'Likely benign']
        snv_type2keep = ['non-synonymous']
        self.all_mutations = self.all_mutations.iloc[np.where(np.logical_not(self.all_mutations['ClinVar Significance'].isin(clin_var2remove)))[0], :]
        self.all_mutations = self.all_mutations.iloc[np.where(self.all_mutations['Substitution Type'].isin(snv_type2keep))[0], :]

        self.spec_by_genes = (self.all_mutations.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack(fill_value=0) > 0).astype(int)

    def load_clinical_data(self,tmb_thresh=10):
        logging.info("Loading Rose et al clincal data")
        self.clinical_data = pd.read_csv(os.path.join(self.data_dir, 'baci.n109.csv'), index_col=0)
        self.clinical_data['high_TMB'] = self.clinical_data['TMB'].apply(lambda x: 'TMB-H' if x > tmb_thresh else 'TMB-L')
        self.clinical_data['CANCER_TYPE'] = 'Bladder Cancer'

        self.clinical_data['binaryResponse'] = self.clinical_data['IO.response'].apply(
            lambda x: 'CR/PR' if x in ['CR', 'PR'] else 'SD/PD')
        self.clinical_data['os'] = self.clinical_data['OS'] / 30.0
        self.clinical_data['censOS'] = self.clinical_data['Alive'].apply(lambda x: 1 if x == 'No' else 0)
        self.clinical_data['cohort'] = 'baci'
        self.clinical_data['TREATMENT'] = self.clinical_data['IO.therapy']
        self.clinical_data['RECIST'] = self.clinical_data['IO.response']
        self.clinical_data = self.clinical_data.iloc[np.where(np.logical_not(self.clinical_data['TMB'].isna()))[0], :]
        self.clinical_data = self.clinical_data.iloc[
                             np.where(~self.clinical_data['RECIST'].isin([np.nan, 'NE', "X"]))[0], :]

        self.clinical_data['cohort'] = 'Rose2021'
        self.spec_by_genes=self.spec_by_genes.loc[self.spec_by_genes.index.intersection(self.clinical_data.index),:]
class WeirMetaDataSet(ClinicalDataSet):
    """
    Combine Cbioportal and Rose et al for this dataset
    """
    def __init__(self):
        super(WeirMetaDataSet,self).__init__()
        self.cbioportal=CBioportalCombined()
        self.rose=Rose2021()
        self.spec_by_genes=pd.concat([self.cbioportal.spec_by_genes, self.rose.spec_by_genes], join='outer', sort=True)
        self.clinical_data = pd.concat([self.cbioportal.clinical_data, self.rose.clinical_data], join='inner', sort=True)
        #some redudancy here.
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df,drop_intersection=False)


class Braun2020Dataset(ClinicalDataSet):

    def __init__(self):
        super(Braun2020Dataset, self).__init__()
        self.data_dir=braun_dir
        self.datafile = os.path.join(self.data_dir, '41591_2020_839_MOESM2_ESM.xlsx')
        self.load_genomic_data()
        self.load_clinical_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_clinical_data(self):
        logger.info("loading Braun clinical data")
        self.clinical_data = pd.read_excel(self.datafile, sheet_name="S1_Clinical_and_Immune_Data", skiprows=[0])
        self.clinical_data = self.clinical_data.dropna(subset=['MAF_Tumor_ID'])
        self.clinical_data['TMB'] = self.clinical_data['TMB_Counts']
        thresh=100
        self.clinical_data['high_TMB'] = self.clinical_data['TMB'].apply(lambda x: 'TMB-H' if x > thresh else 'TMB-L')
        # THE TYPES are CB ICB NCB.  Not sure how to group these. 
        self.clinical_data['binaryResponse'] = self.clinical_data['ORR'].apply(
            lambda x: 'CR/PR' if x in ['CR', 'CRPR', 'PR'] else 'SD/PD')
        self.clinical_data['os'] = self.clinical_data['OS']
        self.clinical_data['censOS'] = self.clinical_data['OS_CNSR']
        self.clinical_data.index = self.clinical_data['MAF_Tumor_ID']
        cols2keep = ['os', 'censOS', 'MAF_Tumor_ID', 'TMB', 'high_TMB', 'ORR',
                     'binaryResponse', 'MSKCC', 'Sex', 'Age', 'Arm', 'Cohort', 'SUBJID',
                     'Tumor_Sample_Primary_or_Metastasis', 'PFS', 'PFS_CNSR', 'Number_of_Prior_Therapies']
        self.clinical_data = self.clinical_data.loc[:, cols2keep]
        self.clinical_data['cohort'] = 'Braun2020'

    def load_genomic_data(self):
        logger.info("loading Braun genomic data")
        self.all_mutations = pd.read_excel(self.datafile, skiprows=[0],
                                      sheet_name="S2_WES_Data", )

        consequence_to_keep = ['missense_mutation', 'nonsense_mutation'
            , 'frameshift', 'nonsense', 'de_novo_start_outofframe',
                               'start_codon_del', 'start_codon_ins',
                               'frame_shift_del', 'frame_shift_ins', 'translation_start_site', 'nonstop_mutation']
        #since this is WES this takes to long so we store it
        mapfile=os.path.join(self.data_dir,'braun_names_map.pkl')
        if not os.path.exists(mapfile):
            orig_names = self.all_mutations['Hugo_Symbol'].unique()
            new_names = match_names_to_symbols(orig_names)
            gene_name_map = dict(zip(orig_names, new_names))
            with gzip.open(mapfile,'wb') as fh:
                pickle.dump(gene_name_map,fh)
        else:
            with gzip.open(mapfile,'rb') as fh:
                gene_name_map=pickle.load(fh)

        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Hugo_Symbol'].apply(lambda x: gene_name_map.get(x, x))
        # filter based on consequence
        self.all_mutations['Variant_Classification'] = self.all_mutations['Variant_Classification'].apply(
            lambda x: str(x).lower())
        print('vars2keep', np.sum(self.all_mutations['Variant_Classification'].isin(consequence_to_keep)))
        self.all_mutations = self.all_mutations.iloc[
                        np.where(self.all_mutations['Variant_Classification'].isin(consequence_to_keep))[0], :]
        self.spec_by_genes = (self.all_mutations.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack(
            fill_value=0) > 0).astype(int)


class Miao2018Dataset(ClinicalDataSet):

    def __init__(self):
        super(Miao2018Dataset, self).__init__()
        self.data_dir=miao_dir
        self.load_genomic_data()
        self.load_clinical_data()
        self.add_all_big_bets_categories(ClinicalDataSet.bigbet_scores_df)

    def load_genomic_data(self):
        miao_data_file = os.path.join(self.data_dir, '41588_2018_200_MOESM6_ESM.txt')
        quality_file = os.path.join(self.data_dir, '41588_2018_200_MOESM3_ESM.csv')
        self.seq_qual_df = pd.read_csv(quality_file, skiprows=28)
        self.seq_qual_df.index = self.seq_qual_df['individual_id']
        self.all_mutations = pd.read_table(miao_data_file)

        orig_names = self.all_mutations['Hugo_Symbol'].unique()
        new_names = match_names_to_symbols(orig_names)
        gene_name_map = dict(zip(orig_names, new_names))
        self.all_mutations['Hugo_Symbol'] = self.all_mutations['Hugo_Symbol'].apply(
            lambda x: gene_name_map.get(x, x))

        # we have to compute TMB values before filtering mutations
        self.total_muts = self.all_mutations['pair_id'].value_counts()
        self.all_mutations['Variant_Classification'] = self.all_mutations['Variant_Classification'].apply(
            lambda x: str(x).lower())
        # logger.info(self.all_mutations['Variant_Classification'].value_counts().index)
        # all_mutations=all_mutations.iloc[np.where(all_mutations['Variant_Classification'].isin(consequence_to_keep))[0],:]

        self.spec_by_genes = (self.all_mutations.groupby(['pair_id', 'Hugo_Symbol']).size().unstack(fill_value=0) > 0).astype(int)


    def load_clinical_data(self):
        self.clinical_data_file = os.path.join(miao_dir, '41588_2018_200_MOESM4_ESM.csv')
        self.clinical_data = pd.read_csv(self.clinical_data_file)
        self.clinical_data.index = self.clinical_data['pair_id']

        def find_substr(val, a):
            val = np.where(list(map(lambda x: re.search(val, x), a)))[0]
            if len(val) > 0:
                return val[0]
            else:
                return -1

        pair_ids = self.all_mutations['pair_id'].unique()

        matches = self.seq_qual_df['individual_id'].apply(lambda x: find_substr(x, pair_ids))
        ids2keep = self.seq_qual_df['individual_id'][matches != -1]
        matches = matches[matches != -1]
        idmap = dict((cid, pair_ids[matches.iloc[i]]) for i, cid in enumerate(ids2keep))
        revidmap = dict((v, k) for k, v in idmap.items())

        self.clinical_data['total_cnts'] = self.total_muts[self.clinical_data.index]
        self.clinical_data['individual_id'] = self.clinical_data['pair_id'].apply(lambda x: revidmap.get(x, 'None'))
        self.clinical_data['bases_covered'] = self.seq_qual_df['somatic_mutation_covered_bases_capture'].reindex(
            self.clinical_data['individual_id']).values
        self.clinical_data['TMB'] = self.clinical_data['total_cnts'] / (self.clinical_data['bases_covered'] / 1000000)

        # one sample was missing total bases covered so we just imput this
        temp = self.clinical_data.loc[:, ['total_cnts', 'TMB']].dropna(axis=0, how='any')
        slope, inter, rval, pval, stderr = stats.linregress(temp)
        # getting missing indices and use mx+b to "predict" tmb
        missing_inds = np.where(self.clinical_data['TMB'].isna())[0]
        self.clinical_data.iloc[missing_inds, self.clinical_data.columns.get_loc('TMB')] = self.clinical_data['total_cnts'].iloc[
            missing_inds].apply(lambda x: slope * x + inter)

        # def get_TMB_status(x):
        #     if x[1]>=cancer_specific_TMB_cuts[x[0]]:
        #         return "TMB-H"
        #     else:
        #         return "TMB-L"
        # cancer_specific_TMB_cuts=self.clinical_data.groupby('cancer_type')['TMB'].apply(lambda x: np.quantile(x,.75)).to_dict()
        # self.clinical_data['high_TMB']=self.clinical_data.loc[:,['cancer_type',"TMB"]].apply(get_TMB_status,axis=1)

        self.clinical_data['high_TMB'] = self.clinical_data['TMB'].apply(lambda x: 'TMB-H' if x > 10 else 'TMB-L')

        # THERE ARE entried labled NE???  What to do with these?
        self.clinical_data['binaryResponse'] = self.clinical_data['RECIST'].apply(
            lambda x: 'CR/PR' if x in ['CR', 'PR'] else 'SD/PD')
        self.clinical_data['os'] = self.clinical_data['os_days'] / 30.
        self.clinical_data['censOS'] = self.clinical_data['os_censor']