import numpy as np
import statsmodels.api as sm
from scipy import interp
import scipy.stats as stats
import pandas as pd
import lifelines
import matplotlib.pyplot as plt

def get_excess_degree(df):
    outvals=pd.DataFrame(columns=['degree','excess_degree'])

    for i in range(df.shape[0]):
        cinds=np.where(df.iloc[i,:]>0)[0]
        c_exc=np.sum(np.sum(df.iloc[cinds,:]))/len(cinds) #average degree of neighbors
        outvals.loc[outvals.shape[0],['degree','excess_degree']]=len(cinds),c_exc

    return outvals

def get_excess_degree_sparse(csr_mat,samples,features):
    outvals=pd.DataFrame(columns=['degree','excess_degree'])

    for i in range(csr_mat.shape[0]):
        # cinds=np.where((csr_mat[i,:]>0).data)[0]
        cinds=csr_mat[i, :].rows[0]
        c_exc=np.sum(np.sum(csr_mat[cinds,:]))/len(cinds) #average degree of neighbors
        outvals.loc[outvals.shape[0],['degree','excess_degree']]=len(cinds),c_exc

    return outvals

def get_ecdf_tmb_df(genes,df,snv_indel_df):
    cinds = list(set(df.index[np.where(df.loc[:, df.columns.intersection(genes)] > 0)[0]]))
    tmbs = np.log1p(snv_indel_df.loc[cinds, 'tmb']).dropna()
    try:
        cdf = sm.distributions.ECDF(tmbs)
    except:
        print(genes)
        print('tmbs',tmbs)
        raise AssertionError
    x = sorted(tmbs)
    return x, cdf(x)


def get_ecdf_tmb(genes, mat,samps,feats, snv_indel_df):
    """Trimmed down method for use on low memory matrix (used
        while running bipartite configuration sampling)"""
    c_cols=np.where(np.isin(feats,genes))[0] #columns to keep from matrix
    # cinds = list(set(samps[np.where((mat[:, c_cols] > 0).data)[0]]))
    cinds=samps[list(set(mat[:,c_cols].nonzero()[0]))]
    # tmbs = np.log1p(snv_indel_df.loc[cinds, 'tmb']).dropna()
    tmbs = np.log1p(snv_indel_df.loc[cinds, 'tmb']).dropna()

    try:
        cdf = sm.distributions.ECDF(tmbs)
    except:
        print(genes)
        print('tmbs', tmbs)
        raise AssertionError
    x = sorted(tmbs)
    return x, cdf(x)

# def get_ecdf_tmb(genes, df, snv_indel_df):
#     cinds = list(set(df.index[np.where(df.loc[:, genes] > 0)[0]]))
#     tmbs = np.log1p(snv_indel_df.loc[cinds, 'tmb']).dropna()
#     try:
#         cdf = sm.distributions.ECDF(tmbs)
#     except:
#         print(genes)
#         print('tmbs', tmbs)
#         raise AssertionError
#     x = sorted(tmbs)
#     return x, cdf(x)

def get_ecdf_upper_lower(all_x, all_ecdf):
    x_cat = np.unique(np.concatenate(all_x))

    all_interp = np.zeros((len(x_cat), len(all_ecdf)))
    for j in range(len(all_ecdf)):
        all_interp[:, j] = interp(x_cat, all_x[j], all_ecdf[j])

    #     print(all_interp.shape)
    mean_ecdf = np.mean(all_interp, axis=1)
    std_ecdf = np.std(all_interp, axis=1)
    ecdf_upper = np.percentile(all_interp, q=97.5, axis=1)
    ecdf_lower = np.percentile(all_interp, q=.05, axis=1)

    #     ecdf_upper = np.minimum(mean_ecdf + std_ecdf, 1)
    #     ecdf_lower = np.maximum(mean_ecdf - std_ecdf, 0)
    return x_cat, mean_ecdf, ecdf_lower, ecdf_upper

def get_tmb_dist_genes(mat,samps,feats, snv_indels_df):
    gene2tmb = {}
    #     for gene in gene_2_path_dict.keys():
    for gene in feats:  # collect all genes
        #         if gene in df.columns:
        gene2tmb[gene] = {}
        x, cdf = get_ecdf_tmb(genes=[gene], mat=mat,samps=samps,feats=feats,
                              snv_indel_df=snv_indels_df)
        gene2tmb[gene]['x'] = x
        gene2tmb[gene]['cdf'] = cdf
    return gene2tmb


def get_tmb_dist_paths(mat,samps,feats, snv_indels_df, path2genedict):
    path2tmb = {}
    path2rm = []
    # path2rm = ['BER']  # only a few mutations in this one
    for path, genes in path2genedict.items():
        if path in path2rm:
            continue
        if not np.any(np.isin(genes,feats)):
            continue #
        path2tmb[path] = {}
        x, cdf = get_ecdf_tmb(genes=genes,mat=mat,samps=samps,feats=feats,
                              snv_indel_df=snv_indels_df)
        path2tmb[path]['x'] = x
        path2tmb[path]['cdf'] = cdf
    return path2tmb




def get_tmb_dist_both_path_and_genes(mat,samps,feats, snv_indels_df, path2genedict):
    path2tmbs = get_tmb_dist_paths(mat,samps,feats, snv_indels_df, path2genedict)
    gene2tmb = get_tmb_dist_genes(mat,samps,feats, snv_indels_df)
    return {'paths': path2tmbs, 'genes': gene2tmb}



def compile_all_runs(all_runs_data):
    def collect_row(x):
        x_s = []
        cdfs = []
        _ = list(map(lambda y: (x_s.append(y['x']), cdfs.append(y['cdf'])), x))
        return (x_s, cdfs)

    gene2_allecdf = all_runs_data.apply(collect_row, axis=1)
    return gene2_allecdf


def compile_all_runs_both_genes_paths(all_runs_data):
    def collect_row(x, tokeep='genes'):
        x_s = []
        cdfs = []
        print(x)
        _ = list(map(lambda y: (x_s.append(y[tokeep]['x']), cdfs.append(y[tokeep]['cdf'])), x))
        return (x_s, cdfs)

    gene2_allecdf = all_runs_data.apply(collect_row, axis=1)
    paths2_allecdf = all_runs_data.apply(lambda x: collect_row(x, 'paths', axis=1))

    return gene2_allecdf, paths2_allecdf


def get_os_stats(clust, samp_df,os='os',censOS='censOS'):
    df2test = samp_df.loc[:, [os, censOS]]
    df2test['clust'] = clust
    df2test = pd.get_dummies(df2test, columns=['clust'])
    df2test = df2test.iloc[:, :-1]
    cph = lifelines.CoxPHFitter()
    cph.fit(df2test, os, event_col=censOS)
    llrstat = cph.log_likelihood_ratio_test()
    pval = llrstat.p_value

    res = lifelines.statistics.multivariate_logrank_test(df2test[os],
                                                         df2test[censOS],
                                                         clust)

    logrank = res.test_statistic
    logrankpval = res.p_value
    outdf = pd.DataFrame({"llr": [llrstat.test_statistic], 'DF': [llrstat.degrees_freedom],
                          'llr_pval': [llrstat.p_value],
                          'log_rank': [logrank], "logrank_pval": [logrankpval]})

    return outdf


def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height])
    subax
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    subax.patch.set_facecolor(axisbg)
    return subax

def get_percents_bar_by_group(df,groups,var):
    outdf=pd.DataFrame(columns=['Response','group','count','percent'])
    varvals=df[var].unique()
    groups=df.groupby(groups)
    group_names=list(groups.groups.keys())
    for k in group_names:
        if type(k)==str:
            outk=k
        else:
            outk="_".join([str(x) for x in k])
        ccnts=groups[var].value_counts()[k]
        for ind in ccnts.index:
            cind=outdf.shape[0]
            outdf.loc[cind,['Response','group','count','percent']]=[ind,outk,ccnts[ind],ccnts[ind]/np.sum(ccnts)]
    return outdf

def cal_mwu_pval(genes,df,snv_indel_df,outgroup,att='tmb'):
    cinds=list(set(np.where(df.loc[:,df.columns.intersection(genes)]>0)[0]))
    cspecs=df.index[cinds]
    ctmbs=snv_indel_df.loc[cspecs,att]
    altertmbs=snv_indel_df.loc[outgroup,att]
    mwu,pval=stats.mannwhitneyu(x=ctmbs,y=altertmbs,alternative='greater')
    return  mwu,pval

def get_tmb_values_genes(genes,df,snv_indel_df,att='tmb'):
    "return the TMB values for a group of genes given mutations df and tmb df"
    cinds=list(set(np.where(df.loc[:,df.columns.intersection(genes)]>0)[0]))
    cspecs=df.index[cinds]
    ctmbs=snv_indel_df.loc[cspecs,att]
    return ctmbs

def get_gene_signature(df,sig_genes):
    sig_means=np.mean(df.loc[:,sig_genes],axis=1)
    sig_means=(sig_means-np.median(sig_means))/np.std(sig_means)
    return sig_means