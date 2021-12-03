import mygene
import pandas as pd
import numpy as np

def match_names_to_symbols(genenames):

    mg = mygene.MyGeneInfo()
    ressyms = mg.querymany(genenames, scopes='symbol', fields='symbol,entrezgene',
                           species='human', returnall=True)
    cnt=0
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
    #or you could try lining up with entrez id but this seems to work decently well.
    my_res_df = my_res_df.loc[my_res_df.groupby('query')['_score'].idxmax().dropna(), :]
    my_res_df.index = my_res_df['query']

    #map gene names to symbols found
    return list(map(lambda x: my_res_df['symbol'].get(x, x),genenames))
