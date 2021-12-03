import os,sys,re
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import logging
from time import time
from contextlib import contextmanager
from multiprocessing import Pool
import tqdm


LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
#LOG_LEVEL = logging.ERROR
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
logging.basicConfig(format = LOG_FORMAT,level = LOG_LEVEL)

#context manager for openning and closing parallel processes
@contextmanager
def terminating(obj):
	'''
	Context manager to handle appropriate shutdown of processes
	:param obj: obj to open
	:return:
	'''
	try:
		yield obj
	finally:
		obj.terminate()


#call object in parallel loop
def parallel_worker(args):
    np.random.seed()
    obj = args[0]
    res = obj.collect_data(*args[1:])
    return res


class ConfigurationRewiring(object):

    def __init__(self,array,directed=False):
        logging.debug("Command: %s", " ".join(sys.argv))
        self.sample_rates = np.sum(array, axis=1)
        self.directed=directed
        is_pandas = hasattr(array,'index') # is df?

        if is_pandas:
            self.samples = array.index.values
            self.features = array.index.values #same for both here
            self.array = sparse.lil_matrix(array.values)

        else:
            self.samples = np.arange(array.shape[0])
            self.features = self.samples
            self.array = sparse.lil_matrix(array)

        # self.array = pd.DataFrame(array, index=self.samples, columns=self.samples)
        #make data frame spare
        # self.array = self.array.to_sparse(fill_value=0)

        # if self.keep_rewired:
        #     self.rewired_arrays = []
        # else:
        #     self.rewired_arrays = None
        
    def rewire(self,cur_array,single_edges=True):
        """Select four different samples and attempt to swap an edge.

        Will return -1 if attempt to swap fails

        :param cur_array: array to perform rewiring on
        :return:
        """
        samp1,samp2=np.random.choice(self.samples,replace=False,size=2)

        s1inds=cur_array.columns[np.where(cur_array.loc[samp1,:]!=0)[0]]
        s2inds=cur_array.columns[np.where(cur_array.loc[samp2,:]!=0)[0]]

        if len(s1inds)==0 or len(s2inds)==0:
            # logging.debug("Rewire failed from selection of node degree zero")
            return cur_array,-1

        samp3=np.random.choice(s1inds,size=1)[0]
        samp4=np.random.choice(s2inds,size=1)[0]
        #We constrict here for single edges
        if single_edges:
            if samp3 == samp4:
                # logging.debug("Rewire failed from single edge restriction: {:d} == {:d} ".format(loc1,loc2))
                # logging.debug("{} --- {}".format(str(s1inds),str(s2inds)))
                return cur_array,-1
            if samp3 in s2inds or samp4 in s1inds: #these
                return cur_array,-1
        #swap edge between samp1-sampe3 and sampe2-sample4
        cur_array.loc[samp1,samp3]-=1
        cur_array.loc[samp1,samp4]+=1
        cur_array.loc[samp2, samp3] += 1
        cur_array.loc[samp2, samp4] -= 1
        
        if not self.directed: #keep track of lower triangle of adjacency
            cur_array.loc[samp3,samp1]-=1
            cur_array.loc[samp4,samp1]+=1
            cur_array.loc[samp3, samp2] += 1
            cur_array.loc[samp4, samp2] -= 1

        # assert np.all(np.sum(cur_array,axis=0) == old_sum_0), "sum over axis 0 has changed"
        # assert np.all(np.sum(cur_array,axis=1) == old_sum_1), "sum over axis 1 has changed"

        return cur_array,0

    def perform_rewires(self,cur_array=None,num_rewires=100,num_fails_allowed=100,single_edges=True):
        """

        :param rewire_succeed:
        :param num_fails_allowed:
        :param single_edges:
        :return:
        """
        rewire_succeed=0
        rewire_fail=0
        if cur_array is None:
            next_array=self.array.copy()
        else:
            next_array=cur_array.copy()

        t0=time()
        while rewire_succeed < num_rewires and rewire_fail<num_fails_allowed:
            next_array,status=self.rewire(next_array,single_edges)

            if status==0:
                rewire_succeed+=1
            if status==-1:
                rewire_fail+=1
        logging.debug("success {:d} failed: {:d} time: {:.3f}.".format(rewire_succeed,rewire_fail,time()-t0))
        return next_array,rewire_succeed

    def collect_data(self,function2apply,burninwires=1000,num_samples=100,rewirespersample=100,
                     num_fails_allowed=100,single_edges=True):
        """
        :param num_samples:
        :param rewirespersample:
        :param num_fails_allowed:
        :param single_edges:
        :return:
        """
        collected_data={}
        cur_array=self.array
        cur_rewires=0

        #burn in
        logging.debug("Performing burn in rewires.")
        while cur_rewires<burninwires:
            cur_array,num_rewires=self.perform_rewires(cur_array=cur_array,
                                 num_rewires=burninwires,
                                 num_fails_allowed=num_fails_allowed,single_edges=single_edges)
            cur_rewires+=num_rewires
        logging.debug("Finished burn in")
        logging.debug("Collecting samples at {:d} rewire".format(rewirespersample))
        t0=time()
        for i in range(num_samples):
            if i%5==0:
                logging.debug("Collected {:d} samples successfully in {:.3f}".format(i,time()-t0))
            cur_array,num_rewires=self.perform_rewires(cur_array=cur_array,
                                 num_rewires=rewirespersample,
                                 num_fails_allowed=num_fails_allowed,single_edges=single_edges)
            collected_data[i]=function2apply(cur_array,self.samples,self.features)
            # sdf=pd.SparseDataFrame(cur_array,index=self.samples,columns=self.features)
            # collected_data[i]=function2apply(sdf)
        logging.debug("Time for {:d} samples: {:.3f}".format(num_samples,time()-t0))
        return collected_data


class BipartiteMatching(ConfigurationRewiring):


    def __init__(self,array,samples=None,features=None,sample_classes=None):
        logging.debug("Command: %s", " ".join(sys.argv))
        is_pandas = hasattr(array,'index') # is df?

        self.array=array #binary array of connections.  This is observed
        self.sample_rates=np.sum(array,axis=1)
        self.feature_rates=np.sum(array,axis=0)
        self.sample_classes=sample_classes

        if samples is None:
            if hasattr(array,"index"):
                self.samples=array.index.values
            else:
                self.samples=np.arange(array.shape[0])
        else:
            self.samples=samples
        if features is None:
            if hasattr(array,"columns"):
                self.features=array.columns.values
            else:
                self.features=np.arange(array.shape[1])
        else:
            self.features=features

        if is_pandas:
            # self.array=sparse.csr_matrix(array.values)
            self.array=sparse.lil_matrix(array.values)

        else:
            self.array=sparse.lil_matrix(array)
            # self.array=sparse.csr_matrix(array)

        # self.array=pd.DataFrame(array,index=self.samples,columns=self.features)
        # self.array = self.array.to_sparse(fill_value=0)

    def rewire(self,cur_array,single_edges=True):
        """Select two (different) features and two different samples and attempt to swap an edge.

        Will return -1 if attempt to swap fails


        :param cur_array: array to perform rewiring on
        :return:
        """

        if self.sample_classes is not None:
            class2switch = np.random.choice(self.sample_classes, replace=True, size=1)[0]

            poss_inds = np.where(self.sample_classes == class2switch)[0]
            if len(poss_inds) < 2:  # must have two samples in a given class
                return cur_array, -1

            samp1, samp2 = np.random.choice(poss_inds, replace=False, size=2)
        else:
            samp1, samp2 = np.random.choice(np.arange(len(self.samples)), replace=False, size=2)

        s1inds = cur_array[samp1, :].nonzero()[1]
        s2inds = cur_array[samp2, :].nonzero()[1]


        s1inds=cur_array[samp1,:].nonzero()[1]
        s2inds=cur_array[samp2,:].nonzero()[1]

        # s1inds=cur_array.columns[np.where(cur_array.loc[samp1,:]!=0)[0]]
        # s2inds=cur_array.columns[np.where(cur_array.loc[samp2,:]!=0)[0]]
        if len(s1inds)==0 or len(s2inds)==0:
            logging.debug("Rewire failed from selection of the same samples")
            return cur_array,-1

        loc1=np.random.choice(s1inds,size=1)[0]
        loc2=np.random.choice(s2inds,size=1)[0]

        #We only allow single edges in rewired graph (no multiedges)
        if single_edges:
            if loc1 == loc2:
                # logging.debug("Rewire failed from single edge restriction: {:} == {:} ".format(str(loc1),str(loc2)))
                # logging.debug("{} --- {}".format(str(s1inds),str(s2inds)))
                return cur_array,-1
            if loc1 in s2inds or loc2 in s1inds: #connected to same gene
                return cur_array,-1

        #swap edge between samp1-feat1 and sampe2-feat2
        # cur_array.loc[samp1,loc1]-=1
        # cur_array.loc[samp1,loc2]+=1
        # cur_array.loc[samp2, loc1] += 1
        # cur_array.loc[samp2, loc2] -= 1
        cur_array[samp1, loc1] -= 1
        cur_array[samp1, loc2] += 1
        cur_array[samp2, loc1] += 1
        cur_array[samp2, loc2] -= 1

        # assert np.all(np.sum(cur_array,axis=0) == old_sum_0), "sum over axis 0 has changed"
        # assert np.all(np.sum(cur_array,axis=1) == old_sum_1), "sum over axis 1 has changed"

        return cur_array,0


class ParallelRewiringChains(object):

    def __init__(self,array,nsamples,nprocesses,func2call,samples=None,features=None,
                 sample_classes=None,
                 burninwires=None,nrewires_per_sample=None):

        is_pandas = hasattr(array, 'index')  # is df?

        self.sample_rates = np.sum(array, axis=1)
        self.feature_rates = np.sum(array, axis=0)

        self.sample_classes=sample_classes
        if samples is None:
            if hasattr(array, "index"):
                self.samples = array.index.values
            else:
                self.samples = np.arange(array.shape[0])
        else:
            self.samples = samples
        if features is None:
            if hasattr(array, "columns"):
                self.features = array.columns.values
            else:
                self.features = np.arange(array.shape[1])
        else:
            self.features = features


        if is_pandas:
            # self.array=sparse.csr_matrix(array.values)
            self.array = sparse.lil_matrix(array.values)
        else:
            self.array = sparse.lil_matrix(array)

        self.nprocesses = nprocesses

        #number of chains to start
        self.chains=[ BipartiteMatching(self.array,samples=samples,features=features,
                                        sample_classes=self.sample_classes) for _ in range(self.nprocesses) ]

        #divide up remainder evenly across chains
        self.samples_per_chain=[ int(nsamples/self.nprocesses) + int(i<nsamples%self.nprocesses) \
                                 for i in range(nprocesses)]

        self.totaledges=np.sum(np.sum(array))
        if burninwires is None:
            self.burnin = int(1.5*self.totaledges)
        else:
            self.burnin=burninwires

        self.rewirespersample=nrewires_per_sample
        if self.rewirespersample is None:
            self.rewirespersample=int(self.totaledges)
        self.allowed_fail=max(int(self.rewirespersample/5),10)
        self.func2call=func2call

    def run_allchains(self):
        #create arg to send to parallel function
        args=[ (chain,self.func2call,
                self.burnin,self.samples_per_chain[i],
                self.rewirespersample,self.allowed_fail,True) for i,chain in enumerate(self.chains) ]


        # self.all_res=list(map(parallel_worker,args)) #for testing no parallel
        self.all_res=[]
        with terminating(Pool(processes=self.nprocesses)) as pool:
            tot = len(args)
            with tqdm.tqdm(total=tot) as pbar:
                for i, res in tqdm.tqdm(enumerate(pool.imap(parallel_worker, args)), miniters=tot):
                    self.all_res.append(res)
                    pbar.update()

        #compile all of the trails together
        #we assume that the output of func2call is dict with each trial number
        #being a key
        compiled_res={}
        offset=0
        for i,vals in enumerate(self.all_res):
            for k,val in vals.items():
                compiled_res[k+offset]=val
            offset+=len(vals)
        self.all_res=compiled_res
