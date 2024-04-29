#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Using FastSLAM and the UTIAS Multi-Robot Cooperative Localization and Mapping to test Monte Carlo methods
to learn the probabilistic motion and measurement model of the ground robots present in the dataset

Author: Mahdi Chaari
Email: mchaari@unistra.fr
'''
#%%
from Fast_SLAM_1_Particles import fastSLAM_FK, fastSLAM_SMC, fastSLAM_SMC2, fastSLAM_SSM

import particles
from particles import distributions as dists  # where probability distributions are defined
from particles import mcmc
from particles.collectors import Moments

import matplotlib.pyplot as plt
import seaborn as sb


#%%
if __name__ == '__main__':
    my_model = fastSLAM_SSM()
    fk_model = fastSLAM_FK(ssm=my_model)
    pf = fastSLAM_SMC(fk = fk_model, n_proc= 1,
                  collect=[Moments()], store_history=True, verbose=False)  # the algorithm)
#%%
    # run the particle filter
    pf.next()
    pf.run()

#%%
    prior_dict = {  'sigma_v': dists.Uniform(a=0., b=0.5), 
                    'sigma_w': dists.Uniform(a=0., b=0.5),
                    'gamma': dists.Uniform(a=0., b=0.5), 
                    'sigma_range': dists.Uniform(a=0., b=0.5), 
                    'sigma_bearing': dists.Uniform(a=0., b=0.5),
                    }
    
    my_prior = dists.StructDist(prior_dict)

#%%
    # pmmh = mcmc.PMMH(ssm_cls=fastSLAM_SSM, smc_cls= fastSLAM_SMC, fk_cls= fastSLAM_FK, prior=my_prior, data=None, Nx=100, niter = 1000, verbose= 1)
    # pmmh.run()  # Warning: takes a few seconds

#%%
    fk_smc2 = fastSLAM_SMC2(ssm_cls=fastSLAM_SSM, fk_cls = fastSLAM_FK, data=None, prior=my_prior,init_Nx=100,
                   ar_to_increase_Nx=0.1, len_chain= 10)
    results_smc2 = particles.multiSMC(fk=fk_smc2, N=50, nruns=15, verbose = True)
    #plt.figure()
    #sb.boxplot(x=[r['output'].logLt for r in results_smc2], y=[r['qmc'] for r in results_smc2]);

#%% 

    fk_smc2 = fastSLAM_SMC2(ssm_cls=fastSLAM_SSM, fk_cls = fastSLAM_FK, data=None, prior=my_prior,init_Nx=200,
                   ar_to_increase_Nx=-0.1, len_chain= 10)
    alg_smc2 = particles.SMC(fk=fk_smc2, N=50, verbose= True)
    alg_smc2.run()

#%%
    i = 0
    another_theta = my_prior.rvs(size=1000)
    for p in prior_dict.keys():
        plt.subplot(2, 5, i + 1)
        # plt.ylim((0, 0.8))
        # plt.xlim((0, 6))
        sb.distplot(alg_smc2.X.theta[p], 40)
        sb.distplot(another_theta[p], 40)
        plt.xlabel(p)
        i += 1
    plt.show()

# %%
