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
    pf = fastSLAM_SMC(fk = fk_model, resampling='stratified', n_proc= 1,
                   collect=[Moments()], store_history=True, verbose=True)  # the algorithm)
    # run the particle filter
    pf.run()

#%%
    prior_dict = {'sigma_x':dists.Uniform(a=0., b=1),
                    'sigma_y':dists.Uniform(a=0., b=1),
                    'sigma_theta':dists.Uniform(a=0., b=1),
                    'sigma_v': dists.Uniform(a=0., b=1), 
                    'sigma_w': dists.Uniform(a=0., b=1),
                    'gamma': dists.Uniform(a=0., b=1), 
                    'sigma_range': dists.Uniform(a=0., b=1), 
                    'sigma_bearing': dists.Uniform(a=0., b=1),
                    'bias_v' : dists.Normal(scale=0.05),
                    'bias_w' : dists.Normal(scale=0.05),}
    
    my_prior = dists.StructDist(prior_dict)

#%%
    pmmh = mcmc.PMMH(ssm_cls=fastSLAM_SSM, smc_cls= fastSLAM_SMC, fk_cls= fastSLAM_FK, prior=my_prior, data=None, Nx=100, niter = 1000, verbose= 1)
    pmmh.run()  # Warning: takes a few seconds
#%% 

    fk_smc2 = fastSLAM_SMC2(ssm_cls=fastSLAM_SSM, fk_cls = fastSLAM_FK, data=None, prior=my_prior,init_Nx=30,
                   ar_to_increase_Nx=0.1)
    alg_smc2 = particles.SMC(fk=fk_smc2, N=200, verbose= True)
    alg_smc2.run()

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

# %%
