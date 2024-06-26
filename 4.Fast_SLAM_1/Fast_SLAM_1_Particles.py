#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Adapting FastSLAM1 class and method to the particles library nomenclature in order to run
Sequential Monte Carlo and Markov Chain Monte Carlo methods to estimate the state and parameters
of the state space model of the ground robot represented in the dataset

Particles Library : https://github.com/nchopin/particles/tree/master

Author: Mahdi Chaari
Email: mchaari@unistra.fr
'''


import numpy as np

from lib import MotionModel
from lib import MeasurementModel
from lib import Particle
from src.Fast_SLAM_1_known_correspondences import FastSLAM1

# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles import resampling as rs
from particles import smc_samplers as ssp
from particles import utils


class fastSLAM_SSM(ssm.StateSpaceModel):
    default_params = {'sigma_x':0.0, 'sigma_y':0.0, 'sigma_theta':0.0, \
                           'sigma_v': 0.1, 'sigma_w': 0.15, 'gamma': 0.0, \
                            'sigma_range': 0.05, 'sigma_bearing': 0.02,\
                                'bias_v': -0.0, 'bias_w':0.0} 
    @staticmethod
    def create_fast_slam():
        dataset = "0.Dataset0"
        start_frame = 800
        end_frame = 40000
        ## Initialize FastSLAM1 object and load data
        fast_slam = FastSLAM1(None, None)
        # load data
        try:
            fast_slam.load_data(dataset, start_frame, end_frame)
        except:
            fast_slam.load_data("../"+dataset, start_frame, end_frame)

        # First state is obtained from ground truth
        fast_slam.states = np.array([fast_slam.groundtruth_data[0]])
        
        # Number of landmarks
        N_landmarks = len(fast_slam.landmark_indexes)

        # Landmark states: [x, y]
        fast_slam.landmark_states = np.zeros((N_landmarks, 2))

        return(fast_slam)
    
    _fast_slam = create_fast_slam()

    @property
    def fast_slam(self):
        return type(self)._fast_slam

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ## define motion model
        # [noise_x, noise_y, noise_theta, noise_v, noise_w, gamma]
        self.motion_noise = np.array([self.sigma_x, self.sigma_y, self.sigma_theta, \
                                         self.sigma_v, self.sigma_w, self.gamma])
        self.motion_model = MotionModel(self.motion_noise)
        
        ## Initialize Measurement Model object
        # Measurement covariance matrix
        Q = np.diagflat(np.array([self.sigma_range, self.sigma_bearing])) ** 2
        self.measurement_model = MeasurementModel(Q)
        
        # Number of landmarks
        self.N_landmarks = len(self.fast_slam.landmark_indexes)

    def PX0(self): 
        "Distribution of X_0"
        loc = self.fast_slam.groundtruth_data[0][1:]
        # update the timestep
        self.timestep = self.fast_slam.groundtruth_data[0][0]
        # this is definitely not the best way to sample \theta
        return dists.MvNormal(loc = loc, scale = self.motion_noise[:3])
    
    def POdom(self, t):
        "Law of odometry at time t"
        # if the data corresponds to odometry
        if self.fast_slam.data[t][1] == -1 :
            # the distribution of the odometry readings
            v = self.fast_slam.data[t][2] # linear velocity
            w = self.fast_slam.data[t][3] # angular velocity
            delta_t = self.fast_slam.data[t][0] - self.timestep
            # update timestep
            self.timestep = self.fast_slam.data[t][0]
            return [dists.MvNormal(loc = np.array([v, w, 0]), scale = self.motion_noise[3:]), delta_t]
        else :
            # data does not correspond to odometry
            return (None,0)
    
    def measure_data(self, t):
        """measurement data at time t"""
        measurement = self.fast_slam.data[t]
        if not self.fast_slam.data[t][1] in self.fast_slam.landmark_indexes :
            # data does not correspond to measurement data
            measurement_model = None
        else :
            # update timestep
            # self.timestep = measurement[0]
            measurement_model = self.measurement_model
            # Get landmark index
            measurement[1] = self.fast_slam.landmark_indexes[measurement[1]]
        return (measurement, measurement_model)


def landmark_update(particle, measurement_model, measurement, landmark_idx):
    if not particle.lm_ob[landmark_idx]:
        particle = measurement_model.initialize_landmark(particle,\
                                        measurement,landmark_idx, 0)
    # Update landmark by EKF if it has been observed before
    else:
        particle = measurement_model.\
            landmark_update(particle, measurement, landmark_idx)
    return(particle)

class fastSLAM_FK(ssm.Bootstrap):
    def __init__(self, ssm : fastSLAM_SSM, data=None):
        self.ssm = ssm
        self.data = data

    @property
    def T(self):
        return (len(self.ssm.fast_slam.data))


    def M0(self, N):
        """Sample N times from initial distribution M_0 of the FK model"""
        return self.ssm.PX0().rvs(size=N)

    def M(self, t, xp):
        """Generate X_t according to kernel M_t, conditional on X_{t-1}=xp"""
        noisy_control_dist, delta_t = self.ssm.POdom(t)
        if not(noisy_control_dist is None):
            noisy_control = noisy_control_dist.rvs(size = xp.shape[0])
            v = noisy_control[:,0] + self.ssm.bias_v
            w = noisy_control[:,1] + self.ssm.bias_w
            gamma = noisy_control[:,2]
            xp[:,0] = xp[:,0] - v/w * np.sin(xp[:,2]) + v/w * np.sin(xp[:,2] + w * delta_t) # x coordinate
            xp[:,1] = xp[:,1] + v/w * np.cos(xp[:,2]) - v/w * np.cos(xp[:,2] + w * delta_t) # y coordinate
            xp[:,2] += w * delta_t + gamma * delta_t # theta
        return xp

    def logG(self, t, xp, x):
        """Evaluates log of function G_t(x_{t-1}, x_t)"""
        raise NotImplementedError(self._error_msg("logG"))

    def logG(self, t, xp, x, x_ev, n_proc = 1):
        """Evaluates log of function G_t(x_t, measurement)"""
        measurement, measurement_model = self.ssm.measure_data(t)
        landmark_idx = int(measurement[1])
        # incremental log-weights
        inc_lw = np.ones(len(x_ev))
        # no multiprocessing
        if (not(measurement_model is None) and (n_proc == 1)):
            for i in range(len(x_ev)):
                # Initialize landmark by measurement if it is newly observed
                if not x_ev[i].lm_ob[landmark_idx]:
                    x_ev[i] = measurement_model.initialize_landmark(x_ev[i],\
                                                    measurement,landmark_idx, 0)
                # Update landmark by EKF if it has been observed before
                else:
                    x_ev[i] = measurement_model.\
                        landmark_update(x_ev[i], measurement, landmark_idx)
                    inc_lw[i] = x_ev[i].weight + 10**(-300)
            inc_lw = np.log(inc_lw)
        # multiprocessing
        elif not(measurement_model is None):
            # update the particles
            up_x_ev = utils.multiplexer(f= landmark_update, nprocs= n_proc, particle=x_ev, measurement_model= \
                              measurement_model, measurement = measurement, landmark_idx = landmark_idx)
            # sort the results after multiprocessing
            up_x_ev.sort(key=lambda output: output["particle"].particle_idx)
            # save results in proper format
            for i in range(len(x_ev)):
                x_ev[i] = up_x_ev[i]["output"]
                inc_lw[i] = x_ev[i].weight + 10**(-10)
            inc_lw = np.log(inc_lw)
        else:
            inc_lw = np.zeros(len(x_ev))
        return(x_ev, inc_lw)


class fastSLAM_SMC(particles.SMC):
    def __init__(
        self,
        fk : fastSLAM_FK,
        N=1000,
        qmc=False,
        resampling="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,
        n_proc = 1
    ):
        super().__init__(fk, N, qmc, resampling, ESSrmin, store_history, verbose, collect)
        # multiprocessing
        self.n_proc = n_proc
        # particles evolved (with a Kalman Filter)
        self.X_ev, self.Xp_ev, self.Aev = [], [], []
    
    def reset_weights(self):
        super().reset_weights()
        for i in range(len(self.X_ev)):
            self.X_ev[i].weight = 1.0 / self.N
            self.X_ev[i].particle_idx = i

    def generate_particles(self):
        # First state is obtained from ground truth
        states = np.array([self.fk.ssm.fast_slam.groundtruth_data[0]])
        
        # Landmark states: [x, y]
        landmark_states = np.zeros((self.fk.ssm.N_landmarks, 2))

        # Table to record if each landmark has been seen or not
        # [0] - [14] represent for landmark# 6 - 20
        # self.landmark_observed = np.full(self.N_landmarks, False)

        # Initialize particles
        self.X = self.fk.M0(self.N)
        # Limit θ within [-pi, pi]
        for i in range(len(self.X)):
            if (self.X[i][2] > np.pi):
                self.X[i][2] -= 2 * np.pi
            elif (self.X[i][2] < -np.pi):
                self.X[i][2] += 2 * np.pi 
        
        # initialize evolved particles that keep track of landmark positions
        for i in range(self.N):
            particle = Particle(i)
            particle.initialization(states[0], self.N,
                                    self.fk.ssm.N_landmarks)
            particle.x = self.X[i][0]
            particle.y = self.X[i][1]
            particle.theta = self.X[i][2]
            self.X_ev.append(particle)

    def reweight_particles(self):
        self.X_ev, wgts = self.fk.logG(self.t, self.Xp, self.X, self.X_ev, n_proc = self.n_proc)
        self.wgts = self.wgts.add(wgts)
    
    def resample_move(self):
        self.rs_flag = self.fk.time_to_resample(self)
        if self.rs_flag:  # if resampling
            self.A = rs.resampling(self.resampling, self.aux.W, M=self.N)
            # we always resample self.N particles, even if smc.X has a
            # different size (example: waste-free)
            self.Xp = self.X[self.A]
            # resample the evolved particles with EKFs
            X_ev_arr = np.array(self.X_ev)
            X_ev_arr = X_ev_arr[self.A]
            self.Xp_ev = list(X_ev_arr)
            self.reset_weights()
        else:
            self.A = np.arange(self.N)
            self.Xp = self.X
            self.Xp_ev = self.X_ev
        self.X = self.fk.M(self.t, self.Xp)
        # Limit θ within [-pi, pi]
        for i in range(len(self.X)):
            if (self.X[i][2] > np.pi):
                self.X[i][2] -= 2 * np.pi
            elif (self.X[i][2] < -np.pi):
                self.X[i][2] += 2 * np.pi
        
        for i in range(self.N):
            self.X_ev[i].x = self.X[i][0]
            self.X_ev[i].y = self.X[i][1]
            self.X_ev[i].theta = self.X[i][2]

    def __next__(self):
        """One step of a particle filter."""
        if self.fk.done(self):
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.setup_auxiliary_weights()  # APF
            if self.qmc:
                self.resample_move_qmc()
            else:
                self.resample_move()
        self.reweight_particles()
        self.compute_summaries()
        self.t += 1
        if self.verbose :
            # plot the estimated trajectory and position of landmarks
            self.fk.ssm.fast_slam.state_update_(self.X_ev, self.wgts.W)
            if (len(self.fk.ssm.fast_slam.states) % 100 == 0):
                self.fk.ssm.fast_slam.plot_data(self.X_ev)


class fastSLAM_SMC2(ssp.SMC2):
    def __init__(
            self,
            ssm_cls : fastSLAM_SSM,
            prior=None,
            data=None,
            smc_options=None,
            fk_cls = None,
            init_Nx=100,
            ar_to_increase_Nx=-1,
            wastefree=True,
            len_chain=10,
            move=None):
        super().__init__(ssm_cls, prior, data, smc_options, fk_cls, \
                          init_Nx, ar_to_increase_Nx, wastefree, len_chain, move)

    @property
    def T(self):
        return (len(self.ssm_cls._fast_slam.data))
    
    def alg_instance(self, theta, N):
        return fastSLAM_SMC(
            fk=self.fk_cls(ssm=self.ssm_cls(**theta), data=self.data),
            N=N,
            n_proc= -1,
            **self.smc_options
        )

if __name__ == '__main__':
    pass