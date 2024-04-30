#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Definition of the M*N  EK-filters of the landmarks

This class is designed to help parallelize the computation of the particle filters every
time a new measurement comes

It keeps track of all the means and covariance of each landmark across all the particles

This class takes advantage of the fact that at each timestep, only one measurement of a 
single landmark is available

Author: Mahdi Chaari
Email: mchaari@unistra.fr
'''
import numpy as np

class EKF_Landmarks():
    def __init__(self, N_particles, N_landmarks, Q) -> None:
        '''
        Input:
            N_particles : the number of particles in the particle filter
            N_landmarks : the number of landmarks in the environment
            Q: Measurement covariance matrix.
               Dimension: [2, 2].
        '''
        # All landmarks' Mean and Covariance
        self.lm_mean = np.zeros((N_landmarks, 2*N_particles))
        self.lm_cov  = np.zeros((N_landmarks, 2*N_particles, 2*N_particles))

        # Table to record if each landmark has been seen or not
        # INdex [0] - [14] represent for landmark# 6 - 20
        self.lm_ob = np.full(N_landmarks, False)

        # self.Q : Dimension [2*N_particles, 2*N_particles] 
        self.Q = np.kron(np.eye(N_particles,dtype=int),Q)
        
        self.N_particles = N_particles
        self.N_landmarks = N_landmarks

    def compute_expected_measurement(self, particles : np.ndarray, landmark_idx):
        '''
        Compute the expected range and bearing given current robot state and
        landmark state.

        Measurement model: (expected measurement)
        range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t

        Input:
            particles: np array (N_particles, 3)
                        [x, y, theta]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            range, bearing: the expected measurement.
        '''
        delta_x = self.lm_mean[landmark_idx, ::2  ] - particles[:,0]
        delta_y = self.lm_mean[landmark_idx, 1::2 ] - particles[:,1]
        q = delta_x.T ** 2 + delta_y ** 2

        range = np.sqrt(q)
        bearing = np.arctan2(delta_y, delta_x) - particles[:,2]

        return range, bearing
    
    def compute_expected_landmark_state(self, particles : np.ndarray, measurement):
        '''
        Compute the expected landmark location [x, y] given current robot state
        and measurement data.

        Expected landmark state: inverse of the measurement model.
        x_l = x_t + range_t * cos(bearing_t + theta_t)
        y_l = y_t + range_t * sin(bearing_t + theta_t)

        Input:
            particles: np array (N_particles, 3)
                        [x, y, theta]
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
        Output:
            x, y: expected landmark state [x, y]
        '''
        x = particles[:,0] + measurement[2] *\
            np.cos(measurement[3] + particles[:,2])
        y = particles[:,1] + measurement[2] *\
            np.sin(measurement[3] + particles[:,2])
        return x, y
    
    def compute_landmark_jacobian(self, particles, landmark_idx) -> np.ndarray:
        '''
        Computing the Jacobian wrt landmark state X_l.

        Jacobian of measurement: only take derivatives of landmark X_l.
                                 H = d h(x_t, x_l) / d (x_l)
        H_m =  delta_x/√q  delta_y/√q
               -delta_y/q  delta_x/q

        Input:
            particles: np array (N_particles, 3)
                        [x, y, theta]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            H_m: Jacobian h'(X_t, X_l)
                 Dimension: [2*N_particles, 2*N_particles]
        '''
        delta_x = self.lm_mean[landmark_idx, ::2  ] - particles[:,0]
        delta_y = self.lm_mean[landmark_idx, 1::2 ] - particles[:,1]
        q = delta_x.T ** 2 + delta_y ** 2

        H_0_0 = delta_x/np.sqrt(q)
        H_0_1 = delta_y/np.sqrt(q)
        H_1_0 = -delta_y/q
        H_1_1 = delta_x/q

        # fill the H_m matrix
        # this is not an elegant way of filling the matrix
        # but it is not computationally costly
        H_m = np.zeros((2*self.N_particles, 2*self.N_particles))
        
        for i in range(self.N_particles):
            H_m[2*i,   2*i  ] = H_0_0[i]
            H_m[2*i,   2*i+1] = H_0_1[i]
            H_m[2*i+1, 2*i  ] = H_1_0[i]
            H_m[2*i+1, 2*i+1] = H_1_1[i]

        return H_m
    
    def initialize_landmark(self, particles, measurement, landmark_idx) -> None:
        '''
        Initialize landmark mean and covariance for one landmark of a given
        particle.
        This landmark is the first time to be observed.
        Based on EKF method.

        Input:
            particles: np array (N_particles, 3)
                        [x, y, theta]
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            None
        '''
        # Update landmark mean by inverse measurement model
        x , y = self.compute_expected_landmark_state(particles, measurement)
        self.lm_mean[landmark_idx, ::2 ] = x
        self.lm_mean[landmark_idx, 1::2] = y

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particles, landmark_idx)

        # Update landmark covariance
        H_inverse = np.linalg.inv(H_m)
        self.lm_cov[landmark_idx] = H_inverse @ self.Q @ (H_inverse.T)

        # Mark landmark as observed
        self.lm_ob[landmark_idx] = True

    def landmark_update(self, particles, measurement, landmark_idx):
        '''
        Implementation for Fast SLAM 1.0.
        Update landmark mean and covariance for one landmarks of a given
        particle.
        This landmark has to be observed before.
        Based on EKF method.

        Input:
            particles: np array (N_particles, 3)
                        [x, y, theta]
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            weights : np.array.
        '''
        # Compute expected measurement
        range_expected, bearing_expected =\
            self.compute_expected_measurement(particles, landmark_idx)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particles, landmark_idx)

        # Compute Kalman gain
        Q = H_m.dot(self.lm_cov[landmark_idx]).dot(H_m.T) + self.Q
        K = self.lm_cov[landmark_idx].dot(H_m.T).dot(np.linalg.inv(Q))

        # Update mean
        difference = np.zeros((2*self.N_particles))
        difference[::2 ] = measurement[2] - range_expected
        difference[1::2] = measurement[3] - bearing_expected
        innovation = K @ difference

        self.lm_mean[landmark_idx] += innovation

        # Update covariance
        self.lm_cov[landmark_idx] =\
            (np.identity(2*self.N_particles) - K @ H_m) @self.lm_cov[landmark_idx]

        # Importance factor
        weights = np.zeros((self.N_particles))
        Q_det  = np.zeros((self.N_particles))
        for i in range(self.N_particles):
            Q_det[i] = np.linalg.det(2 * np.pi * Q[2*i:2*i+2,2*i:2*i+2]) ** (-0.5) 
        difference = np.kron(np.eye(N=self.N_particles,dtype=int),np.array([1,1])) * difference.T
        weights = np.diag(Q_det * np.exp(-0.5 * difference @ (np.linalg.inv(Q)) @ (difference.T)))

        return (weights)
    
    def resample_landmarks(self, A: np.ndarray)-> None:
        '''
        updates the means and covariance of the landmarks after a resampling move
        Input :
            A : np array containing the resampled indices
        '''
        # update the means of landmarks
        self.lm_mean[:,::2 ] = self.lm_mean[:,A*2]
        self.lm_mean[:,1::2] = self.lm_mean[:,A*2 + 1]
        # update the cov of landmarks
        for i in range (self.N_particles):
            j = A[i]
            self.lm_cov[:,2*i:2*i+2,2*i:2*i+2] = self.lm_cov[:,2*j:2*j+2,2*j:2*j+2]
if __name__ == '__main__':
    pass