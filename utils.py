#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:42:59 2018

@author: achaman
"""


import numpy as np
from numpy.random import randint
import scipy


def calculate_mccs_message_power(g_filter, Noise, L_g):

    total_mccs_message_power = 0
    
    num_speakers = len(Noise)

    for i in range(num_speakers):

        message = np.convolve(g_filter[i*L_g:(i+1)*L_g], Noise[i])

        total_mccs_message_power += np.linalg.norm(message)**2/np.prod(message.shape)
        
    return total_mccs_message_power
    


def crop_start_end_rir(h_listeners, h_eaves, L_h):
    
    num_listeners = len(h_listeners)
    num_speakers = len(h_listeners[0])

    start_time_all_listeners = []

    for i in range(num_listeners):

        start = 1000000000
        for j in range(num_speakers):

            peak_locs = scipy.signal.find_peaks(h_listeners[i][j])[0]

            start_point = peak_locs[h_listeners[i][j][peak_locs].argmax()]

            if start> start_point:
                start = start_point


        start_time_all_listeners.append(start)



    for i in range(num_listeners):
        for j in range(num_speakers):
            h_listeners[i][j] = h_listeners[i][j][start_time_all_listeners[i] : start_time_all_listeners[i] + L_h]



    # similarly find the earliest arrival for h_eaves-> useful during evaluation. 
    # Though we choose the same L_h for eavesdropper, it is not necessary

    start_eaves = 100000
    L_h_eaves = L_h

    for j in range(num_speakers):
        peak_locs = scipy.signal.find_peaks(h_eaves[j])[0]
        start_point = peak_locs[h_eaves[j][peak_locs].argmax()]

        if start_point< start_eaves:
            start_eaves = start_point

    for j in range(num_speakers):
        h_eaves[j] = h_eaves[j][start_eaves: start_eaves + L_h_eaves]
        
        
    return h_listeners, h_eaves


        
        
        
        
        

def find_start_and_end_points(z, max_value):
    L=len(z)

    start_points=-1*np.ones(len(z))
    end_points=-1*np.ones(len(z))
        
    for p in range(len(z)):
        peak_locs=scipy.signal.find_peaks(z[p])[0]
        max_val=z[p][peak_locs].max()
        temp=np.where(z[p]==max_val)[0][0]
        start_points[p]=temp
    
    
    
    
    for p in range(L):
        norm_len=[]
    
        
        ztemp=z[p]
        
        for q in range(0,len(z[p])):
            energy=norm(ztemp[0:q])**2/(norm(ztemp)**2)
            
            norm_len.append(energy)
           
        norm_len=np.array(norm_len)
        
        end_points[p]=np.argmax(norm_len>max_value)
    
    earliest_start=int(np.min(start_points))
    last_end=int(np.max(end_points))
    
        
    
    return [start_points,end_points,earliest_start,last_end]
            
        
















def crop_signal(signal, fs1):
    
    if len(signal)>4*fs1:
        signal = signal[2*fs1 : 4*fs1]
        
    elif len(signal>2*fs1) and len(signal)<4*fs1:
        signal = signal[0 : 2*fs1]
        
    return signal



def yield_speaker_listener_config(room_dims, num_speakers, num_listeners):


    # speaker and listener coordinates are chosen with a 1m margin from room walls
    
    l1 = room_dims[0]-1
    l2 = room_dims[1]-1

    #define sources

    
    source_x_cor = np.random.randint(1, l1*100, size = num_speakers)/100 
    source_y_cor = np.random.randint(1, l2*100, size = num_speakers)/100 
    
    source_coordinates = np.stack((source_x_cor, source_y_cor), axis = 1)

    
    #define listeners
    
    listener_x_cor = np.random.randint(1, l1*100, size = num_listeners)/100 
    listener_y_cor = np.random.randint(1, l2*100, size = num_listeners)/100 
    
    listener_coords = np.stack((listener_x_cor, listener_y_cor), axis = 1)
    
    
    #define 1 eavesdropper location coordinates
    
    eavesdropper_coords = np.array([np.random.randint(1, l1*100)/100, np.random.randint(1, l2*100)/100])
    
    
    return source_coordinates, listener_coords, eavesdropper_coords












