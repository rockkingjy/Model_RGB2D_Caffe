#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

import numpy as np
import cv2
#import cv
import caffe
from caffe.proto import caffe_pb2
import sys

from google.protobuf import text_format
import argparse

path_to_global_context_network_caffemodel = \
"/media/enroutelab/sdd/mycodes/Depth-Estimation/source/global_context_network/snaps/_iter_100000.caffemodel"
# ATTENTION!!!-- This should be net_deploy not net_train!!!!!
path_to_gradient_network_definition_file = "/media/enroutelab/sdd/mycodes/Depth-Estimation/source/gradient_network/abs/net_deploy.prototxt"
path_to_gradient_network_caffemodel = \
"/media/enroutelab/sdd/mycodes/Depth-Estimation/source/gradient_network/snaps/_iter_100000.caffemodel"

#Create solver
caffe.set_mode_gpu()
caffe.set_device(1)
solver = caffe.get_solver('solver.prototxt') 

#1. copy net parameters from global contex network
solver.net.copy_from(path_to_global_context_network_caffemodel)

#2. copy net parameters from gradient network
gradPart = caffe.Net(path_to_gradient_network_definition_file, \
				path_to_gradient_network_caffemodel, caffe.TRAIN)

params = gradPart.params.keys() #get the name of the blobs and layers in gradient network
#get the names from gradient network
source_params = {pr: (gradPart.params[pr][0].data, \
		gradPart.params[pr][1].data) for pr in params}
#get the names form this refined network
target_params = {pr: (solver.net.params[pr][0].data, \
		solver.net.params[pr][1].data) for pr in params}
for pr in params:
    if pr == 'conv1': #copy 'conv1' -> 'conv1-grad'
		solver.net.params['conv1-grad'][1].data[...] = source_params [pr][1]  #biases
		solver.net.params['conv1-grad'][0].data[...] = source_params [pr][0]  #weights
    else:#copy others -> the same name;
		target_params[pr][1][...] = source_params [pr][1]  #bias
		target_params[pr][0][...] = source_params [pr][0]  #weights

#3. copy net parameters form alexNet, 'conv1' -> 'conv1-refine';
alexNet = caffe.Net(path_to_gradient_network_definition_file, \
			'../../bvlc_alexnet.caffemodel', caffe.TRAIN)
solver.net.params['conv1-refine'][1].data[...] = alexNet.params['conv1'][1].data  #biases
solver.net.params['conv1-refine'][0].data[...] = alexNet.params['conv1'][0].data  #weights

# run solver
solver.solve()
