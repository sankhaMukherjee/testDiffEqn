import os, sys

from logs import logDecorator as lD 
import json

import tensorflow as tf
import numpy as np
from scipy.integrate import odeint

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.odeModels.simpleODE_tf'

class Fullyconnected_noBias_tf:

    @lD.log(logBase + '.fullyconnected_noBias_tf.__init__')
    def __init__(logger, self, X, layers, activations, gpu_device=0):
        '''Given layers (no. of neurons in each layer) & activations (name of activation
        function to be applied on each layer), to create feedforward neural network
                
        Arguments:
            X : {ndarray} : 
                input data in following ndarray shape [n_cases, n_timepoints, n_variables], 
                to be saved in tf.Variable
                
                 n_cases: number of individuals for fitting longitudinal progressions 
                          (can be multiple progression line for 1 individual)
                 
                 n_timepoints: number of timepoints in such longitudinal progressions

                 n_variables: number of variables (or dimensions of data) for an individual to be used 
                 in longitudinal progressions fitting

            layers : {list} : 
                list of (int) to specify number of neurons for each layer
            activations : {list}: 
                list of (str) to specify name of activation function 
                to be applied on each layer
            gpu_device : {int} : 
                gpu id for running fullyconnected neural network
        '''

        try:
            self.device          = '/device:GPU:{}'.format(gpu_device)

            self.layers          = layers
            self.activations     = [self.get_activation(a) for a in activations]
            self.weights_list    = self.create_random_weights()

            with tf.device(self.device):

                with tf.variable_scope('inputs'):
                    self.inputX      = tf.Variable(X, dtype=tf.float32, name='inputX')
                    self.weights     = tf.Variable(self.weights_list, name='weights')

                with tf.variable_scope('fullyconnected'):
                    for index, (w, a) in enumerate(self.weights, self.activations):

                        if index == 0:
                            self.output = tf.matmul(self.inputX, w)
                        else:
                            self.output = tf.matmul(self.output, w)

                        self.output = a(self.output)

                with tf.variable_scope('misc'):
                    self.init       = tf.global_variables_initializer()

            tf_config = tf.ConfigProto(gpu_options={'allow_growth'                    : True,
                                                    'per_process_gpu_memory_fraction' : 0.5
                                                    },
                                       allow_soft_placement = True,
                                       log_device_placement = False)

            self.sess = tf.Session(config=tf_config)
            self.sess.run( self.init )

        except Exception as e:

            logger.error('Unable to initialize and create network \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.get_structural_weights')
    def get_structural_weights(logger, self, weights):
        '''Given list of new weights (float), create list of ndarray of new weights in 
        structural form based on current network structure
                
        Arguments:
            weights {list or 1D array} : 
                list of float values representing new weights

        Returns:
            {list} : 
                list of weights ndarray based on current network structure
        '''

        try:
            start        = 0
            weights_list = []

            for sublist in self.weights_list:
                wt_list = []
                for wt in sublist:
                    end    = np.prod(wt.shape)
                    new_wt = np.array(weights[start:(start+end)]).reshape(wt.shape)
                    start  = start + end
                    wt_list.append(new_wt)
                weights_list.append(wt_list)

            return weights_list

        except Exception as e:

            logger.error('Unable to get structural weights \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.create_random_weights')
    def create_random_weights(logger, self):
        '''Given layers (no. of neurons in each layer), create list of ndarray of random float values as weights
        
        Returns:
            {list} : list of ndarray of weights based on random float
        '''

        try: 
            prev_rows     = self.layers[0]
            weights_list  = []

            for i, l in enumerate(self.layers[1:]):

                w1        = np.random.random(size=(prev_rows, l)).astype(np.float32)
                weights_list.append(w1)

                prev_rows = l

            return weights_list

        except Exception as e:
            logger.error('Unable to create random weights \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.get_weight_shapes')
    def get_weight_shapes(logger, self):
        '''function to call self.get_weights(), 
        iterate through each ndarray and return weight.shape in list
        
        Returns:
            {list} : list of shape of weights ndarray
        '''

        try:
            return [wt.shape for wt in self.get_weights()]

        except Exception as e:
            logger.error('Unable to get weight shapes \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.get_num_weights')
    def get_num_weights(logger, self):
        '''function to call self.get_weight_shapes(), 
        iterate through each shape of weights ndarray and return total no. of weights
        
        Returns:
            {int} : 
                total no. of weights required for current construct of neural network
        '''

        try:
            return sum([np.prod(wt) for wt in self.get_weight_shapes()])

        except Exception as e:
            logger.error('Unable to get number of weights \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.get_weights')
    def get_weights(logger, self):
        '''function to return self.weights_list which is 
        list of weights ndarray
        
        Returns:
            {list} : 
                list of weights ndarray
        '''

        try:
            return self.weights_list

        except Exception as e:
            logger.error('Unable to get weights \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.get_activation')
    def get_activation(logger, self, activation):
        '''Given name of activation function, return tensorflow function
            
        Arguments:
            activation : {str} : 
                name of activation function
        
        Returns:
            {tf.activation function} : 
                corresponding tensorflow function based on name of activation function given
        
        Raises:
            Exception : 
                raise Exception when activation function not found
        '''

        try:
            if   activation == 'tanh':
                return tf.tanh

            elif activation == 'relu':
                return tf.nn.relu
                
            elif activation == 'sigmoid':
                return tf.sigmoid

            elif activation == 'linear':
                return tf.identity

            else:
                raise Exception('activation function not found')

        except Exception as e:
            logger.error('Unable to get activation function \n{}'.format(str(e)))

    @lD.log(logBase + '.fullyconnected_noBias_tf.feedforward')
    def feedforward(logger, self, weights):
        '''
        Given input data X, pass through feedforward network & output results
        
        Arguments:
            weights {list or 1D array} : 
                list of float values representing new weights
        
        Returns:
            {ndarray} : 
                outputs after passing input data X through feedforward network
        '''

        try:
            self.weights       = self.get_structural_weights(weights)
            output             = self.sess.run( self.output )

            return output

        except Exception as e:
            logger.error('Unable to pass feedforward network \n{}'.format(str(e)))

