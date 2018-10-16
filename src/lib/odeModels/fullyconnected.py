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

class multipleODE_new:
    '''[summary]
    
    [description]
    '''


    @lD.log(logBase + '.__init__')
    def __init__(logger, self, Npat, Nnt, Nl, tspan, Atimesj, Btimesj, fj, rj, mj, 
                 stress_t, stress_v, layers, activations, gpu_device='0'):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        try:
            self.Npat        = Npat     # --> 1 number 
            self.Nnt         = Nnt      # --> 1 number
            self.Nl          = Nl       # --> 1 number
            self.NperUser    = Nnt + Nl # --> 1 number
            self.tspan       = tspan    # --> 1D array
            self.Atimesj     = Atimesj  # --> Npat arrays
            self.Btimesj     = Btimesj  # --> Npat arrays
            self.fj          = fj       # --> Npat arrays
            self.rj          = rj       # --> Npat arrays
            self.mj          = mj       # --> Npat arrays
            self.stress_t    = stress_t
            self.stress_v    = stress_v

            activation_map   = { 'tanh'    : tf.nn.tanh,
                                 'sigmoid' : tf.nn.sigmoid,
                                 'relu'    : tf.nn.relu,
                                 'linear'  : tf.identity    }
            activations      = [ activation_map[a] for a in activations ]

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
            self.device      = ['/device:GPU:{}'.format(g) for g in gpu_device.split(',')]

            for d in self.device:
                with tf.device(d):
                    self.tf_opsFlow(layers, activations)

        except Exception as e:
            logger.error('Unable to initialize multipleODE \n{}'.format(str(e)))

    @lD.log(logBase + '.tf_opsFlow')
    def tf_opsFlow(logger, self, layers, activations):

        try:
            with tf.variable_scope('Constant'):

                self.tspan_tf    = tf.constant(self.tspan, dtype=tf.float32, name='tspan')

                self.Atimesj_tf  = [  tf.constant(dose_vec, dtype=tf.float32, name='doseA_vec_{}'.format(index)) 
                                            for index, dose_vec in enumerate(self.Atimesj)]

                self.Btimesj_tf  = [  tf.constant(dose_vec, dtype=tf.float32, name='doseB_vec_{}'.format(index)) 
                                            for index, dose_vec in enumerate(self.Btimesj)]

                self.fj_tf       = [  tf.constant(fj_vec, dtype=tf.float32, name='fj_{}'.format(index)) 
                                            for index, fj_vec in enumerate(self.fj)]

                self.rj_tf       = [  tf.constant(rj_vec, dtype=tf.float32, name='rj_{}'.format(index)) 
                                            for index, rj_vec in enumerate(self.rj)]

                self.mj_tf       = [  tf.constant(mj_vec, dtype=tf.float32, name='mj_{}'.format(index)) 
                                            for index, mj_vec in enumerate(self.mj)]

                self.stress_t_tf = [  tf.constant(stress_tvec, dtype=tf.float32, name='stress_tvec_{}'.format(index)) 
                                            for index, stress_tvec in enumerate(self.stress_t)]

                self.stress_v_tf = [  tf.constant(stress_vvec, dtype=tf.float32, name='stress_vvec_{}'.format(index)) 
                                            for index, stress_vvec in enumerate(self.stress_v)]

            with tf.variable_scope('Variable'):

                self.NNwts_tf    = []
                self.NNb_tf      = []
                self.NNact_tf    = []
                self.Taus_tf     = []

                for i in range(len(layers)):
                    if i == 0:
                        wts = tf.Variable(np.random.random(size=(layers[i], self.Nnt + 1)), dtype=tf.float32, name='wts_{}'.format(i))
                    else:
                        wts = tf.Variable(np.random.random(size=(layers[i+1], layers[i])), dtype=tf.float32, name='wts_{}'.format(i))

                    bias = tf.Variable(np.random.rand(), dtype=tf.float32, name='bias_{}'.format(i))
                    act  = activations[i]
                    tau  = tf.Variable(np.random.rand(), dtype=tf.float32, name='tau_{}'.format(i))

                    self.NNwts_tf.append(wts)
                    self.NNb_tf.append(bias)
                    self.NNact_tf.append(act)
                    self.Taus_tf.append(tau)

            with tf.variable_scope('rhs_operation'):

                self.rhs_results = []
                self.y_tf        = tf.placeholder(dtype=tf.float32, name='y_tf')
                self.t           = tf.placeholder(dtype=tf.float32, name='dt')
                
                for user in range(self.Npat):

                    Aj = self.interpolate(dx_T=self.tspan_tf, dy_T=self.Atimesj_tf[user], x=self.t)
                    Bj = self.interpolate(dx_T=self.tspan_tf, dy_T=self.Btimesj_tf[user], x=self.t)
                    
                    # Calculate the neurotransmitters
                    result_neurotransmitters = self.fj_tf[user] - self.rj_tf[user] * self.y_tf[(user*self.NperUser) : (user*self.NperUser+self.Nnt)] / ( 1 + Aj ) \
                                                                - self.mj_tf[user] * self.y_tf[(user*self.NperUser) : (user*self.NperUser+self.Nnt)] / ( 1 + Bj ) 


                    # Calculate long-term dependencies
                    # This is the NN([ n1, n2, n3, s ])

                    res_ls = []

                    for j in range(self.Nl):

                        # Extract [n1, n2, n3]
                        neurotransmitters_list = self.y_tf[ (user*self.NperUser) : (user*self.NperUser + self.Nnt)]

                        # get interpolated s at t
                        stress_value           = self.interpolate(dx_T=self.stress_t_tf[user], dy_T=self.stress_v_tf[user], x=self.t)

                        # concatenate to [ n1, n2, n3, s ]
                        res                    = tf.concat([neurotransmitters_list, [stress_value]], axis=0)
                        res                    = tf.reshape(res, [-1, 1])

                        for index, (w, b, a) in enumerate(zip(self.NNwts_tf, self.NNb_tf, self.NNact_tf)):
                            res = tf.matmul(w, res) + b
                            res = a(res)

                        res  = res[0][0] - self.y_tf[ user*self.NperUser + self.Nnt + j] / self.Taus_tf[j]
                        res_ls.append(res)

                    results     = tf.concat([result_neurotransmitters, res_ls], axis=0)
                    self.rhs_results.append(results)

                self.rhs_results = tf.concat(self.rhs_results, axis=0)

            self.init  = tf.global_variables_initializer()

            config     = tf.ConfigProto(gpu_options={'allow_growth':True})
            self.sess  = tf.Session(config=config)
            tfwriter   = tf.summary.FileWriter('./tensorlog/', self.sess.graph)
            tfwriter.close()
            self.sess.run( self.init )

        except Exception as e:

            logger.error('Unable to create tensorflow ops flow \n{}'.format(str(e)))

    @lD.log(logBase + '.interpolate')
    def interpolate(logger, self, dx_T, dy_T, x, name='interpolate' ):
        
        try:
            with tf.variable_scope(name):
                
                with tf.variable_scope('neighbors'):
                    
                    delVals = dx_T - x
                    ind_1   = tf.argmax(tf.sign( delVals ))
                    ind_0   = ind_1 - 1
                    
                with tf.variable_scope('calculation'):
                    
                    value   = tf.cond( x[0] <= dx_T[0], 
                                      lambda : dy_T[:1], 
                                      lambda : tf.cond( 
                                             x[0] >= dx_T[-1], 
                                             lambda : dy_T[-1:],
                                             lambda : (dy_T[ind_0] +                \
                                                       (dy_T[ind_1] - dy_T[ind_0])  \
                                                       *(x-dx_T[ind_0])/            \
                                                       (dx_T[ind_1]-dx_T[ind_0]))
                                     ))
                    
                result = tf.multiply(value[0], 1, name='y')
            
            return result

        except Exception as e:
            
            logger.error('Unable to interpolate \n{}'.format(str(e)))

    # @jit
    def jac(self, y, t, NNwts, NNb, NNact, NNactD, Taus):
        '''[summary]
        
        This has not been implemented yet
        
        Parameters
        ----------
        y : {[type]}
            [description]
        t : {[type]}
            [description]
        NNwts : {[type]}
            [description]
        NNb : {[type]}
            [description]
        NNact : {[type]}
            [description]
        NNactD : {[type]}
            [description]
        Taus : {[type]}
            [description]
        
        Returns
        -------
        [type]
            [description]
        '''

        # print('.', end='')
        result = np.zeros((self.Nnt+self.Nnt, self.Nnt+self.Nnt))

        Aj = self.AjFunc(t, self.Atimesj)
        Bj = self.BjFunc(t, self.Btimesj)

        for i in range(self.Nnt):
            
            ntArr    = np.zeros( self.Nnt + 1 ) # 1 stressor added
            ntArr[i] = 1

            for j in range(self.Nl):

                resD = (ntArr * 1).reshape((-1, 1))
                res = np.hstack((y[ : self.Nnt], np.array([self.stress(t)]) ))
                res = res.reshape((-1, 1))

                # Find the divergence ...
                for w, b, a, da in zip(NNwts, NNb, NNact, NNactD):

                    res = np.matmul(w, res) #+ b

                    resD = np.matmul(w, resD) #+ b
                    resD = resD * da( res ) 

                    res = a(res)
                    

                # final value
                resD = resD[0][0]

                result[i , j+self.Nnt] = resD

        for i in range(self.Nnt):
            result[i, i] -= self.rj[i]/( 1 + Aj ) 
            result[i, i] -= self.mj[i]/( 1 + Bj ) 

        for i in range(self.Nl):
            result[i+self.Nnt, i+self.Nnt] = -1/Taus[i]

        return result

    @lD.log(logBase + '.dy')
    def dy(logger, self, y, t):
        '''[summary]
        
        [description]
        
        Arguments:
            y {[type]} -- [description]
            t {[type]} -- [description]
        '''

        try:
            rhs_results = self.sess.run( self.rhs_results, 
                                            feed_dict={
                                                self.y_tf : y,
                                                self.t    : [t]
                                        })

            return rhs_results

        except Exception as e:

            logger.error('Unable to get dy result \n{}'.format(str(e)))

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            NNwts, NNb, Taus  = args
            for index, (w, b, t) in enumerate(zip(NNwts, NNb, Taus)):
                self.sess.run( self.NNwts_tf[i].assign(w) )
                self.sess.run( self.NNb_tf[i].assign(b) )
                self.sess.run( self.Taus_tf[i].assign(t) )

            jac = None
            if useJac:
                jac = self.jac

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, Dfun=jac, full_output=True, mxstep=50000)
            else:
                y_t = odeint(self.dy, y0, t, Dfun=jac, full_output=False)

            # if useJac:
            #     print('')

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))

