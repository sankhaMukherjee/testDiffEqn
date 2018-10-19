import os, sys

from logs import logDecorator as lD 
import json

from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.client import timeline

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.multipleODE.multipleODE_tf'


class multipleODE:
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
            self.Npat     = Npat     # --> 1 number 
            self.Nnt      = Nnt      # --> 1 number
            self.Nl       = Nl       # --> 1 number
            self.NperUser = Nnt + Nl # --> 1 number
            self.tspan    = tspan    # --> 1D array
            self.fj       = fj       # --> Npat arrays
            self.rj       = rj       # --> Npat arrays
            self.mj       = mj       # --> Npat arrays

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
            self.device   = ['/device:GPU:{}'.format(g) for g in gpu_device.split(',')]

            self.stressInterp = [interp1d(t_vec, s_vec) for t_vec, s_vec in zip(stress_t, stress_v)]
            self.AjInterp     = [interp1d(tspan, a_vec) for a_vec in Atimesj]
            self.BjInterp     = [interp1d(tspan, b_vec) for b_vec in Btimesj]

            activation_map    = { 'tanh'    : tf.nn.tanh,
                                  'sigmoid' : tf.nn.sigmoid,
                                  'relu'    : tf.nn.relu,
                                  'linear'  : tf.identity    }
            activations       = [ activation_map[a] for a in activations ]

            start             = time.time()

            for d in self.device:
                with tf.device(d):
                    self.tf_opsFlow(layers=layers, activations=activations)
            
            timespent         = time.time() - start
            print('graphTime', timespent)

            # self.options      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.run_metadata = tf.RunMetadata()

        except Exception as e:
            logger.error('Unable to initialize multipleODE \n{}'.format(str(e)))

    @lD.log(logBase + '.tf_opsFlow')
    def tf_opsFlow(logger, self, layers, activations):

        try:
            with tf.variable_scope('weights'):

                self.fj_tf       = [  tf.Variable(fj_vec, dtype=tf.float32, name='fj_{}'.format(index)) 
                                            for index, fj_vec in enumerate(self.fj)]

                self.rj_tf       = [  tf.Variable(rj_vec, dtype=tf.float32, name='rj_{}'.format(index)) 
                                            for index, rj_vec in enumerate(self.rj)]

                self.mj_tf       = [  tf.Variable(mj_vec, dtype=tf.float32, name='mj_{}'.format(index)) 
                                            for index, mj_vec in enumerate(self.mj)]

                self.NNwts_tf    = []
                self.NNb_tf      = []
                self.NNact_tf    = []
                self.Taus_tf     = []
                prev_l           = self.Nnt + 1

                for i, l in enumerate(layers):

                    wts    = tf.Variable(np.random.random(size=(l, prev_l)), dtype=tf.float32, name='wts_{}'.format(i))
                    prev_l = l

                    bias   = tf.Variable(np.random.rand(), dtype=tf.float32, name='bias_{}'.format(i))
                    act    = activations[i]
                    tau    = tf.Variable(np.random.rand(), dtype=tf.float32, name='tau_{}'.format(i))

                    self.NNwts_tf.append(wts)
                    self.NNb_tf.append(bias)
                    self.NNact_tf.append(act)
                    self.Taus_tf.append(tau)

            with tf.variable_scope('rhs_operation'):

                self.y_tf        = tf.placeholder(dtype=tf.float32, name='y_tf')
                self.t           = tf.placeholder(dtype=tf.float32, name='dt')
                self.stress_val  = tf.placeholder(dtype=tf.float32, name='stress_val')
                self.Aj          = tf.placeholder(dtype=tf.float32, name='Aj')
                self.Bj          = tf.placeholder(dtype=tf.float32, name='Bj')

                def get_slowVaryingComponents(j, Nnt_list, stressVal, slowComponents):

                    with tf.variable_scope('cpnt{}'.format(j)):

                        # concatenate to [ n1, n2, n3, s ]
                        res  = tf.concat([Nnt_list, [stressVal]], axis=0, name='concat{}'.format(j))
                        res  = tf.reshape(res, [-1, 1])

                        for index, (w, b, a) in enumerate(zip(self.NNwts_tf, self.NNb_tf, self.NNact_tf)):
                            res = tf.matmul(w, res) + b
                            res = a(res)

                        res  = res[0][0] - slowComponents[j] / self.Taus_tf[j]
                        
                        return res

                def rhs_operation(user):

                    with tf.variable_scope('user_{:05}'.format(user)):

                        # Extract [n1, n2, n3]
                        Nnt_list       = self.y_tf[ (user*self.NperUser) : (user*self.NperUser + self.Nnt)]
                        stressVal      = self.stress_val[user]
                        slowComponents = self.y_tf[ (user*self.NperUser) : (user*self.NperUser + 3)]
                       
                        with tf.variable_scope('Nnt_parts'):
                            
                            # Calculate the neurotransmitters
                            Nnt_result = self.fj_tf[user] - self.rj_tf[user] * Nnt_list / ( 1 + self.Aj[user] ) \
                                                          - self.mj_tf[user] * Nnt_list / ( 1 + self.Bj[user] ) 

                        with tf.variable_scope('slow_Components'):
                            
                            # Calculate long-term dependencies
                            # This is the NN([ n1, n2, n3, s ])
                            res_ls     = [get_slowVaryingComponents(j, Nnt_list, stressVal, slowComponents) for j in range(self.Nl)]
                            results    = tf.concat([Nnt_result, res_ls], axis=0)

                        return results
                
                self.rhs_results = [rhs_operation(user) for user in range(self.Npat)]
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

    # @lD.log(logBase + '.dy')
    def dy(self, y, t):
        '''[summary]
        
        [description]
        
        Arguments:
            y {[type]} -- [description]
            t {[type]} -- [description]
        '''

        try:
        
            rhs_results = self.sess.run( self.rhs_results, 
                                         # options=self.options, run_metadata=self.run_metadata,
                                         feed_dict={
                                             self.y_tf       : y,
                                             self.t          : [t],
                                             self.stress_val : [interp(t) for interp in self.stressInterp],
                                             self.Aj         : [interp(t) for interp in self.AjInterp],
                                             self.Bj         : [interp(t) for interp in self.BjInterp]
                                    })

            return rhs_results

        except Exception as e:

            # logger.error('Unable to get dy result \n{}'.format(str(e)))
            print('Unable to get dy result \n{}'.format(str(e)))

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            NNwts, NNb, NNact, Taus  = args

            for i, (weights, bias, tau) in enumerate(zip(NNwts, NNb, Taus)):
                self.sess.run( self.NNwts_tf[i].assign(weights) )
                self.sess.run( self.NNb_tf[i].assign(bias) )
                self.sess.run( self.Taus_tf[i].assign(tau) )

            jac = None
            if useJac:
                jac = self.jac

            start = time.time()

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, Dfun=jac, full_output=True, mxstep=50000)
            else:
                y_t = odeint(self.dy, y0, t, Dfun=jac, full_output=False)
            
            timespent = time.time() - start
            print('odeTime', timespent)

            # fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            # chrome_trace     = fetched_timeline.generate_chrome_trace_format()

            # with open('timeline_step.json', 'w') as f:
            #     f.write(chrome_trace)

            # if useJac:
            #     print('')

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))

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
            self.Npat     = Npat     # --> 1 number 
            self.Nnt      = Nnt      # --> 1 number
            self.Nl       = Nl       # --> 1 number
            self.NperUser = Nnt + Nl # --> 1 number
            self.tspan    = tspan    # --> 1D array
            self.fj       = fj       # --> Npat arrays
            self.rj       = rj       # --> Npat arrays
            self.mj       = mj       # --> Npat arrays

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
            self.device   = ['/device:GPU:{}'.format(g) for g in gpu_device.split(',')]

            self.stressInterp = [interp1d(t_vec, s_vec) for t_vec, s_vec in zip(stress_t, stress_v)]
            self.AjInterp     = [interp1d(tspan, a_vec) for a_vec in Atimesj]
            self.BjInterp     = [interp1d(tspan, b_vec) for b_vec in Btimesj]

            activation_map    = { 'tanh'    : tf.nn.tanh,
                                  'sigmoid' : tf.nn.sigmoid,
                                  'relu'    : tf.nn.relu,
                                  'linear'  : tf.identity    }
            activations       = [ activation_map[a] for a in activations ]

            start             = time.time()

            for d in self.device:
                with tf.device(d):
                    self.tf_opsFlow(layers=layers, activations=activations)
            
            timespent         = time.time() - start
            print('graphTime', timespent)

            # self.options      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.run_metadata = tf.RunMetadata()

        except Exception as e:
            logger.error('Unable to initialize multipleODE \n{}'.format(str(e)))

    @lD.log(logBase + '.tf_opsFlow')
    def tf_opsFlow(logger, self, layers, activations):

        try:
            with tf.variable_scope('weights'):

                self.fj_tf       = tf.Variable(self.fj, dtype=tf.float32, name='fj')
                self.rj_tf       = tf.Variable(self.rj, dtype=tf.float32, name='rj')
                self.mj_tf       = tf.Variable(self.mj, dtype=tf.float32, name='mj')

                self.NNwts_tf    = []
                self.NNb_tf      = []
                self.NNact_tf    = []
                self.Taus_tf     = []
                prev_l           = self.Nnt + 1

                for i, l in enumerate(layers):

                    wts    = tf.Variable(np.random.random(size=(l, prev_l)), dtype=tf.float32, name='wts_{}'.format(i))
                    prev_l = l

                    bias   = tf.Variable(np.random.rand(), dtype=tf.float32, name='bias_{}'.format(i))
                    act    = activations[i]
                    tau    = tf.Variable(np.random.rand(), dtype=tf.float32, name='tau_{}'.format(i))

                    self.NNwts_tf.append(wts)
                    self.NNb_tf.append(bias)
                    self.NNact_tf.append(act)
                    self.Taus_tf.append(tau)

            with tf.variable_scope('rhs_operation'):

                self.y_tf        = tf.placeholder(dtype=tf.float32, name='y_tf')
                self.stress_val  = tf.placeholder(dtype=tf.float32, name='stress_val')
                self.Aj          = tf.placeholder(dtype=tf.float32, name='Aj')
                self.Bj          = tf.placeholder(dtype=tf.float32, name='Bj')

                def get_slowVaryingComponents(j, Nnt_list, stressVal, slowComponents):

                    with tf.variable_scope('cpnt{}'.format(j)):

                        # concatenate to [ n1, n2, n3, s ]
                        res  = tf.concat([Nnt_list, [stressVal]], axis=0, name='concat{}'.format(j))
                        res  = tf.reshape(res, [-1, 1])

                        for index, (w, b, a) in enumerate(zip(self.NNwts_tf, self.NNb_tf, self.NNact_tf)):
                            res = tf.matmul(w, res) + b
                            res = a(res)

                        res  = res[0][0] - slowComponents[j] / self.Taus_tf[j]
                        
                        return res

                def rhs_operation(user):

                    with tf.variable_scope('user_{:05}'.format(user)):

                        # Extract [n1, n2, n3]
                        Nnt_list       = self.y_tf[ (user*self.NperUser) : (user*self.NperUser + self.Nnt)]
                        stressVal      = self.stress_val[user]
                        slowComponents = self.y_tf[ (user*self.NperUser) : (user*self.NperUser + 3)]
                       
                        with tf.variable_scope('Nnt_parts'):
                            
                            # Calculate the neurotransmitters
                            Nnt_result = self.fj_tf[user] - self.rj_tf[user] * Nnt_list / ( 1 + self.Aj[user] ) \
                                                          - self.mj_tf[user] * Nnt_list / ( 1 + self.Bj[user] ) 

                        with tf.variable_scope('slow_Components'):
                            
                            # Calculate long-term dependencies
                            # This is the NN([ n1, n2, n3, s ])
                            res_ls     = [get_slowVaryingComponents(j, Nnt_list, stressVal, slowComponents) for j in range(self.Nl)]
                            results    = tf.concat([Nnt_result, res_ls], axis=0)

                        return results
                
                # self.rhs_results = [rhs_operation(user) for user in range(self.Npat)]
                # self.rhs_results = tf.concat(self.rhs_results, axis=0)
                self.rhs_results = self.fj_tf - self.rj_tf * self.y_tf / ( 1 + self.Aj ) \
                                              - self.mj_tf * self.y_tf / ( 1 + self.Bj ) 
                self.rhs_results = tf.reshape(self.rhs_results, shape=[-1], name='flattenOps')

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

    # @lD.log(logBase + '.dy')
    def dy(self, y, t):
        '''[summary]
        
        [description]
        
        Arguments:
            y {[type]} -- [description]
            t {[type]} -- [description]
        '''

        try:
        
            rhs_results = self.sess.run( self.rhs_results, 
                                         # options=self.options, run_metadata=self.run_metadata,
                                         feed_dict={
                                             self.y_tf       : np.array(y).reshape(self.Npat, -1),
                                             self.stress_val : np.array([interp(t) for interp in self.stressInterp]).reshape(self.Npat, -1),
                                             self.Aj         : np.array([interp(t) for interp in self.AjInterp]).reshape(self.Npat, -1),
                                             self.Bj         : np.array([interp(t) for interp in self.BjInterp]).reshape(self.Npat, -1)
                                    })

            return rhs_results

        except Exception as e:

            # logger.error('Unable to get dy result \n{}'.format(str(e)))
            print('Unable to get dy result \n{}'.format(str(e)))

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            NNwts, NNb, NNact, Taus  = args

            for i, (weights, bias, tau) in enumerate(zip(NNwts, NNb, Taus)):
                self.sess.run( self.NNwts_tf[i].assign(weights) )
                self.sess.run( self.NNb_tf[i].assign(bias) )
                self.sess.run( self.Taus_tf[i].assign(tau) )

            jac = None
            if useJac:
                jac = self.jac

            start = time.time()

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, Dfun=jac, full_output=True, mxstep=50000)
            else:
                y_t = odeint(self.dy, y0, t, Dfun=jac, full_output=False)
            
            timespent = time.time() - start
            print('odeTime', timespent)

            # fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            # chrome_trace     = fetched_timeline.generate_chrome_trace_format()

            # with open('timeline_step.json', 'w') as f:
            #     f.write(chrome_trace)

            # if useJac:
            #     print('')

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))