from logs import logDecorator as lD 
import json

from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np
import tensorflow as tf

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.multipleODE.multipleODE_tf'


class multipleODE:
    '''[summary]
    
    [description]
    '''


    @lD.log(logBase + '.__init__')
    def __init__(logger, self, Npat, Nnt, Nl, tspan, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v):
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
            self.Atimesj  = Atimesj  # --> Npat arrays
            self.Btimesj  = Btimesj  # --> Npat arrays
            self.fj       = fj       # --> Npat arrays
            self.rj       = rj       # --> Npat arrays
            self.mj       = mj       # --> Npat arrays

            self.stress_t = stress_t
            self.stress_v = stress_v

        except Exception as e:
            logger.error('Unable to initialize multipleODE \n{}'.format(str(e)))

    @lD.log(logBase + '.tf_opsFlow')
    def tf_opsFlow(logger, self):

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
                        res                    = tf.reshape(res, [1, -1])

                        for w, b, a in zip(self.NNwts_tf, self.NNb_tf, self.NNact_tf):
                            res = tf.matmul(res, w) + b
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

    @staticmethod
    def AjFunc(t, Atimes):
        '''Find one time-dependent constant
        
        This is the measure of the effectiveness of all the
        drugs during a a prticular period.
        
        Parameters
        ----------
        t : {float}
            The time component for a particular value
        Atimes : {list of 3-tuples}
            The tuples represent the starttime, stoptime and
            a set of values. This will allow the tuples to be 
            calculated for a particular time.
        
        Returns
        -------
        np.array
            The value of the 
        '''
        val = 0

        for t1, t2, v in Atimes:
            if (t>=t1) and (t<t2):
                return v

        return val

    @staticmethod
    def BjFunc(t, Btimes):
        '''Find one time-dependent constant
        
        This is the measure of the effectiveness of all the
        drugs during a a prticular period.
        
        Parameters
        ----------
        t : {float}
            The time component for a particular value
        Atimes : {list of 3-tuples}
            The tuples represent the starttime, stoptime and
            a set of values. This will allow the tuples to be 
            calculated for a particular time.
        
        Returns
        -------
        np.array
            The value of the 
        '''
        val = 0

        for t1, t2, v in Btimes:
            if (t>=t1) and (t<t2):
                return v

        return val

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
                                                self.t : t
                                        })

            return rhs_results

        except Exception as e:

            logger.error('Unable to get dy result \n{}'.format(str(e)))

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            NNwts, NNb, NNact, Taus = args
            activation_map          = {
                                            'tanh'    : tf.nn.tanh,
                                            'sigmoid' : tf.nn.sigmoid,
                                            'relu'    : tf.nn.relu,
                                            'linear'  : tf.identity
                                      }

            self.NNwts_tf           = [  tf.constant(wts, dtype=tf.float32, name='wts_{}'.format(index)) 
                                               for index, wts in enumerate(NNwts)]
            self.NNb_tf             = [  tf.constant(b, dtype=tf.float32, name='bias_{}'.format(index)) 
                                               for index, b in enumerate(NNb)]
            self.NNact_tf           = [  activation_map[a] 
                                               for a in NNact]
            self.Taus_tf            = [  tf.constant(t, dtype=tf.float32, name='tau_{}'.format(index)) 
                                               for index, t in enumerate(Taus)]

            self.tf_opsFlow()

            jac = None
            if useJac:
                jac = self.jac

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, Dfun=jac, full_output=True)
            else:
                y_t = odeint(self.dy, y0, t, Dfun=jac, full_output=False)

            # if useJac:
            #     print('')

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))

