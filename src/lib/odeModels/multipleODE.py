from logs import logDecorator as lD 
import json

from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np

# from numba import jit

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.multipleODE.multipleODE'


class multipleODE_old:
    '''[summary]
    
    [description]
    '''


    @lD.log(logBase + '.__init__')
    def __init__(logger, self, Npat, Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        self.Npat    = Npat  # --> 1 number 
        self.Nnt     = Nnt   # --> 1 number
        self.Nl      = Nl    # --> 1 number
        self.Atimesj = Atimesj # --> Npat arrays
        self.Btimesj = Btimesj # --> Npat arrays
        self.fj      = fj      # --> Npat arrays
        self.rj      = rj      # --> Npat arrays
        self.mj      = mj      # --> Npat arrays
        self.stress  = []      # --> Npat functions
        for s_t, s_v in zip(stress_t, stress_v):
            self.stress.append( interp1d(s_t, s_v) )

        return

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
    def dy(logger, self, y, t, NNwts, NNb, NNact, NNactD, Taus):
        '''[summary]
        

        Current Assumptions:

        1. At the moment, we assume that a single NN will suffice for the 
           entire model. This will not always be the case. However, we shall
           calculate this at every iteration of the latent space to compute
           the cost having a number of neural network layers

        2. There is only 1 vector for stress that will represent the stressors.
           This can easily be changed to a list of stressors. Medications can
           also be inserted in the same manner later. 
           We might also be interested in smoothing eevrything later.


        
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
        Taus : {[type]}
            [description]
        
        Returns
        -------
        [type]
            [description]
        '''
       
        NperUser = self.Nnt + self.Nl
        result   = np.zeros( NperUser * self.Npat )

        try:

            # Add this per user
            for user in range(self.Npat):
                
                # Calculate the neurotransmitters
                for j in range(self.Nnt):

                    Aj = self.AjFunc(t, self.Atimesj[user])
                    Bj = self.BjFunc(t, self.Btimesj[user])

                    v  = self.fj[user][j]
                    v -= self.rj[user][j]*y[user*NperUser + j]/( 1 + Aj )
                    v -= self.mj[user][j]*y[user*NperUser + j]/( 1 + Bj )

                    result[ user*NperUser + j] = v

                # Calculate long-term dependencies
                for j in range(self.Nl):

                    # This is the NN([ n1, n2, n3, s ])
                    res = np.hstack(( 
                        y[ user*NperUser: user*NperUser+self.Nnt], 
                        np.array([self.stress[user](t)]) 
                        ))
                    res = res.reshape((-1, 1))

                    for index, (w, b, a) in enumerate(zip(NNwts, NNb, NNact)):
                        res = np.matmul(w, res) + b
                        res = a(res)

                    result[user*NperUser+self.Nnt+j] = res[0][0] - y[user*NperUser+self.Nnt+j]/Taus[j]

        except Exception as e:
            
            logger.error('Unable to calculate the dy properly: {}'.format(str(e)))

        return np.array(result)

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            jac = None
            if useJac:
                jac = self.jac

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, args=args, Dfun=jac, full_output=True)
            else:
                y_t = odeint(self.dy, y0, t, args=args, Dfun=jac, full_output=False)

            # if useJac:
            #     print('')

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))

class multipleODE:
    '''[summary]
    
    [description]
    '''


    @lD.log(logBase + '.__init__')
    def __init__(logger, self, Npat, Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_v, tspan):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        self.Npat         = Npat    # --> 1 number 
        self.Nnt          = Nnt     # --> 1 number
        self.Nl           = Nl      # --> 1 number
        self.fj           = fj      # --> Npat arrays
        self.rj           = rj      # --> Npat arrays
        self.mj           = mj      # --> Npat arrays

        self.AjInterp     = interp1d(tspan, Atimesj,  bounds_error=False, fill_value=(Atimesj[:,:,0],  Atimesj[:,:,-1]))
        self.BjInterp     = interp1d(tspan, Btimesj,  bounds_error=False, fill_value=(Btimesj[:,:,0],  Btimesj[:,:,-1]))
        self.stressInterp = interp1d(tspan, stress_v, bounds_error=False, fill_value=(stress_v[:,:,0], stress_v[:,:,-1]))

        return

    @lD.log(logBase + '.dy')
    def dy(logger, self, y, t, NNwts, NNb, NNact, NNactD, Taus):
        '''[summary]
        

        Current Assumptions:

        1. At the moment, we assume that a single NN will suffice for the 
           entire model. This will not always be the case. However, we shall
           calculate this at every iteration of the latent space to compute
           the cost having a number of neural network layers

        2. There is only 1 vector for stress that will represent the stressors.
           This can easily be changed to a list of stressors. Medications can
           also be inserted in the same manner later. 
           We might also be interested in smoothing eevrything later.


        
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
        Taus : {[type]}
            [description]
        
        Returns
        -------
        [type]
            [description]
        '''

        try:

            y           = np.array(y).reshape(self.Npat, -1)
            Nnt_val     = y[:, :self.Nnt]
            nn_inputs   = np.concatenate([Nnt_val, self.stressInterp(t)], axis=1)

            for index, (w, b, a) in enumerate(zip(NNwts, NNb, NNact)):
                if index == 0:
                    nn_res = np.dot(nn_inputs, w) + b
                else:
                    nn_res = np.dot(nn_res, w) + b
                nn_res = a(nn_res)

            nn_res    = nn_res - y[:, self.Nnt:] / Taus
            meds_res  = self.fj - self.rj * Nnt_val / (1 + self.AjInterp(t)) - self.mj * Nnt_val / (1 + self.BjInterp(t))

            result    = np.concatenate([nn_res, meds_res], axis=1)
            result    = result.flatten()

        except Exception as e:
            
            logger.error('Unable to calculate the dy properly: {}'.format(str(e)))

        return result

    @lD.log(logBase + '.solveY')
    def solveY(logger, self, y0, t, args, useJac=False, full_output=False):

        try:
            jac = None
            if useJac:
                jac = self.jac

            result_dict = {}
            if full_output:
                y_t, result_dict = odeint(self.dy, y0, t, args=args, Dfun=jac, full_output=True, mxstep=50000)
            else:
                y_t = odeint(self.dy, y0, t, args=args, Dfun=jac, full_output=False)

            return y_t, result_dict

        except Exception as e:

            logger.error('Unable to solve Y \n{}'.format(str(e)))
