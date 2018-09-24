from logs import logDecorator as lD 
import json

from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.odeModels.simpleODE'


class simpleODE:
    '''[summary]
    
    [description]
    '''


    @lD.log(logBase + '.__init__')
    def __init__(logger, self, Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        self.Nnt     = Nnt
        self.Nl      = Nl
        self.Atimesj = Atimesj
        self.Btimesj = Btimesj
        self.fj      = fj
        self.rj      = rj
        self.mj      = mj
        self.stress  = interp1d(stress_t, stress_v)

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

    def dy(self, y, t, NNwts, NNb, NNact, Taus):
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
       
        result = np.zeros(self.Nnt + self.Nl)

        try:
            # Calculate the neurotransmitters
            for j in range(self.Nnt):
                Aj = self.AjFunc(t, self.Atimesj)
                Bj = self.BjFunc(t, self.Btimesj)

                v  = self.fj[j]
                v -= self.rj[j]*y[j]/( 1 + Aj )
                v -= self.mj[j]*y[j]/( 1 + Bj )

                result[j] = v

            # Calculate long-term dependencies
            for j in range(self.Nl):
                res = np.hstack((y[ : self.Nnt], np.array([self.stress(t)]) ))
                res = res.reshape((-1, 1))
                
                for w, b, a in zip(NNwts, NNb, NNact):
                    res = np.matmul(w, res) #+ b
                    res = a(res)

                result[j+self.Nnt] = res[0][0] - y[j+self.Nnt]/Taus[j]

        except Exception as e:
            print('Unable to calculate the dy properly: {}'.format(e))

        return np.array(result)

    def solveY(self, y0, t, args, useJac=False, full_output=False):

        result_dict = {}
        if full_output:
            y_t, result_dict = odeint(self.dy, y0, t, args=args, full_output=True)
        else:
            y_t = odeint(self.dy, y0, t, args=args, full_output=False)

        return y_t, result_dict

