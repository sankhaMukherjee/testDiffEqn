from logs import logDecorator as lD 
import json

from scipy.integrate import solve_ivp
import numpy as np

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.odeModels.simpleODE'


class simpleODE:
    '''[summary]
    
    [description]
    '''

    @lD.log(logBase + '.__init__')
    def __init__(logger, self):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        return


    @staticmethod
    def Aj(t, Atimes):
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



        return val

    @staticmethod
    def Bj(t):
        val = 2
        return

    @staticmethod
    def dy(t, y):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        t : {float}
            a time vector over which the solution must be presented
        y : {float}
            vector for y at a given time ``t``
        '''



        return