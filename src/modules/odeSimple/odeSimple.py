from logs import logDecorator as lD 
from lib.odeModels import simpleODE as sOde
import json
import numpy as np


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.odeSimple.odeSimple'


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    print('We are in odeSimple')
    t = np.linspace(0, 100, 101)

    # ---------------------------------------------------------
    # These can be specific to a person or to the simulation
    # as a whole ... 
    # ---------------------------------------------------------
    Nnt      = 3
    Nl       = 3
    Atimesj  = [(  5, 15, 3    ),
                ( 35, 50, 12.5 ),
                ( 75, 80, 7.6  ),]
    Btimesj  = [(  5, 15, 3    ),
                ( 35, 50, 12.5 ),
                ( 75, 80, 7.6  ),]
    fj       = [2   ,2   ,2    ]
    rj       = [0.5 ,0.5 ,0.5  ]
    mj       = [0.5 ,0.5 ,0.5  ]
    stress_t = t.copy()
    stress_v = np.random.rand(len(t))
    stress_v = np.zeros(len(t))

    model = sOde.simpleODE(Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v)

    y0    = np.array([1,1,1,2,2,2])
    NNwts = [ np.random.rand(12,  4), 
              np.random.rand( 3, 12),
              np.random.rand( 1,  3) ]
    NNb   = [ 0, 1, -1 ]
    NNact = [ np.tanh, np.tanh, np.tanh ]
    Taus  = [1, 4, 12]

    args = (NNwts, NNb, NNact, Taus)


    tNew = np.linspace(5, 75, 100)
    result, specs =  model.solveY( y0, tNew, args )

    
    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    doSomething()

    return

