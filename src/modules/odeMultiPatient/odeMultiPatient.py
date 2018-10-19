from logs import logDecorator as lD 
from lib.odeModels import multipleODE as mOde
import json
import numpy as np
from time import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy import signal

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.odeMultiPatient.odeMultiPatient'

def dTanh(x):
    return 1-(np.tanh(x))**2

@lD.log(logBase + '.solveODE')
def solveODE(logger, Npat, numSim, filehandler, plotData=False):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''
    now      = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    header   = '-' * 30 + '\n'
    header   = header + '{}'.format(now) + '\n'
    header   = header + 'We are in odeMultiPatient: {}\n'.format(Npat)
    header   = header + '-' * 30 + '\n'

    print(header)
    filehandler.write(header)

    
    tspan    = np.linspace(0, 100, 101)

    # ---------------------------------------------------------
    # These can be specific to a person or to the simulation
    # as a whole ... 
    # ---------------------------------------------------------
    Nnt       = 3
    Nl        = 3
    
    # This is patient-specific stuff ...
    Atimesj   = []
    Btimesj   = []
    fj        = []
    rj        = []
    mj        = []
    stress_v  = []

    for i in range(Npat):
        
        tmp_doseA, tmp_doseB   = np.zeros(shape=tspan.shape), np.zeros(shape=tspan.shape)
        random_scale           = np.random.uniform(low=0.0, high=3.0)
        
        for trange, dose in [ ([  5 * random_scale, 15 * random_scale],    3 * random_scale ),
                              ([ 35 * random_scale, 50 * random_scale],   35 * random_scale ),
                              ([ 50 * random_scale, 60 * random_scale],    3 * random_scale ),
                              ([ 60 * random_scale, 75 * random_scale],  300 * random_scale ),
                              ([ 75 * random_scale, 80 * random_scale],  7.6 * random_scale ) ]:
            twindow            = range(round(trange[0]), min(round(trange[1]) + 1, int(tspan.min())))
            tmp_doseA[twindow] = dose

        random_scale = np.random.uniform(low=0.0, high=3.0)
        for trange, dose in [ ([  5 * random_scale, 15 * random_scale], 70   * random_scale ),
                              ([ 35 * random_scale, 50 * random_scale], 12.5 * random_scale ),
                              ([ 75 * random_scale, 80 * random_scale], 7.6  * random_scale ) ]:
            twindow            = range(round(trange[0]), min(round(trange[1]) + 1, int(tspan.min())))
            tmp_doseB[twindow] = dose

        Atimesj.extend( [tmp_doseA, tmp_doseA, tmp_doseA] )
        Btimesj.extend( [tmp_doseB, tmp_doseB, tmp_doseB] )

        random_scale = np.random.uniform(low=0.0, high=3.0)
        fj.append(np.array([12 * random_scale,  7 * random_scale, 15 * random_scale ]))

        random_scale = np.random.uniform(low=0.0, high=3.0)
        rj.append(np.array([6  * random_scale,  3 * random_scale, 8  * random_scale ]))

        random_scale = np.random.uniform(low=0.0, high=3.0)
        mj.append(np.array([10 * random_scale, 17 * random_scale, 2  * random_scale ]))

        random_scale = np.random.uniform(low=0.0, high=3.0)
        stress_v.extend( [signal.square(2 * np.pi * tspan / (20.0 * random_scale)) * 3, 
                          signal.square(2 * np.pi * tspan / (15.0 * random_scale)) * 3,
                          signal.square(2 * np.pi * tspan / (10.0 * random_scale)) * 3] )

    Atimesj   = np.array(Atimesj).reshape(Npat, 3, -1)
    Btimesj   = np.array(Btimesj).reshape(Npat, 3, -1)
    fj        = np.array(fj).reshape(Npat, -1)
    rj        = np.array(rj).reshape(Npat, -1)
    mj        = np.array(mj).reshape(Npat, -1)
    stress_v  = np.array(stress_v).reshape(Npat, 3, -1)

    model     = mOde.multipleODE(Npat, Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_v, tspan)

    allTimes  = []

    for i in range(numSim):
        
        y0_list      = []

        for j in range(Npat):
            random_scale = np.random.uniform(low=0.0, high=3.0)
            y0_list.append(np.array([1 * random_scale, 
                                     2 * random_scale, 
                                     3 * random_scale, 
                                     3 * random_scale, 
                                     2 * random_scale, 
                                     1 * random_scale ]))

        y0    = np.hstack(y0_list)
        NNwts = [ np.random.rand( 6, 12), 
                  np.random.rand(12,  3),
                  np.random.rand( 3,  3) ]
        NNb    = [ 0, 1, -1 ]
        NNact  = [ np.tanh, np.tanh, np.tanh ]
        NNactD = [ dTanh,   dTanh,   dTanh ] # Differentiation of tanh
        Taus   = np.array([ 1, 4, 12 ] * Npat).reshape(Npat, -1)
        
        args   = (NNwts, NNb, NNact, NNactD, Taus)

        tNew          = np.linspace(5, 75, 50)
        startTime     = time()
        result, specs =  model.solveY( y0, tNew, args, full_output=True )
        tDelta        = time() - startTime
        allTimes.append( tDelta )

        statusUpdate  = '[No Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> {:.6f}'.format(
                            specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1], tDelta)
        print(statusUpdate)
        filehandler.write(statusUpdate + '\n')

    allTimes      = np.array(allTimes)
    summaryUpdate = '[No Jac] Mean = {}, Std  = {}, Nusers = {}, perUser = {}'.format( 
                        allTimes.mean(), allTimes.std(), Npat, allTimes.mean() / Npat )
    print(summaryUpdate)
    filehandler.write(summaryUpdate + '\n')

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


    numList     = []
    filehandler = open('../results/odeint_performance.txt', 'a')

    for Npat in numList:
        solveODE(Npat, 10, filehandler, plotData=False)

    filehandler.close()

    return

