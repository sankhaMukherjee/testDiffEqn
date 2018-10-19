from logs import logDecorator as lD 
from lib.odeModels import multipleODE_tf as mOde
import json
import numpy as np
from time import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

from scipy import signal
import tensorflow as tf

from tqdm import tqdm

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.odeMultiPatient.odeMultiPatient_tf'

@lD.log(logBase + '.solveODE_old')
def solveODE_old(logger, N, numSim, plotData=False):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    print('-'*30)
    print('We are in odeMultiPatient: {}'.format(N))
    print('-'*30)
    now      = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    Npat     = N
    tspan    = np.linspace(0, 100, 101)

    # ---------------------------------------------------------
    # These can be specific to a person or to the simulation
    # as a whole ... 
    # ---------------------------------------------------------
    Nnt      = 3
    Nl       = 3
    
    # This is patient-specific stuff ...
    Atimesj  = []
    Btimesj  = []
    fj       = []
    rj       = []
    mj       = []
    stress_v = []
    stress_t = []

    for i in range(Npat):
        
        tmp_doseA, tmp_doseB   = np.zeros(shape=tspan.shape), np.zeros(shape=tspan.shape)
        
        for trange, dose in [ ([  5, 15],    3 ),
                              ([ 35, 50],   35 ),
                              ([ 50, 60],    3 ),
                              ([ 60, 75],  300 ),
                              ([ 75, 80],  7.6 ) ]:
            twindow            = range(trange[0], trange[1] + 1)
            tmp_doseA[twindow] = dose

        for trange, dose in [ ([  5, 15], 70   ),
                              ([ 35, 50], 12.5 ),
                              ([ 75, 80], 7.6  ) ]:
            twindow            = range(trange[0], trange[1] + 1)
            tmp_doseB[twindow] = dose   

        Atimesj.append(tmp_doseA)
        Btimesj.append(tmp_doseB)
        fj.append(np.array([12   ,7   ,15 ]))
        rj.append(np.array([6    ,3   ,8  ]))
        mj.append(np.array([10   ,17  ,2  ]))
        stress_t.append( tspan.copy() )
        stress_v.append( signal.square(2 * np.pi * tspan / 20.0) * 50)

    model       = mOde.multipleODE(Npat=Npat, Nnt=Nnt, Nl=Nl, tspan=tspan, Atimesj=Atimesj, Btimesj=Btimesj, 
                                   fj=fj, rj=rj, mj=mj, stress_t=stress_t, stress_v=stress_v, 
                                   layers=[12, 3, 1], activations=[ 'tanh', 'tanh', 'tanh' ])
    allTimes    = []
    allTimesJac = []
    allResults  = []
    pbar        = tqdm(range(numSim))
    np.random.seed(1234)

    for i in pbar:

        y0     = np.hstack([np.array([1,1,1,2,2,2]) for j in range(Npat)])

        NNwts  = [ np.random.random(size=(12, 4)), 
                   np.random.random(size=(3, 12)), 
                   np.random.random(size=(1, 3))
                   ]
        NNb    = [ 0, 1, -1 ]
        NNact  = [ 'tanh', 'tanh', 'tanh' ]
        Taus   = [ 1, 4, 12 ]

        args   = (NNwts, NNb, NNact, Taus)

        tNew          = np.linspace(5, 75, 50)
        startTime     = time()
        result, specs =  model.solveY( y0, tNew, args, full_output=True )
        tDelta        = time() - startTime
        allTimes.append( tDelta )
        pbar.set_description('time spent: {:.3f}s'.format(tDelta))

        '''
        with open('../results/allData.csv', 'a') as f:
            f.write('[No Jacobian][{}], # steps {:6d}, fn eval {:6d}, jac eval {:6d}, {}\n '.format(
            now, specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1], tDelta))
        '''

        print('[No Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> '.format(
            specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1]), end = '')
        print(tDelta)

        allResults.append(result)

        for key in specs.keys():
            if isinstance(specs[key], np.ndarray):
                specs[key] = specs[key].tolist()

        with open('../results/odeint_TFspecs.json', 'w') as f:
            json.dump(specs, f)

        # startTime = time()
        # result_1, specs_1 =  model.solveY( y0, tNew, args, useJac=True, full_output=True )
        # tDelta = time() - startTime

        # if i > 0:
        #     allTimesJac.append( tDelta )

        # print('[   Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> '.format(
        #     specs_1['nst'][-1], specs_1['nfe'][-1], specs_1['nje'][-1]), end = '')
        # print(tDelta)

        # error = np.mean(np.abs(result - result_1))
        # print('error = {}'.format(error))

        # if plotData:
        #     now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        #     for i in range(result.shape[1]):
        #         plt.figure()
        #         plt.plot(tNew, result[:, i])
        #         plt.plot(tNew, result_1[:, i])
        #         plt.savefig('../results/img/simpleODE-{}_{:05}.png'.format(now, i))
        #     plt.close('all')

    allResults = np.concatenate(allResults, axis=0)
    np.save('../results/allresults_dy_tensorflow.npy', allResults)

    allTimes = np.array(allTimes)
    # allTimesJac = np.array(allTimesJac)
    print('[No Jac] Mean = {}, Std  = {}, Nusers = {}, perUser = {}'.format( allTimes.mean(), allTimes.std(), Npat, allTimes.mean()/Npat ))
    # print('[   Jac] Mean = {}, Std  = {}'.format( allTimesJac.mean(), allTimesJac.std() ))
    
    '''    
    with open('../results/allData.csv', 'a') as f:
        f.write('[Summary][No Jac], Mean = {}, Std  = {}, Nusers = {}, perUser = {}\n'.format( 
            allTimes.mean(), allTimes.std(), Npat, allTimes.mean()/Npat ))
    '''

    return allTimes.mean()/Npat

@lD.log(logBase + '.solveODE_new')
def solveODE_new(logger, N, numSim, plotData=False):

    print('-'*30)
    print('We are in odeMultiPatient: {}'.format(N))
    print('-'*30)
    now      = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    Npat     = N
    tspan    = np.linspace(0, 100, 101)

    Nnt      = 3
    Nl       = 3
    
    Atimesj  = []
    Btimesj  = []
    fj       = []
    rj       = []
    mj       = []
    stress_v = []
    stress_t = []

    for i in range(Npat):
        
        tmp_doseA, tmp_doseB   = np.zeros(shape=tspan.shape), np.zeros(shape=tspan.shape)
        
        for trange, dose in [ ([  5, 15],    3 ),
                              ([ 35, 50],   35 ),
                              ([ 50, 60],    3 ),
                              ([ 60, 75],  300 ),
                              ([ 75, 80],  7.6 ) ]:
            twindow            = range(trange[0], trange[1] + 1)
            tmp_doseA[twindow] = dose

        for trange, dose in [ ([  5, 15], 70   ),
                              ([ 35, 50], 12.5 ),
                              ([ 75, 80], 7.6  ) ]:
            twindow            = range(trange[0], trange[1] + 1)
            tmp_doseB[twindow] = dose   

        Atimesj.append(tmp_doseA)
        Btimesj.append(tmp_doseB)
        fj.append(np.array([12   ,7   ,15 ]))
        rj.append(np.array([6    ,3   ,8  ]))
        mj.append(np.array([10   ,17  ,2  ]))
        stress_t.append( tspan.copy() )
        stress_v.append( signal.square(2 * np.pi * tspan / 20.0) * 50 )

    Atimesj     = np.array(Atimesj).reshape(Npat, -1)
    Btimesj     = np.array(Btimesj).reshape(Npat, -1)
    fj          = np.array(fj).reshape(Npat, -1)
    rj          = np.array(rj).reshape(Npat, -1)
    mj          = np.array(mj).reshape(Npat, -1)

    model       = mOde.multipleODE_new(Npat=Npat, Nnt=Nnt, Nl=Nl, tspan=tspan, Atimesj=Atimesj, Btimesj=Btimesj, 
                                       fj=fj, rj=rj, mj=mj, stress_t=stress_t, stress_v=stress_v, 
                                       layers=[12, 3, 1], activations=[ 'tanh', 'tanh', 'tanh' ])
    allTimes    = []
    allTimesJac = []
    allResults  = []
    pbar        = tqdm(range(numSim))
    np.random.seed(1234)

    for i in pbar:

        y0     = np.hstack([np.array([1,1,1]) for j in range(Npat)])

        NNwts  = [ np.random.random(size=(12, 4)), 
                   np.random.random(size=(3, 12)), 
                   np.random.random(size=(1, 3))
                   ]
        NNb    = [ 0, 1, -1 ]
        NNact  = [ 'tanh', 'tanh', 'tanh' ]
        Taus   = [ 1, 4, 12 ]

        args   = (NNwts, NNb, NNact, Taus)

        tNew          = np.linspace(5, 75, 50)
        startTime     = time()
        result, specs =  model.solveY( y0, tNew, args, full_output=True )
        tDelta        = time() - startTime
        allTimes.append( tDelta )
        pbar.set_description('time spent: {:.3f}s'.format(tDelta))

        '''
        with open('../results/allData.csv', 'a') as f:
            f.write('[No Jacobian][{}], # steps {:6d}, fn eval {:6d}, jac eval {:6d}, {}\n '.format(
            now, specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1], tDelta))
        '''

        print('[No Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> '.format(
            specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1]), end = '')
        print(tDelta)

        allResults.append(result)

        for key in specs.keys():
            if isinstance(specs[key], np.ndarray):
                specs[key] = specs[key].tolist()

        with open('../results/odeint_TFspecs.json', 'w') as f:
            json.dump(specs, f)

        # startTime = time()
        # result_1, specs_1 =  model.solveY( y0, tNew, args, useJac=True, full_output=True )
        # tDelta = time() - startTime

        # if i > 0:
        #     allTimesJac.append( tDelta )

        # print('[   Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> '.format(
        #     specs_1['nst'][-1], specs_1['nfe'][-1], specs_1['nje'][-1]), end = '')
        # print(tDelta)

        # error = np.mean(np.abs(result - result_1))
        # print('error = {}'.format(error))

        # if plotData:
        #     now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        #     for i in range(result.shape[1]):
        #         plt.figure()
        #         plt.plot(tNew, result[:, i])
        #         plt.plot(tNew, result_1[:, i])
        #         plt.savefig('../results/img/simpleODE-{}_{:05}.png'.format(now, i))
        #     plt.close('all')

    allResults = np.concatenate(allResults, axis=0)
    np.save('../results/allresults_dy_tensorflow.npy', allResults)

    allTimes = np.array(allTimes)
    # allTimesJac = np.array(allTimesJac)
    print('[No Jac] Mean = {}, Std  = {}, Nusers = {}, perUser = {}'.format( allTimes.mean(), allTimes.std(), Npat, allTimes.mean()/Npat ))
    # print('[   Jac] Mean = {}, Std  = {}'.format( allTimesJac.mean(), allTimesJac.std() ))
    
    '''    
    with open('../results/allData.csv', 'a') as f:
        f.write('[Summary][No Jac], Mean = {}, Std  = {}, Nusers = {}, perUser = {}\n'.format( 
            allTimes.mean(), allTimes.std(), Npat, allTimes.mean()/Npat ))
    '''

    return allTimes.mean()/Npat

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


    # numList = [1, 5, 10, 20, 50, 100]
    numList = [1]
    # results = [solveODE(N, 10, plotData=False) for N in numList]

    for N in numList:
        print('N:', N)
        results = solveODE_new(N, 1, plotData=False)
        now     = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    '''
    with open('../results/summary-{}.csv'.format(now), 'w') as f:
        for n, l in zip(numList, results):
            f.write('{},{}\n'.format( n, l ))
    '''

        
    # compareJac()

    return