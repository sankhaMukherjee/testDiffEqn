from logs import logDecorator as lD 
from lib.odeModels import multipleODE as mOde
import json
import numpy as np
from time import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

from scipy import signal




config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.odeMultiPatient.odeMultiPatient'

def dTanh(x):
    return 1-(np.tanh(x))**2

def one(x):
    return x

def sqr(x):
    return x**2

def Dsqr(x):
    return x*2

# def compareJac():

#     np.set_printoptions(precision=1)

#     t = np.linspace(0, 100, 101)

#     # ---------------------------------------------------------
#     # These can be specific to a person or to the simulation
#     # as a whole ... 
#     # ---------------------------------------------------------
#     Nnt      = 3
#     Nl       = 3
#     Atimesj  = [(  5, 15, 3    ),
#                 ( 35, 50, 12.5 ),
#                 ( 75, 80, 7.6  ),]
#     Btimesj  = [(  5, 15, 3    ),
#                 ( 35, 50, 12.5 ),
#                 ( 75, 80, 7.6  ),]
#     fj       = [2   ,2   ,2    ]
#     rj       = [0.5 ,0.5 ,0.5  ]
#     mj       = [0.5 ,0.5 ,0.5  ]
#     stress_t = t.copy()
#     stress_v = np.random.rand(len(t))
#     stress_v = np.zeros(len(t))


#     model = sOde.simpleODE(Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v)
#     y0    = np.array([1.0,1,1,2,2,2])
#     NNwts = [ np.random.rand(12,  4), 
#               np.random.rand( 3, 12),
#               np.random.rand( 1,  3) ]
#     NNb    = [ 0, 1, -1 ]

#     NNact  = [ np.tanh, np.tanh, np.tanh ]
#     NNactD = [ dTanh,   dTanh,   dTanh ] # Differentiation of tanh

#     # NNact  = [ one, one, sqr ]
#     # NNactD = [ one, one, Dsqr ] # Differentiation of tanh


#     Taus   = [1, 4, 12]

#     i  = 0
#     delY0 = 1e-10


#     dy  = model.dy(y0, 0, NNwts, NNb, NNact, NNactD, Taus)
#     jac = model.jac(y0, 0, NNwts, NNb, NNact, NNactD, Taus)

#     jacFD = []
#     for i in range(len(y0)):
#         y1    = y0.copy()
#         y1[i] = y1[i] + delY0

#         dy1 = model.dy(y1, 0, NNwts, NNb, NNact, NNactD, Taus)

#         jacFD.append((dy1 - dy)/delY0)

#     jacFD = np.array(jacFD)
#     print('------------[Finite difference]------------')
#     print(jacFD)
#     print('------------[Calculated]------------')
#     print(jac)
#     print('------------[Ratio]------------')
#     print(jac/jacFD)


#     return

def r():
    return np.random.rand() + 0.5

@lD.log(logBase + '.solveODE')
def solveODE(logger, N, numSim, plotData=False):
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
    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    Npat = N
    t = np.linspace(0, 100, 101)

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
        

        Atimesj.append(    [(  5, 15,    3 ),
                            ( 35, 50,   35 ),
                            ( 50, 60,    3 ),
                            ( 60, 75,  300 ),
                            ( 75, 80,  7.6 ),]
                        )

        Btimesj.append(    [(  5, 15, 70    ),
                            ( 35, 50, 12.5  ),
                            ( 75, 80, 7.6   ),]
                        )


        fj.append(np.array([12   ,7   ,15    ]))
        rj.append(np.array([6    ,3   ,8  ]))
        mj.append(np.array([10 ,17 ,2  ]))
        stress_t.append(t.copy())
        stress_v.append( signal.square(2 * np.pi * t / 20.0) * 50)


    model = mOde.multipleODE_old(Npat, Nnt, Nl, Atimesj, Btimesj, fj, rj, mj, stress_t, stress_v)

    allTimes    = []
    allTimesJac = []
    allResults  = []
    np.random.seed(1234)

    for i in range(numSim):

        y0    = np.hstack([np.array([1,1,1,2,2,2]) for j in range(Npat)])

        NNwts = [ np.random.rand(12,  4), 
                  np.random.rand( 3, 12),
                  np.random.rand( 1,  3) ]
        NNb    = [ 0, 1, -1 ]
        NNact  = [ np.tanh, np.tanh, np.tanh ]
        NNactD = [ dTanh,   dTanh,   dTanh ] # Differentiation of tanh
        Taus   = [1, 4, 12]

        args   = (NNwts, NNb, NNact, NNactD, Taus)

        tNew          = np.linspace(5, 75, 50)
        startTime     = time()
        result, specs =  model.solveY( y0, tNew, args, full_output=True )
        tDelta        = time() - startTime
        allTimes.append( tDelta )

        '''
        with open('../results/allData.csv', 'a') as f:
            f.write('[No Jacobian][{}], # steps {:6d}, fn eval {:6d}, jac eval {:6d}, {}\n '.format(
            now, specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1], tDelta))
        '''

        print('[No Jacobian] # steps {:6d}, fn eval {:6d}, jac eval {:6d} --> '.format(
            specs['nst'][-1], specs['nfe'][-1], specs['nje'][-1]), end = '')
        print(tDelta)

        # allResults.append(result)

        # for key in specs.keys():
        #     if isinstance(specs[key], np.ndarray):
        #         specs[key] = specs[key].tolist()

        # with open('../results/odeint_Numpyspecs.json', 'w') as f:
        #     json.dump(specs, f)

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


    # allResults = np.concatenate(allResults, axis=0)
    # np.save('../results/allresults_dy_numpy.npy', allResults)

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


    numList = [1, 5, 10, 20, 50, 100]
    # results = [solveODE(N, 10, plotData=False) for N in numList]
    results = [solveODE(N, 10, plotData=False) for N in numList]
    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    '''
    with open('../results/summary-{}.csv'.format(now), 'w') as f:
        for n, l in zip(numList, results):
            f.write('{},{}\n'.format( n, l ))
    '''

        
    # compareJac()

    return

