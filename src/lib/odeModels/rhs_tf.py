import tensorflow as tf
import numpy as np

from scipy import signal

def interpolate( dx_T, dy_T, x, name='interpolate' ):
        
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

if __name__ == '__main__':

    Npat     = 2
    Nnt      = 3
    Nl       = 3
    NperUser = Nnt + Nl
    NNwts    = [ np.random.random(size=(4, 12)), 
                 np.random.random(size=(12, 3)), 
                 np.random.random(size=(3, 1))
                 ]
    NNb      = [ 0, 1, -1 ]
    NNact    = [ 'tanh', 'tanh', 'tanh' ]
    Taus     = [ 1, 4, 12 ]

    tspan    = np.linspace(0, 100, 101)
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

    activation_map  = {
                            'tanh'    : tf.nn.tanh,
                            'sigmoid' : tf.nn.sigmoid,
                            'relu'    : tf.nn.relu,
                            'linear'  : tf.identity
                      }

    with tf.variable_scope('Constant'):
        
        tspan_tf    = tf.constant(tspan, dtype=tf.float32, name='tspan')
        Atimesj_tf  = [tf.constant(dose_vec, dtype=tf.float32, name='doseA_vec_{}'.format(index)) for index, dose_vec in enumerate(Atimesj)]
        Btimesj_tf  = [tf.constant(dose_vec, dtype=tf.float32, name='doseB_vec_{}'.format(index)) for index, dose_vec in enumerate(Btimesj)]
        fj_tf       = [tf.constant(fj_vec, dtype=tf.float32, name='fj_{}'.format(index)) for index, fj_vec in enumerate(fj)]
        rj_tf       = [tf.constant(rj_vec, dtype=tf.float32, name='rj_{}'.format(index)) for index, rj_vec in enumerate(rj)]
        mj_tf       = [tf.constant(mj_vec, dtype=tf.float32, name='mj_{}'.format(index)) for index, mj_vec in enumerate(mj)]
        stress_t_tf = [tf.constant(stress_tvec, dtype=tf.float32, name='stress_tvec_{}'.format(index)) for index, stress_tvec in enumerate(stress_t)]
        stress_v_tf = [tf.constant(stress_vvec, dtype=tf.float32, name='stress_vvec_{}'.format(index)) for index, stress_vvec in enumerate(stress_v)]
        NNwts_tf    = [tf.constant(wts, dtype=tf.float32, name='wts_{}'.format(index)) for index, wts in enumerate(NNwts)]
        NNb_tf      = [tf.constant(b, dtype=tf.float32, name='bias_{}'.format(index)) for index, b in enumerate(NNb)]
        NNact_tf    = [activation_map[a] for a in NNact]
        Taus_tf     = [tf.constant(t, dtype=tf.float32, name='tau_{}'.format(index)) for index, t in enumerate(Taus)]

    t           = [0.234]
    y           = [1] * Npat * NperUser
    y_tf        = tf.constant(y, dtype=tf.float32, name='y')
    all_results = []

    for user in range(Npat):

        Aj = interpolate(tspan_tf, Atimesj_tf[user], t)
        Bj = interpolate(tspan_tf, Btimesj_tf[user], t)
        
        # Calculate the neurotransmitters
        result_neurotransmitters = fj_tf[user] - rj_tf[user] * y_tf[( user * NperUser) : ( user * NperUser + Nnt)] / ( 1 + Aj ) \
                                               - mj_tf[user] * y_tf[( user * NperUser) : ( user * NperUser + Nnt)] / ( 1 + Bj ) 


        # Calculate long-term dependencies
        # This is the NN([ n1, n2, n3, s ])

        res_ls = []

        for j in range(Nl):

            # Extract [n1, n2, n3]
            neurotransmitters_list = y_tf[ (user*NperUser) : (user*NperUser + Nnt)]

            # get interpolated s at t
            stress_value           = interpolate(stress_t_tf[user], stress_v_tf[user], t)

            # concatenate to [ n1, n2, n3, s ]
            res                    = tf.concat([neurotransmitters_list, [stress_value]], axis=0)
            res                    = tf.reshape(res, [1, -1])

            for w, b, a in zip(NNwts_tf, NNb_tf, NNact_tf):
                res = tf.matmul(res, w) + b
                res = a(res)

            res  = res[0][0] - y_tf[ user * NperUser + Nnt + j] / Taus_tf[j]
            res_ls.append(res)

        results     = tf.concat([result_neurotransmitters, res_ls], axis=0)
        all_results.append(results)

    all_results = tf.concat(all_results, axis=0)