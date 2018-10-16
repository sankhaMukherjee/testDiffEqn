import os

import tensorflow as tf
import numpy as np
from scipy.integrate import odeint

from tensorflow.python.client import timeline
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_simulation_slice(nUser):

    print('running simulation for nUser', nUser)
    tf.reset_default_graph()

    # options      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    start        = time.time()
    with tf.device('/device:GPU:0'):

        y_tf      = tf.placeholder(dtype=tf.float32, name='y_tf')
        t_tf      = tf.placeholder(dtype=tf.float32, name='t_tf')
        bigMatrix = tf.Variable(np.random.random(size=(250 * nUser, 250)), dtype=tf.float32, name='bigMatrix')
        outputls  = []

        for user in range(nUser):
            with tf.variable_scope('user{:03}'.format(user)):
                with tf.variable_scope('matrix_multiplication'):
                    for i in range(100):
                        if i == 0:
                            output = tf.sigmoid(bigMatrix[(250*user):(250*user+250), :] * bigMatrix[(250*user):(250*user+250), :])
                        else:
                            output = tf.sigmoid(output * bigMatrix[(250*user):(250*user+250), :])
                    outputls.append(y_tf + output[0][0])
            
        init     = tf.global_variables_initializer()
    
    Graphtime = time.time() - start
    config    = tf.ConfigProto(gpu_options={'allow_growth':True})
    sess      = tf.Session(config=config)
    sess.run( init )
    tfwriter  = tf.summary.FileWriter('./tensorlog/simulation_slice/nUser{}/'.format(nUser), sess.graph)
    tfwriter.close()

    def rhs(y, t):
        
        # output = sess.run(outputls, feed_dict={
        #     y_tf : y,
        #     t_tf : t
        # }, options=options, run_metadata=run_metadata)

        output = sess.run(outputls, feed_dict={
            y_tf : y,
            t_tf : t
        })
        
        return -y

    y0        = 1.0
    tspan     = np.linspace(0, 100, 101)

    start     = time.time()
    y         = odeint(rhs, y0, tspan)
    ODEtime   = time.time() - start

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace     = fetched_timeline.generate_chrome_trace_format()

    # with open('timeline_step_[nUser_{}].json'.format(nUser), 'w') as f:
    #     f.write(chrome_trace)

    sess.close()

    return (Graphtime, ODEtime)

def run_simulation_list(nUser):

    print('running simulation for nUser', nUser)
    tf.reset_default_graph()

    # options      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    start        = time.time()
    with tf.device('/device:GPU:0'):

        y_tf      = tf.placeholder(dtype=tf.float32, name='y_tf')
        t_tf      = tf.placeholder(dtype=tf.float32, name='t_tf')
        bigMatrix = [tf.Variable(np.random.random(size=(250, 250)), dtype=tf.float32, name='bigMatrix_{}'.format(i)) for i in range(nUser)]
        outputls  = []

        for user in range(nUser):
            with tf.variable_scope('user{:03}'.format(user)):
                with tf.variable_scope('matrix_multiplication'):
                    for i in range(100):
                        if i == 0:
                            output = tf.sigmoid(bigMatrix[user] * bigMatrix[user])
                        else:
                            output = tf.sigmoid(output * bigMatrix[user])
                    outputls.append(y_tf + output[0][0])
            
        init     = tf.global_variables_initializer()
    
    Graphtime = time.time() - start
    config    = tf.ConfigProto(gpu_options={'allow_growth':True})
    sess      = tf.Session(config=config)
    sess.run( init )
    tfwriter  = tf.summary.FileWriter('./tensorlog/simulation_list/nUser{}/'.format(nUser), sess.graph)
    tfwriter.close()

    def rhs(y, t):
        
        # output = sess.run(outputls, feed_dict={
        #     y_tf : y,
        #     t_tf : t
        # }, options=options, run_metadata=run_metadata)

        output = sess.run(outputls, feed_dict={
            y_tf : y,
            t_tf : t
        })
        
        return -y

    y0        = 1.0
    tspan     = np.linspace(0, 100, 101)

    start     = time.time()
    y         = odeint(rhs, y0, tspan)
    ODEtime   = time.time() - start

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace     = fetched_timeline.generate_chrome_trace_format()

    # with open('timeline_step_[nUser_{}].json'.format(nUser), 'w') as f:
    #     f.write(chrome_trace)

    sess.close()

    return (Graphtime, ODEtime)

def run_simulation_yesterday(nUser):

    print('running simulation for nUser', nUser)
    tf.reset_default_graph()

    # options      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    start        = time.time()

    with tf.device('/device:GPU:0'):

        bigMatrix = tf.Variable(np.random.random(size=(250, 250)), dtype=tf.float32, name='bigMatrix')

        for user in range(nUser):
            with tf.variable_scope('user{:05}'.format(user)):
                with tf.variable_scope('matrix_multiplication'):
                    for i in range(100):
                        if i == 0:
                            output = tf.sigmoid(bigMatrix * bigMatrix)
                        else:
                            output = tf.sigmoid(output * bigMatrix)
            
        init     = tf.global_variables_initializer()
    
    Graphtime = time.time() - start
    config    = tf.ConfigProto(gpu_options={'allow_growth':True})
    sess      = tf.Session(config=config)
    sess.run( init )
    tfwriter  = tf.summary.FileWriter('./tensorlog/simulation_yesterday/nUser{}/'.format(nUser), sess.graph)
    tfwriter.close()

    def rhs(y, t):
        
        # _ = sess.run(outputls, options=options, run_metadata=run_metadata)
        _ = sess.run(output)
        
        return -y

    y0        = 1.0
    tspan     = np.linspace(0, 100, 101)

    start     = time.time()
    y         = odeint(rhs, y0, tspan)
    ODEtime   = time.time() - start

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace     = fetched_timeline.generate_chrome_trace_format()

    # with open('timeline_step_[nUser_{}].json'.format(nUser), 'w') as f:
    #     f.write(chrome_trace)

    sess.close()

    return (Graphtime, ODEtime)

def main():

    nUser_list = [1, 10, 100]
    timespent  = [run_simulation_yesterday(n) for n in nUser_list]

    time_dict  = dict(zip(nUser_list, timespent))

    print('-' * 30)
    print('nUser: (graphTime, odeTime)')
    print(time_dict)


if __name__ == '__main__':

    main()