import os, sys

from lib.odeModels import simpleODE_tf
from logs import logDecorator as lD 
import json

import numpy as np
from scipy.integrate import odeint

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.odeSimple.odeSimple_tf'


if __name__ == '__main__':

	X  			     = np.random.random(size=(1000, 3))
	weights 		 = np.random.random(size=(100,))

	fullyconnectedNN = simpleODE_tf.Fullyconnected_noBias_tf(X=X, layers=[3, 10, 10, 3], activations=['linear', 'linear', 'linear'])