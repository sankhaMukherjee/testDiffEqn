import numpy as np
from logs import logDecorator as lD 
import json

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.testing.check_Y'

@lD.log(logBase + '.main')
def main(logger):

    try:
        y_numpy = np.load('../results/allresults_dy_numpy.npy')
        y_tf    = np.load('../results/allresults_dy_tensorflow.npy')

        print('-'*50)
        print('y_numpy')
        print(y_numpy[:5])
        print('')
        print('-'*50)
        print('y_tf')
        print(y_tf[:5])
        print('')
        print('diff:')
        print(np.square(y_numpy - y_tf).mean())

    except Exception as e:

        logger.error('Unable to run main in check_Y \n{}'.format(str(e)))