from logs import logDecorator as lD 
import json

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


    def dy():