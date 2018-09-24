from logs import logDecorator as lD 
import json

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

