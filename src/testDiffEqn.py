from logs import logDecorator as lD
from lib.testLib import simpleLib as sL
import json
from importlib import util

config   = json.load(open('../config/config.json'))
logBase  = config['logging']['logBase']
logLevel = config['logging']['level']
logSpecs = config['logging']['specs']

@lD.log(logBase + '.importModules')
def importModules(logger):
    '''import and execute required modules
    
    This function is used for importing all the 
    modules as defined in the ../config/modules.json
    file and executing the main function within it
    if present. In error, it fails gracefully ...
    
    Parameters
    ----------
    logger : {logging.Logger}
        logger module for logging information
    '''
    modules = json.load(open('../config/modules.json'))

    for m in modules:

        try:
            if not m['execute']:
                logger.info('Module {} is being skipped'.format(m['moduleName']))
                continue
        except Exception as e:
            logger.error('Unable to check whether ')

        try:
            name, path = m['moduleName'], m['path']
            logger.info('Module {} is being executed'.format( name ))

            module_spec = util.spec_from_file_location(
                name, path)
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            module.main()
        except Exception as e:
            print('Unable to load module: {}->{}\n{}'.format(name, path, str(e)))

    return

@lD.logInit(logBase, logLevel, logSpecs)
def main(logger):
    '''main program
    
    This is the place where the entire program is going
    to be generated.
    '''

    # First import all the modules, and run 
    # them
    # ------------------------------------
    importModules()

    return

if __name__ == '__main__':
    main()