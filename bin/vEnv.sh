#!/bin/bash

#----------------------------------------------
# Note that this is the standard way of doing 
# things in Python 3.6. Earlier versions used
# virtulalenv. It is best to convert to 3.6
# before you do anything else. 
# Note that the default Python version in 
# the AWS Ubuntu is 3.5 at the moment. You
# will need to upgrade the the new version 
# if you wish to use this environment in 
# AWS
#----------------------------------------------
python3.6 -m venv env

# this is for bash. Activate
# it differently for different shells
#--------------------------------------
source env/bin/activate 

pip3 install --upgrade pip

if [ -e requirements.txt ]; then

    pip3 install -r requirements.txt

else

    pip3 install pytest
    pip3 install pytest-cov
    pip3 install sphinx
    pip3 install sphinx_rtd_theme

    # Logging into logstash
    pip3 install python-logstash
    
    # Utilities
    pip3 install ipython
    pip3 install tqdm

    # scientific libraries
    pip3 install numpy
    pip3 install scipy
    pip3 install pandas

    # ML libraries
    pip3 install sklearn
    pip3 install --upgrade ../../installers/tensorflowGPU/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

    # Charting libraries
    pip3 install matplotlib
    
    pip3 freeze > requirements.txt

fi

deactivate