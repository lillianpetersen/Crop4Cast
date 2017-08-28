#==============================================================================
# install script
#==============================================================================


sudo apt-get install ipython
sudo apt-get install software-properties-common
sudo apt-get update
sudo apt-get install python-pip
sudo apt-get install python-tk
sudo pip install -U "celery[redis]"
sudo apt-get -y install redis-server

pip install --upgrade pip
sudo pip install matplotlib
sudo pip install descarteslabs
sudo pip install numpy
sudo pip install sklearn
sudo pip install scipy
easy_install celery
sudo pip install redis

export CLIENT_ID=ZOBAi4UROl5gKZIpxxlwOEfx8KpqXf2c
export CLIENT_SECRET=G4aHSt4UIUkq_TEIvIKlKr7CGQUMfRecOLQ72g2hpG1CE