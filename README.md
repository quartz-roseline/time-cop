# TimeCop: SUMO-based Distributed Traffic-Control Framework
This repository contains a framework to:
1) Import a SUMO Scenario
2) Train Deep RL-based Traffic Controllers for each imported intersection
3) Run a Deep RL-controlled traffic simulation

### Installation ###

The app requires the [SUMO](http://sumo.dlr.de/wiki/Simulation_of_Urban_MObility_-_Wiki) open-source traffic simulator to run. All the below instructions have been tested to work on Ubuntu 16.04. Before performing any of the steps, please clone the git repository

#### Install the SUMO simulator ####

```sh
$ sudo add-apt-repository ppa:sumo/stable
$ sudo apt-get update
$ sudo apt-get install sumo sumo-tools sumo-doc
```
Set the environment variable ```SUMO_HOME```
```sh
$ export SUMO_HOME=/usr/share/sumo
```
The above line can be added to ```.bashrc```.


#### Install the required machine-learning frameworks ####

```sh
$ pip install tensorflow
$ pip install theano
$ pip install keras
```

#### Install the required dependencies ####
```
$ pip install paho-mqtt 
$ pip install protobuf 
$ pip install pause
```

### Import a Scenario from SUMO ###
Use SUMO's OSM Web wizard to import a traffic scenario. Once you have the scenario run the following script on the `.net.xml`  file in the scenario folder to generate lane detectors at every intersection for the scenario. These lane detectorsare analogous to cameras which can tell the number of vehicles queued up in a lane at the intersection.
```
$ /usr/share/sumo/tools/output/generateTLSE2Detectors.py --net <path-to-.net.xml-file>
```

The scenarios pre-defined in the `Scenarios` folder can also be used.

### Train the model ###
To train the traffic models run the following python script (First navigate to the main directory of the repository). 
```sh
$ cd src
$ python traffic_light_train.py --train --nogui
```
The training script saves the trained traffic light controller models in the `ScenarioModels` folder with the following nomenclature reinf_traf_control_light_<light_name>_<training_timestamp>.h5. Removing the `--nogui` option will train the model while spinning upa GUI to visualize the simulation.

**Note**: The `traffic_light_train.py` script has been tested to work with Python 2.7. Although, it is expected to work with Python 3 as well.

### Run the simulation with the controllers ###
First start the Deep RL-based controllers
```sh
$ cd src
$ python traffic_controller_qot.py
```
The controller can be run with various command line options. Look at the file to know more, some of the important options are:
```
--nogui    : Run the simulation without the GUI
--tlid     : Specify a single traffic light ID for which to run the controller, by default the controller is run for all the intersections with traffic lights in the scenario
--qotflag  : Flag which uses the QoT API in the code, defaults to not using the QoT API
--debug    : Enable debug prints, disabled by default
--simscale : The size of the simulation step, defaults to 1 second
```
**Note**: The `traffic_controller_qot.py` script has been tested to work with Python 3. 

Then start the simulation scenario
```sh
$ cd src
$ python traffic_interface_real.py
```
The interface can be run with various command line options. Look at the file to know more, some of the important options are:
```
--nogui    : Run the simulation without the GUI
--simscale : The size of the simulation step, defaults to 1 second
--static   : Use the predefined phase timings, ignore controller commands
```
**Note**: The `traffic_controller_qot.py` script has been tested to work with Python 2.7. 

## Containerized Controller ##
A `Dockerfile` for a containerized version of the controller is present in the `docker` directory. To build the container run the following script:
```
$ cd docker
$ ./build.sh
```
You should now have a container image. To launch the controller, obtain a shell to the container using `docker exec` and run the `traffic_controller_qot.py` script present in the `/usr/local/bin` directory.

* * *

&copy; Anon 2018
&copy; Anon Inc. 2018

