# Source Code #

The following files are present in this directory:

## Traffic Light Controllers ##
* **traffic_controller.py** - Implements a traffic-light controller for an intersection. This conmtroller does not have any notion of time and is driven by the simulation. It is triggered by an input state from the simulator, and generates an action which it sends back to the simulator.
* **traffic_controller_qot.py** - Implements a nicer traffic-light controller for an intersection, with QoT integration. It maintains a shared notion of time. Receives data and takes action at specific period boundaries.

## Traffic Simulation Interface ##
* **traffic_interface.py** - Loads a traffic scenario using SUMO and TraCI and sets up a simulation testbed, with MQTT endpoints for each intersection to communicate its state and receive commands. This interface expects the controllers to provide only green phases encoded as actions:  Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6.
* **traffic_interface_real.py** - Loads a traffic scenario using SUMO and TraCI and sets up a simulation testbed, with MQTT endpoints for each intersection to communicate its state and receive commands. The only difference from the previous interface is it moves more logic to the controller and expects the traffic controllers to provide the phase.

## Traffic Light Controller Training ##
* **traffic_light_train.py** - For a given scenario, this script finds intersections with traffic lights and sensors, creates per-intersection deep reinforcement-learning controllers, and finally performs a simulation to train the controllers.
* **traffic_light_train_coord.py** - Performs the same function as the previous script, with one difference. The training model even uses the state from the neighbouring intersections.

## Library Files ##
* **DQNmodels** - Contains a class defining a prototype deep RL model 
* **ring_buffer.py** - Contains a class defining a ring buffer (circular buffer)

## External Libraries ##
* **lib/qot_coreapi.py** - Library file with the QoT API
