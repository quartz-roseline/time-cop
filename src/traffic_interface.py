# @file traffic_interface.py
# @brief MQTT-based Traffic Light Controller Interface for SUMO Scenarios
#        (Expects coded actions which are converted to appropriates phases, and injects yellow phases
#         Action coding: # Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6)
# @author Anon D'Anon
#  
# Copyright (c) Anon, 2018. All rights reserved.
# Copyright (c) Anon Inc., 2018. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import
from __future__ import print_function

# Import system modules
import os
import sys
import optparse
import subprocess
import random
import time

# Check if SUMO_HOME is defined
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Import the TraCI SUMO COntrol Interface
import traci

# Import modules required for interfacing with Keras
import math
import random
import numpy as np
import h5py
import math
from collections import deque

# Import the Pause library
import pause

# Import Paho MQTT Client
import paho.mqtt.client as mqtt

# Multi-threading lock
from threading import Thread, Lock

# JSON Handling
import json

# Disable SSL hostname matching
import ssl
ssl.match_hostname = lambda cert, hostname: True

# Lock for locking the action variable
mutex = Lock()

# Simulation scale in seconds
simScale = 0.2

########################### Numpy to JSON encoder ######################################
class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

########################### MQTT Client Callbacks ######################################
def on_subscribe(mosq, obj, mid, granted_qos):
	print("Subscribed: " + str(mid) + " " + str(granted_qos))

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
	global sub_list
	print("Connected to MQTT Broker with result code "+str(rc))

	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe(sub_list)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global TLIds
	global sub_topic
	global recv_action
	# print("Received Command: %s on topic %s" % (str(msg.payload), str(msg.topic)))
	light_name = str(msg.topic)[len(sub_topic):]
	# print("Light name is %s" % light_name)

	# Write the received action into a variable
	mutex.acquire()
	try:
		recv_action[light_name] = int(str(msg.payload))
	finally:
		mutex.release()    


########################### Sumo Scenario Class ######################################
class SumoScenario:
	def __init__(self):
		# we need to import python modules from the $SUMO_HOME/tools directory
		try:
			sys.path.append(os.path.join(os.path.dirname(
				__file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
			sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
				os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
			# from sumolib import checkBinary  # noqa
		except ImportError:
			sys.exit(
				"please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

	# Parse the command line inputs
	def get_options(self):
		optParser = optparse.OptionParser()
		optParser.add_option("--nogui", action="store_true",
							 default=False, help="run the commandline version of sumo")
		optParser.add_option("-c", "--config",
					  action="store", # optional because action defaults to "store"
					  default="../Scenarios/Sunnyvale-40/osm.sumocfg",
					  help="SUMO Scenario model to load",)
		optParser.add_option('--broker', '-b', default='test.mosquitto.org', type="string", help='IP or URL of the MQTT broker')
		optParser.add_option('--port', '-p', default='1883', type="int", help='MQTT port')
		optParser.add_option('--cacerts', default=None, type="string", help='CA certificate path')
		optParser.add_option('--certfile', default=None, type="string", help='Certificate file path')
		optParser.add_option('--keyfile', default=None, type="string", help='Keyfile path')
		optParser.add_option('--simscale', default=0.2, type="float", help='Simulation Timescale')
		options, args = optParser.parse_args()
		return options

	# Get the state of a traffic light
	def getState(self, lightID, detectorIDs):
		# Get the number of vehicles from the detectors
		queue_lengths = []
		for detector in detectorIDs:
			veh_num = traci.lanearea.getLastStepVehicleNumber(detector)
			queue_lengths.append(veh_num)
		queue_lengths = np.array(queue_lengths)
		queue_lengths = queue_lengths.reshape(1, queue_lengths.shape[0], 1)

		# Get the state of the light
		light = []
		if (traci.trafficlight.getPhase(lightID) == 0):
			light = [0, 1]
		elif (traci.trafficlight.getPhase(lightID) == 2):
			light = [0, 0]
		elif (traci.trafficlight.getPhase(lightID) == 4):
			light = [1, 0]
		elif (traci.trafficlight.getPhase(lightID) == 6):
			light = [1, 1]
		else:
			light = [0, 0]

		lgts = np.array(light)
		lgts = lgts.reshape(1, 2, 1)

		return [queue_lengths.tolist(), lgts.tolist()]

################## Wait until next period implementation ##############################
def waituntil_nextperiod(period_ns):
	# Get current time
	now = time.time()

	# Compute the next wakeup time (assuming offset is 0)
	num_periods = int(math.floor(now*1000000000))//int(period_ns) + 1
	wakeup_time = (num_periods*period_ns)/1000000000

	# Sleep until the given core time
	pause.until(wakeup_time) 
	return 


################## Simulation Interface (State/Actions) ##############################
# Get the waiting time of the step
def get_step_waittime(detectorIDs):
	step_waiting_time = 0
	for detector in detectorIDs:
		step_waiting_time += traci.lanearea.getJamLengthVehicle(detector)
	return step_waiting_time

# Get the number of vehicles which were waiting during the last step
def get_waiting_vehicles(detectorIDs):
	step_waiting_time = 0
	for detector in detectorIDs:
		step_waiting_time += traci.lanearea.getJamLengthVehicle(detector)
	return step_waiting_time	

# Take a simulation step
def take_simstep():
	waituntil_nextperiod(simScale*1000000000) # -> Migrate this line to QoT Stack code
	traci.simulationStep()

# Set lights as per described actions
def set_lights(light, action, TLIds, detectorDictionary):
	global reward2
	global waiting_time
	global stepz
	# Take light phase decisions for 4 timesteps 
	# -> This is required for safety as we need to inject yellow phases between green phases
	# -> Note: The Controller in this case only returns green phases
	for i in range(4):
		stepz += 1
		# Take decisions over all the lights based on the next phase (action) and the previous phase (encoded in light)
		for t_light in TLIds:
			if action[t_light] == 0: # Phase 0
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 3)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 0)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 7) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

			elif action[t_light] == 1: # Phase 2
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 2)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 7) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

			elif action[t_light] == 2: # Phase 4
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 3)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 4)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 7) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])


			elif action[t_light] == 3: # Phase 6
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 3)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 6) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])	

		# Take a simulation step			
		take_simstep()

	# Transition to the chosen action phase for the next 6 time steps
	for i in range(6):
		stepz += 1
		# Take decisions over all the lights based on the next phase (action) 
		for t_light in TLIds:
			if action[t_light] == 0: # Phase 0
				# Transition to Phase 0
				reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
				traci.trafficlight.setPhase(t_light, 0) 
				waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

			elif action[t_light] == 1: # Phase 2
				# Transition to Phase 2
				reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
				traci.trafficlight.setPhase(t_light, 2) 
				waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

			elif action[t_light] == 2: # Phase 4
				# Transition to Phase 4
				reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
				traci.trafficlight.setPhase(t_light, 4) 
				waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

			elif action[t_light] == 3: # Phase 6
				# Transition to Phase 6
				reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
				traci.trafficlight.setPhase(t_light, 6) 
				waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])

		take_simstep()

# Get the latest programmed action for the light
def get_action(light):
	# Write the received action into a variable
	global recv_action
	retval = recv_action[light] 
	mutex.acquire()
	try:
		retval = recv_action[light] 
	finally:
		mutex.release()    
	return retval

################################ MAIN ###############################################
if __name__ == '__main__':
	sumoInt = SumoScenario()
	# this script has been called from the command line. It will start sumo as a
	# server, then connect and run
	options = sumoInt.get_options()

	# Scenario config file
	sumoCfg = options.config 

	# Choose the GUI/Non-GUI binary
	if options.nogui:
		sumoBinary = "/usr/bin/sumo"
	else:
		sumoBinary = "/usr/bin/sumo-gui"

	# Get the simulation time scaling factor
	simScale = options.simscale

	now = time.time()

	# Open logging file
	log = open('log.txt', "w")

	# MQTT Stuff
	client = mqtt.Client()
	client.on_connect = on_connect
	client.on_message = on_message
	client.on_subscribe = on_subscribe

	# Configure TLS Certificate
	if options.cacerts != None and options.certfile != None and options.keyfile != None:
		client.tls_set(ca_certs=options.cacerts, certfile=options.certfile, keyfile=options.keyfile)

	# Initialize the "stepz" variables
	stepz = 0
	max_steps = 7200

	# Start the TraCI simulation
	traci.start([sumoBinary, "-c", sumoCfg, "--start", "--no-warnings", "true"])

	# Create dictionaries of per-trafficlight data in first iteration
	print("Populating the TrafficLight/Detector Data")
	# Get the list of traffic lights and detectors
	TLIds = traci.trafficlight.getIDList()
	detectorIDs = traci.lanearea.getIDList()

	# MQTT Publishing topic prefix
	pub_topic = "qot/traffic_sensor_"

	# MQTT Subscriber topc prefix
	sub_topic = "qot/traffic_light_"

	# Create a list of subscription topics (traffic light actions)
	sub_list = []
	for light in TLIds:
		sub_tuple = (sub_topic + light, 0)
		sub_list.append(sub_tuple)

	# Connect to MQTT Broker
	client.connect(options.broker, options.port)

	# Start the MQTT Client Loop to handle the callbacks
	client.loop_start()

	# Get the Lanes corresponding to traffic lights
	laneDictionary = {}
	for light in TLIds:
		lanes = traci.trafficlight.getControlledLinks(light)
		laneDictionary[light] = lanes

	# Create a traffic light to detector dictionary
	detectorDictionary = {}
	for light in TLIds:
		detectorDictionary[light] = []

	# Assign detectors to a traffic light
	for detector in detectorIDs:
		laneID = traci.lanearea.getLaneID(detector)
		# print("Detector = " + detector + ", LaneID = " + laneID)
		break_flag = 0;
		for light in TLIds:
			for sublist in laneDictionary[light]:
				for subl in sublist:
					if laneID in subl:
						detectorDictionary[light].append(detector)
						break_flag = 1
						break
				if break_flag == 1:
					break
			if break_flag == 1:
				break

	# Print the number of phases and detectors for each traffic light, and set the initial phase
	total_detectors = 0
	for light in TLIds:
		num_detectors = len(detectorDictionary[light])
		num_phases = str(traci.trafficlight.getCompleteRedYellowGreenDefinition(light)).count("Phase:")
		print("Light " + light + " has " + str(num_detectors) + " detectors and " + str(num_phases) + " phases")
		
		total_detectors += len(detectorDictionary[light])

		# Set the initial Phase to "0" 
		traci.trafficlight.setPhase(light, 0)

	print(str(total_detectors) + " detectors exist in the scenario")

	# Take the first step to flush out spurious values from the detectors
	take_simstep()

	# Initialize Dictionaries
	waiting_time = {}
	reward1 = {}
	reward2 = {}
	total_reward = {}
	action = {}
	state = {}
	reward = {}
	lightState = {}
	recv_action = {}
	for light in TLIds:
		waiting_time[light] = 0
		reward1[light] = 0
		reward2[light] = 0
		total_reward[light] = reward1[light] - reward2[light]
		action[light] = 0
		recv_action[light] = 0

		# Do Initial Action
		state[light] = sumoInt.getState(light, detectorDictionary[light])
		action[light] = get_action(light)
		lightState[light] = state[light][1]

	# Run the simulation
	while traci.simulation.getMinExpectedNumber() > 0 and stepz < max_steps:
		# Perform actuation and measure rewards
		set_lights(lightState, action, TLIds, detectorDictionary)

		# Grab the intersection state to publish 
		for light in TLIds:
			new_state = sumoInt.getState(light, detectorDictionary[light])
			state_dict = {}
			state_dict["queue_lengths"] = new_state[0]
			state_dict["lights"] = new_state[1]
			reward[light] = reward1[light] - reward2[light]
			client.publish(pub_topic + light + "/state", json.dumps(state_dict))#, cls=NumpyEncoder))
			client.publish(pub_topic + light + "/reward", str(reward[light]))
			lightState[light] = new_state[1]
			
			# Reset rewards
			reward1[light] = 0
			reward2[light] = 0

		# Sleep until the next second boundary -> Migrate next line to the QoT Stack "wait_until_next_period"
		stepz += 1
		take_simstep()

		# Get the actions from the "Edges"
		for light in TLIds:
			# Action for next step -> # Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6
			action[light] = get_action(light)

	
	# End of the episode, perform logging and model saving
	total_waiting_time = 0
	for light in TLIds:
		print('traffic-light - ' + str(light) + ' total waiting time - ' + str(waiting_time[light]) + ', static waiting time - 338798 \n')
		total_waiting_time += waiting_time[light]
	
	print('Scenario waiting time - ' + str(total_waiting_time) + ', static waiting time - 374750 \n')
	traci.close(wait=False)
	log.close()
	sys.stdout.flush()
