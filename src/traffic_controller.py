# @file traffic_controller.py
# @brief Deep RL-based Traffic Light Controllers
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
from collections import deque

# Import Paho MQTT Client
import paho.mqtt.client as mqtt

# Multi-threading lock
from threading import Thread, Lock

# Import the Deep RL Network Class
from DQNmodels import DQNAgent

# JSON Handling
import json

# Disable SSL hostname matching
import ssl
ssl.match_hostname = lambda cert, hostname: True

mutex = Lock()

########################### MQTT Client Callbacks ######################################
def on_subscribe(mosq, obj, mid, granted_qos):
	print("Subscribed: " + str(mid) + " " + str(granted_qos))

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
	global sub_list
	print("Connected to MQTT Broker with result code "+ str(rc))

	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe(sub_list)

# The callback for when a PUBLISH message is received from the server (receiving state from simulation interface)
def on_message(client, userdata, msg):
	global sub_topic
	global agent
	# print("Received Command: %s on topic %s" % (str(msg.payload), str(msg.topic)))
	light_topic = str(msg.topic)[len(sub_topic):]
	state_index = light_topic.find("/state")
	reward_index = light_topic.find("/reward")

	# Extract the name of the light
	if state_index != -1:
		light_name = light_topic[:state_index]
		# print("Light name is %s" % light_name)
	else:
		return

	# Extract the state from the JSON
	extracted_state = json.loads(msg.payload)
	processed_state = [np.asarray(extracted_state["queue_lengths"]), np.asarray(extracted_state["lights"])]

	# Get the agent, and predict the action
	# print(processed_state)
	# print(agent[light_name].action_size, agent[light_name].input_size)
	action = agent[light_name].act(processed_state)

	# Publish the action (encoded as Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6)
	client.publish(pub_topic + str(light_name), str(action))
	# print("Sent action %d to light %s" % (action, light_name))


########################### Sumo Scenario Class ######################################
class SumoLite:
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
		optParser.add_option("-c", "--config",
					  action="store", # optional because action defaults to "store"
					  default="../Scenarios/Sunnyvale-40/osm.sumocfg",
					  help="SUMO Scenario model to load",)
		optParser.add_option("-m", "--modelts",
					  action="store", # optional because action defaults to "store"
					  default="1533149383.0", #"1532569176.0",
					  help="Timestamp of the Deep RL Keras Models to Load",)
		optParser.add_option('--broker', '-b', default='test.mosquitto.org', type="string", help='IP or URL of the MQTT broker')
		optParser.add_option('--port', '-p', default='1883', type="int", help='MQTT port')
		optParser.add_option('--cacerts', default=None, type="string", help='CA certificate path')
		optParser.add_option('--certfile', default=None, type="string", help='Certificate file path')
		optParser.add_option('--keyfile', default=None, type="string", help='Keyfile path')
		optParser.add_option('--tlid', default=None, type="string", help='Traffic light ID to control, default is all lights')
		options, args = optParser.parse_args()
		return options

################################ MAIN ###############################################
if __name__ == '__main__':
	sumoInt = SumoLite()
	# this script has been called from the command line. It will start sumo as a
	# server, then connect and run
	options = sumoInt.get_options()

	# Scenario config file
	sumoCfg = options.config 

	# Choose the GUI/Non-GUI binary
	sumoBinary = "/usr/bin/sumo"
	
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

	# Start the TraCI simulation
	traci.start([sumoBinary, "-c", sumoCfg, "--start", "--no-warnings", "true"])

	# Create dictionaries of per-trafficlight data in first iteration
	print("Populating the TrafficLight/Detector Data")
	# Get the list of traffic lights and detectors
	TLIds = traci.trafficlight.getIDList()
	detectorIDs = traci.lanearea.getIDList()

	# Check if only a single light must be controlled
	if options.tlid != None:
		if options.tlid in TLIds:
			TLIds = [options.tlid]
		else:
			# If not found close the simulation and exit
			print("Cound not find the specified light, exiting ...")
			traci.close(wait=False)
			exit()

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

	# Dictionary of DQN Agents
	agent = {}
	total_detectors = 0
	for light in TLIds:
		num_detectors = len(detectorDictionary[light])
		num_phases = str(traci.trafficlight.getCompleteRedYellowGreenDefinition(light)).count("Phase:")
		print("Light " + light + " has " + str(num_detectors) + " detectors and " + str(num_phases) + " phases")
		
		total_detectors += len(detectorDictionary[light])

		# Create a Model for each traffic light
		agent[light] = DQNAgent(int(num_phases/2),num_detectors, 0.0)
		try:
			model_timestamp = options.modelts
			agent[light].load('../ScenarioModels/reinf_traf_control_light_' + light + '_' + model_timestamp + '.h5')
		except:
			print('No models found')
		
	# Close the local simulator as we are done with aprsing the scenario
	traci.close(wait=False)

	# MQTT Publishing topic prefix
	sub_topic = "qot/traffic_sensor_"

	# MQTT Subscriber topc prefix
	pub_topic = "qot/traffic_light_"

	# Create a list of subscription topics (traffic light actions)
	sub_list = []
	for light in TLIds:
		sub_tuple = (sub_topic + light + "/state", 0)
		sub_list.append(sub_tuple)
		sub_tuple = (sub_topic + light + "/reward", 0)
		sub_list.append(sub_tuple)

	# Connect to MQTT Broker
	client.connect(options.broker, options.port)

	# Start the MQTT Client Loop to handle the callbacks
	client.loop_start()

	# Busy Loop
	while True:
		time.sleep(10)

	log.close()
sys.stdout.flush()
