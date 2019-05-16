# @file traffic_controller_qot.py
# @brief Deep RL-based Traffic Light Controllers (mimicing real-world action)
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
# from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import time

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Import TraCI and other required libraries
import traci
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
sys.path.append(os.path.abspath("/usr/local/lib"))
from DQNmodels import DQNAgent

# JSON Handling
import json

# Disable SSL hostname matching
import ssl
ssl.match_hostname = lambda cert, hostname: True

# Import the Pause library
import pause

# Import Ring Buffer 
from ring_buffer import RingBuffer

# Import the QoT Stack Python API
sys.path.append(os.path.relpath("lib"))
from qot_coreapi import TimelineBinding
from qot_coreapi import ReturnTypes

# Signal Handling Library
import signal

# Ring Buffer size
RING_BUFSIZE = 20

# Debug flag
DEBUG = False

# Global Variable used to terminate program on SIGINT
running = True

# SIGINT signal handler
def signal_handler(signal, frame):
	print('Program Exiting')
	global running
	running = False

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

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global sub_topic
	global agent
	global ringBuffer
	global startFlag
	global options
	global binding

	startFlag = True
	# print("Received Command: %s on topic %s" % (str(msg.payload), str(msg.topic)))
	light_topic = str(msg.topic)[len(sub_topic):]
	state_index = light_topic.find("/state")
	reward_index = light_topic.find("/reward")

	# Extract the name of the light
	if state_index != -1:
		light_name = light_topic[:state_index]
	else:
		return

	# Extract the state from the JSON
	extracted_state = json.loads(msg.payload.decode('utf-8'))
	processed_state = {}
	processed_state["data"] = [np.asarray(extracted_state["queue_lengths"]), np.asarray(extracted_state["lights"])]
	if options.qotflag:
		processed_state["timestamp"] = binding.timeline_gettime()["time_estimate"]
	else:
		processed_state["timestamp"] = time.time()

	# Append value to the ring buffer
	ringBuffer[light_name].append(processed_state)

########################### Sumo Scenario Class ######################################
class SumoLite:
	def __init__(self):
		# we need to import python modules from the $SUMO_HOME/tools directory
		try:
			sys.path.append(os.path.join(os.path.dirname(
				__file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
			sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
				os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
			
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
		optParser.add_option("--modelpre",
					  action="store", # optional because action defaults to "store"
					  default="'../ScenarioModels/reinf_traf_control_light_'",
					  help="Model Prefix of the Deep RL Keras Models to Load",)
		optParser.add_option("-m", "--modelts",
					  action="store", # optional because action defaults to "store"
					  default="1533501063.0", #"1532569176.0",
					  help="Timestamp of the Deep RL Keras Models to Load",)
		optParser.add_option('--broker', '-b', default='test.mosquitto.org', type="string", help='IP or URL of the MQTT broker')
		optParser.add_option('--port', '-p', default='1883', type="int", help='MQTT port')
		optParser.add_option('--cacerts', default=None, type="string", help='CA certificate path')
		optParser.add_option('--certfile', default=None, type="string", help='Certificate file path')
		optParser.add_option('--keyfile', default=None, type="string", help='Keyfile path')
		optParser.add_option('--tlid', default=None, type="string", help='Traffic light ID to control, default is all lights')
		optParser.add_option('--simscale', default=1, type="int", help='The period at which to control the traffic lights')
		optParser.add_option('--decstep', default=10, type="int", help='The step at which the RL-agent makes decision')
		optParser.add_option('--qotflag', action="store_true", default=False, help='Flag to use the QoT Stack functionality')
		optParser.add_option('--debug', action="store_true", default=False, help='Flag to display debug prints')

		options, args = optParser.parse_args()
		return options

########################## Get State/Phase ###########################################
# This fucntion converts chosen actions into phases based on the mapping
# Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6
# This function also injects appropriate yellow phases
def get_phase(prev_action, action, t_light):
	global step_counters
	global options

	# Increment Step Counter
	step_counters[t_light] = (step_counters[t_light] + 1) % options.decstep

	# Decide on actions -> for the first four timesteps inject a yellow phase if we are transitioning phases
	if step_counters[t_light] < 4:
		if action[t_light] == 0: # Phase 0
			if prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 0: # Phase is 2
				# Transition to Phase 3 first (for safety)
				return 3

			elif prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 1: # Phase is 0
				# Stay in Phase 0
				return 0

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 0: # Phase is 4
				# Transition to Phase 5 first
				return 5

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 1: # Phase is 6
				# Transition to Phase 7 first
				return 7

		elif action[t_light] == 1: # Phase 2
			if prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 0: # Phase is 2
				# Stay in Phase 2
				return 2

			elif prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 1: # Phase is 0
				# Transition to Phase 1 first (for safety)
				return 1

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 0: # Phase is 4
				# Transition to Phase 5 first
				return 5

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 1: # Phase is 6
				# Transition to Phase 7 first
				return 7

		elif action[t_light] == 2: # Phase 4
			if prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 0: # Phase is 2
				# Transition to Phase 3 first (for safety)
				return 3

			elif prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 1: # Phase is 0
				# Transition to Phase 1 first (for safety)
				return 1

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 0: # Phase is 4
				# Stay in Phase 4
				return 4

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 1: # Phase is 6
				# Transition to Phase 7 first
				return 7


		elif action[t_light] == 3: # Phase 6
			if prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 0: # Phase is 2
				# Transition to Phase 3 first (for safety)
				return 3
			elif prev_action[t_light][0][0][0] == 0 and prev_action[t_light][0][1][0] == 1: # Phase is 0
				# Transition to Phase 1 first
				return 1

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 0: # Phase is 4
				# Transition to Phase 5 first
				return 5

			elif prev_action[t_light][0][0][0] == 1 and prev_action[t_light][0][1][0] == 1: # Phase is 6
				# Stay in Phase 6
				return 6

	# Implement and transition to a new phase for the remaining timesteps
	if step_counters[t_light] >= 4:
		if action[t_light] == 0: # Phase 0
			# Transition to Phase 0
			return 0

		elif action[t_light] == 1: # Phase 2
			# Transition to Phase 2
			return 2

		elif action[t_light] == 2: # Phase 4
			# Transition to Phase 4
			return 4

		elif action[t_light] == 3: # Phase 6
			# Transition to Phase 6
			return 6

# Comnsolidate the laste few timesteps of state into one state vector for the neural network input
def consolidate_state(state):
	processed_state = state[0]['data']
	for x in state:
		processed_state[0] += x['data'][0]
		processed_state[1]  = x['data'][1]

	return processed_state, x['timestamp']

# Get the state which the publisher (simulator is dumping to a ring buffer through MQTT)
def get_state(light):
	global ringBuffer
	global options

	data = ringBuffer[light].get()
	if (len(data) < options.decstep):
		return data[len(data)-1]['data'], data[len(data)-1]['timestamp']
	else:
		return consolidate_state(data[-options.decstep:])

################## Wait until next period implementation ##############################
def waituntil_nextperiod(period_ns):
	# Get current time
	now = time.time()

	# Compute the next wakeup time (assuming offset is 0)
	num_periods = int(math.floor(now*1000000000))//int(period_ns) + 1
	wakeup_time = (num_periods*period_ns)/1000000000

	# Sleep until the given core time
	pause.until(wakeup_time) 
	return time.time()

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

	# Get debug flag
	DEBUG = options.debug

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
	app_name = "traffic_light"
	if options.tlid != None:
		app_name += str(options.tlid)
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

	# Create a per-light ring buffer for buffering all the incoming data
	ringBuffer = {}
	for light in TLIds:
		ringBuffer[light] = RingBuffer(RING_BUFSIZE)

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
			agent[light].load(str(options.modelpre) + light + '_' + model_timestamp + '.h5')
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

	# Create local step counters for each light & action variables
	step_counters = {}
	action = {}
	prev_action = {}
	for light in TLIds:
		step_counters[light] = 0
		action[light] = 0
		prev_action[light] = 0

	# Connect to MQTT Broker
	client.connect(options.broker, options.port)

	# Start the MQTT Client Loop to handle the callbacks
	client.loop_start()

	# Register signal handler
	signal.signal(signal.SIGINT, signal_handler)

	if options.qotflag:
		# Bind to the timeline
		binding = TimelineBinding("app")
		retval = binding.timeline_bind("traffic_controllers", app_name, 1000, 1000)
		if retval != ReturnTypes.QOT_RETURN_TYPE_OK:
			print ('Unable to bind to timeline, terminating ....')
			exit (1)

		# Set the Period and Offset (1 second and 0 ns repectively)
		binding.timeline_set_schedparams(options.simscale*1000000000, 0)

	# Busy loop till we get the first data packet
	startFlag = False
	while startFlag == False and running == True:
		time.sleep(0.05)

	# Run the controller loop
	while running:
		# Sleep until the next period boundary
		if options.qotflag:
			now = binding.timeline_waituntil_nextperiod()["time_estimate"]
		else:
			now = waituntil_nextperiod(options.simscale*1000000000)
		for light in TLIds:
			# Get the agent, and predict the action
			if step_counters[light] == 0:
				# Get the current state
				processed_state, last_timestamp = get_state(light)
				prev_action[light] = processed_state[1]

				# Based on the state, generate an action for each light
				action[light] = agent[light].act(processed_state)
				if DEBUG:
					print("Chose action " + str(action[light]) + " at time " + str(now) + " for light " + str(light))

				# Correct the step incase we woke up late
				step_counters[light] = int(math.floor(now)) % options.decstep

			# Convert the action chosen into a traffic light phase
			phase = get_phase(prev_action, action, light)
			
			# Publish the action
			client.publish(pub_topic + str(light), str(phase))

	# Unbind from timeline
	binding.timeline_unbind()

	# Close all open files
	log.close()
	sys.stdout.flush()
