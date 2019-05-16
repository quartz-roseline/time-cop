# @file traffic_light_train.py
# @brief Basic RL-based Traffic Light Controller Training for SUMO Scenarios
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
# Note: Based on code by Tej Patel from https://github.com/TJ1812/Adaptive-Traffic-Signal-Control-Using-Reinforcement-Learning.git


from __future__ import absolute_import
from __future__ import print_function

# Basic Python Imports
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

# Import the Deep RL Network Class
from DQNmodels import DQNAgent

# Multi-threaded process
from threading import Thread

# Sumo Scenario Class
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
		optParser.add_option("--train", action="store_true",
							 default=False, help="train the model")
		optParser.add_option("-m", "--modelts",
					  action="store", # optional because action defaults to "store"
					  default="Models/reinf_traf_control2.h5",
					  help="Timestamp of the Deep RL Keras Models to Load",)
		optParser.add_option("-c", "--config",
					  action="store", # optional because action defaults to "store"
					  default="../Scenarios/Sunnyvale-40/osm.sumocfg",
					  help="SUMO Scenario model to load",)
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

		# Get the state of the light -> Encode 4 green phases as 2 binary bits
		light = []
		if (traci.trafficlight.getPhase(lightID) == 0):
			light = [0, 1]
		elif (traci.trafficlight.getPhase(lightID) == 2):
			light = [0, 0]
		elif (traci.trafficlight.getPhase(lightID) == 4):
			light = [1, 0]
		elif (traci.trafficlight.getPhase(lightID) == 6):
			light = [1, 1]

		# Reshape the current phase of the traffic light
		lgts = np.array(light)
		lgts = lgts.reshape(1, 2, 1)

		# Generate the state vector appending the queue lengths and the current light phase at the intersection
		return [queue_lengths, lgts]

# Get the waiting time of the step -> The negative of this function is the reward function
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
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 0)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 7) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

			elif action[t_light] == 1: # Phase 2
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 2)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 7) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

			elif action[t_light] == 2: # Phase 4
				if light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 0: # Phase is 2
					# Transition to Phase 3 first (for safety)
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 3)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 4)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

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
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 0 and light[t_light][0][1][0] == 1: # Phase is 0
					# Stay in Phase 0
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 1)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 0: # Phase is 4
					# Transition to Phase 5 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 5)
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])
					# traci.simulationStep()

				elif light[t_light][0][0][0] == 1 and light[t_light][0][1][0] == 1: # Phase is 6
					# Transition to Phase 7 first
					reward2[t_light] += get_waiting_vehicles(detectorDictionary[t_light])
					traci.trafficlight.setPhase(t_light, 6) 
					waiting_time[t_light] += get_step_waittime(detectorDictionary[t_light])	

		# Take a simulation step			
		traci.simulationStep()

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

		traci.simulationStep()

######################### Function which learns the traffic Controller ############################
# Note: Experimental Function which will be used for possible multi-threaded implmentation
#       Currently does not work due to keras multi-threading errors
def learnerFunc(light, next_state):
	global agent
	global reward
	global reward1
	global reward2
	global detectorDictionary
	global state
	global action
	global lightState
	global sumoInt
	global batch_size

	# Compute the reward 
	reward[light] = reward1[light] - reward2[light]
	agent[light].remember(state[light], action[light], reward[light], next_state, False)
	# Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
	if(len(agent[light].memory) > batch_size):
		agent[light].replay(batch_size)

	# Action for next step -> # Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6
	state[light] = next_state
	action[light] = agent[light].act(state[light])
	lightState[light] = state[light][1]
	
	# Reset rewards
	reward1[light] = 0
	reward2[light] = 0

######### Main Function ##############
if __name__ == '__main__':
	sumoInt = SumoScenario()
	# this script has been called from the command line. It will start sumo as a
	# server, then connect and run
	options = sumoInt.get_options()

	# Scenario config file
	sumoCfg = options.config #"/media/sf_sherlock/sumo/tools/Sunnyvale-40/osm.sumocfg"

	# Choose the GUI/Non-GUI binary
	if options.nogui:
		sumoBinary = "/usr/bin/sumo"
	else:
		sumoBinary = "/usr/bin/sumo-gui"

	now = time.time()

	# Set training/inference mode
	if options.train:
		print("Model Training Option Selected")
		train_flag = True
		# Main logic
		# parameters
		episodes = 1000		# Number of training episodes
		batch_size = 32		# Batch size for learning
	else:
		print("Model Inference Option Selected")
		# Main logic
		# parameters
		episodes = 1
		batch_size = 32
		train_flag = False

	# Open logging file
	log = open('log.txt', "w")

	for e in range(episodes):
		# Initialize the per-episode variables
		step = 0
		stepz = 0
		max_steps = 3000

		# Start the TraCI simulation
		traci.start([sumoBinary, "-c", sumoCfg, "--start", "--no-warnings", "true"])

		# Create dictionaries of per-trafficlight data in first iteration
		if e == 0:
			print("Populating the TrafficLight/Detector Data")
			# Get the list of traffic lights and detectors
			TLIds = traci.trafficlight.getIDList()
			detectorIDs = traci.lanearea.getIDList()

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

			total_detectors = 0
			# Dictionary of DQN Agents
			agent = {}
			for light in TLIds:
				num_detectors = len(detectorDictionary[light])
				num_phases = str(traci.trafficlight.getCompleteRedYellowGreenDefinition(light)).count("Phase:")
				print("Light " + light + " has " + str(num_detectors) + " detectors and " + str(num_phases) + " phases")
				
				total_detectors += len(detectorDictionary[light])
				
				# Create a Model for each traffic light
				agent[light] = DQNAgent(int(num_phases/2),num_detectors, 1.0)
				try:
					model_timestamp = options.modelts
					agent[light].load('../ScenarioModels/reinf_traf_control_light_' + light + '_' + model_timestamp + '.h5')
				except:
					print('No models found')

				# Set the initial Phase to "0" for 200 time steps
				traci.trafficlight.setPhase(light, 0)
				traci.trafficlight.setPhaseDuration(light, 200)


			print(str(total_detectors) + " detectors exist in the scenario")

			# Create thread dictionary for the learners -> commented out now
			# threads = {}
			# new_state = {}

		# Take the first step to flush out spurious values from the detectors
		traci.simulationStep()

		# Initialize Dictionaries
		waiting_time = {}
		reward1 = {}
		reward2 = {}
		total_reward = {}
		action = {}
		state = {}
		reward = {}
		lightState = {}
		for light in TLIds:
			waiting_time[light] = 0
			reward1[light] = 0
			reward2[light] = 0
			total_reward[light] = reward1[light] - reward2[light]
			action[light] = 0

			# Do Initial Action
			state[light] = sumoInt.getState(light, detectorDictionary[light])
			action[light] = agent[light].act(state[light])
			lightState[light] = state[light][1]

		# Run the simulation
		while traci.simulation.getMinExpectedNumber() > 0 and stepz < max_steps:
			# Perform actuation and measure rewards
			set_lights(lightState, action, TLIds, detectorDictionary)

			# Perform Remember/Train for each intersection's RL agent
			for light in TLIds:
				# Threading code dispatcher -> commented out for now
				# print("dispatching thread for light " + str(light) + " at step " + str(stepz))
				# new_state[light] = sumoInt.getState(light, detectorDictionary[light])
				# threads[light] = Thread(target=learnerFunc, args=(light,new_state[light],))
				# threads[light].start()

				# Get the current state of the intersection
				new_state = sumoInt.getState(light, detectorDictionary[light])

				# Compute the Reward
				reward[light] = reward1[light] - reward2[light]

				# Remeber the state, action and the reward
				agent[light].remember(state[light], action[light], reward[light], new_state, False)
				
				# Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
				if(len(agent[light].memory) > batch_size):
					agent[light].replay(batch_size)

				# Action for next step -> # Action 0 -> Phase 0, Action 1 -> Phase 2, Action 3 -> Phase 4, Action 4 -> Phase 6
				state[light] = new_state
				action[light] = agent[light].act(state[light])
				lightState[light] = state[light][1]
				
				# Reset rewards
				reward1[light] = 0
				reward2[light] = 0
 
			# Make the threads join -> commented out for now
			# for light in TLIds:
			# 	print("joining thread for light " + str(light) + " at step " + str(stepz))
			# 	threads[light].join()

		# Perform end of episode operations
		for light in TLIds:
			mem = agent[light].memory[-1]
			del agent[light].memory[-1]
			agent[light].memory.append((mem[0], mem[1], reward[light], mem[3], True))
		
		# End of the episode, perform logging and model saving
		total_waittime = 0
		for light in TLIds:
			total_waittime += waiting_time[light]
			log.write(' episode - ' + str(e) + ' traffic-light - ' + str(light) + ', total waiting time - ' + str(waiting_time[light]) + ', static waiting time - 338798 \n')
			if e % 10 == 0:
				agent[light].save('../ScenarioModels/reinf_traf_control_light_' + light + '_' + str(math.floor(now)) + '.h5')
		log.write('episode - ' + str(e) + ' total waiting time - ' + str(total_waittime) + ', static waiting time - 338798 \n')
		print('episode - ' + str(e) + ' total waiting time - ' + str(total_waittime) + ', static waiting time - 338798 \n')

		traci.close(wait=False)

# Close the files
log.close()
sys.stdout.flush()
