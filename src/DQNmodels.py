# @file DQNmodels.py
# @brief Deep RL-based controllers for different intersection types
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
# from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import time
import math
import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class DQNAgent:
	def __init__(self, actionSize, inputSize, exploration):
		self.gamma = 0.95   # discount rate
		self.epsilon = exploration  # exploration rate
		self.exploration_decay = 0.995
		self.min_epsilon = 0.01
		self.learning_rate = 0.001
		self.memory = deque(maxlen=200)
		self.action_size = actionSize
		self.input_size = inputSize
		self.model = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		input_1 = Input(shape=(self.input_size, 1))
		x1 = Dense(16, activation='relu')(input_1)
		x1 = Dense(32, activation='relu')(x1)
		x1 = Flatten()(x1)


		input_3 = Input(shape=(2, 1))
		x3 = Flatten()(input_3)

		x = keras.layers.concatenate([x1, x3])
		x = Dense(128, activation='relu')(x)
		x = Dense(64, activation='relu')(x)
		x = Dense(self.action_size, activation='linear')(x)

		model = Model(inputs=[input_1, input_3], outputs=[x])
		model.compile(optimizer=keras.optimizers.RMSprop(
			lr=self.learning_rate), loss='mse')
		model._make_predict_function()

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)

		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.min_epsilon:
			self.epsilon *= self.exploration_decay

	def load(self, name):
		self.model.load_weights(name)
		self.model._make_predict_function()

	def save(self, name):
		self.model.save_weights(name)