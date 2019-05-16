# @file ring_buffer.py
# @brief Ring Buffer implementation 
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

# Multi-threading lock
from threading import Thread, Lock

class RingBuffer:
	""" class that implements a not-yet-full buffer """
	def __init__(self,size_max):
		self.max = size_max
		self.data = []
		self.mutex = Lock()

	class __Full:
		""" class that implements a full buffer """
		def append(self, x):
			""" Append an element overwriting the oldest one. """
			self.mutex.acquire()
			try:
				self.data[self.cur] = x
				self.cur = (self.cur+1) % self.max
			finally:
				self.mutex.release()

		def get(self):
			""" return list of elements in correct order """
			self.mutex.acquire()
			try:
				ret_data = self.data[self.cur:]+self.data[:self.cur]
			finally:
				self.mutex.release()
			return ret_data

	def append(self,x):
		"""append an element at the end of the buffer"""
		self.mutex.acquire()
		try:
			self.data.append(x)
			if len(self.data) == self.max:
				self.cur = 0
				# Permanently change self's class from non-full to full
				self.__class__ = self.__Full
		finally:
			self.mutex.release()

	def get(self):
		""" Return a list of elements from the oldest to the newest. """
		self.mutex.acquire()
		try:
			ret_data = self.data
		finally:
			self.mutex.release()

		return ret_data