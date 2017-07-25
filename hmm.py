"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy
import math
import random

class HMM(Classifier):
        
	def __init__(self, seq_file):
		# initialize necessary data structures
		self.labels = []
		self.features = []

		# matrices with probabilities
		self.transition_matrix = numpy.zeros((1,1))
		self.emission_matrix = numpy.zeros((1,1))

		# matrices with counts
		self.transition_count_table = numpy.ones((1,1)) # hidden state to hidden state
		self.feature_count_table = numpy.ones((1,1)) # observation associate with a particular hidden state

		# label2index: a dictionary map from label string to its index (label codebook)
		self.label2index = {}

        # index2label: a list map from label index to its string 
		self.index2label = {}

		# file path to store predicted sequences
		self.seq_file = './sequences/' + seq_file + '.txt'
		with open(self.seq_file,'w') as f:
			f.write(seq_file+"\n---------------\n\n")
	
	def get_model(self): return None
	def set_model(self, model): pass

	model = property(get_model, set_model)
		
	def _get_all_counts(self, instance_list):
		"""Populates the lists for features and labels

		Returns None
		"""
		set_labels = set()
		set_features = set()
		
		feature_dict = {}
		for instance in instance_list:
			# count up features
			for feature in instance.features():
				if feature not in feature_dict:
					feature_dict[feature] = 1
				else:
					feature_dict[feature] += 1

			# get the labels
			for label in instance.label:
				set_labels.add(label)

		# smooth by removing any features that don't appear at least twice (to replace with <UNK> later)
		for feature, num in feature_dict.items():
			if num >= 2:
				set_features.add(feature)

		self.features = list(set_features)
		self.features.append("<UNK>")
		self.labels = list(set_labels)

		print "\nGot counts of features and labels"

		return None
	
	def _collect_counts(self, instance_list):
		"""Collect counts necessary for fitting parameters

		This function should update self.transition_count_table
		and self.feature_count_table based on this new given instance
		
		Iterate through each instance given
		Within each instance, iterate through the states and observations
		Populate the transition count table with the current and next state
		Populate feature count table with the observation and its state

		Returns None
		"""

		# update matrices
		# use ones instead of zeros for smoothing
		self.transition_count_table = numpy.ones((len(self.labels),len(self.labels)))
		self.feature_count_table = numpy.ones((len(self.features),len(self.labels)))

		# populate tables
		for instance in instance_list:
			labels = instance.label
			obs = instance.features()
			for i in range(len(labels) - 1):
				current_instance_label = labels[i]
				current_state_index = self.label2index[current_instance_label]

				next_instance_label = labels[i + 1]
				next_state_index = self.label2index[next_instance_label]

				current_ob = obs[i]
				if current_ob not in self.features:
					current_ob_index = self.features.index("<UNK>")
				else:
					current_ob_index = self.features.index(current_ob)

				self.transition_count_table[current_state_index][next_state_index] += 1
				self.feature_count_table[current_ob_index][current_state_index] += 1
		
			# make sure to add in the last observation
			last_ob = obs[-1]
			if last_ob not in self.features:
				last_ob_index = self.features.index("<UNK>")
			else:
				last_ob_index = self.features.index(last_ob)
			self.feature_count_table[last_ob_index][current_state_index] += 1

		print "Populated transition and feature count tables"
		
		return None
	
	def _update_codebooks(self):
		"""Update label2index and index2label

		Returns None
		"""
		for label in self.labels:
			self.label2index[label] = self.labels.index(label)

		self.index2label = {v: k for k, v in self.label2index.iteritems()}

		print "Updated codebooks"
		return None
	
	def train(self, instance_list):
		"""Fit parameters for hidden markov model

		Update codebooks from the given data to be consistent with
		the probability tables 

		Transition matrix and emission probability matrix
		will then be populated with the maximum likelihood estimate 
		of the appropriate parameters

		To get the probabilities, simply take relevant count table
		and divide by the sum
		For transition matrix, the sum is the total transitions for a state
		so we sum along the Y-axis of transition count table
		For emission matrix, the sum is the number of times in a state
		so sum along X-axis of the feature count table

		Returns None
		"""
		self._get_all_counts(instance_list)
		self._update_codebooks()
		self._collect_counts(instance_list)

		# update matrices
		self.transition_matrix = numpy.zeros((len(self.labels),len(self.labels)))
		self.emission_matrix = numpy.zeros((len(self.features),len(self.labels)))

		# estimate the parameters from the count tables
		# transition state prob = num transition from i to j / num transitions from state i
		self.transition_matrix = self.transition_count_table / self.transition_count_table.sum(axis=1)[:,None]

		# state observation prob of observation o given state j = num of times in state j with observation v_k / num of times in state j
		self.emission_matrix = self.feature_count_table / self.feature_count_table.sum(axis=0)[None,:]

		print "Done training"
		return None

	def featurize(self, instance):
		"""Populates feature vector for the instance

		Returns None
		"""
		for feat in instance.features():
			if feat in self.features:
				instance.feature_vector.append(feat)
			else:
				instance.feature_vector.append("<UNK>")

		return None
	
	def classify(self, instance):
		"""Viterbi decoding algorithm

		Wrapper for running the Viterbi algorithm
		We can then obtain the best sequence of labels from the backtrace pointers matrix

		First, in the trellis, find the state with the highest probability for the last observation
		Then, using the backtrace pointers, going backwards, append the state given

		Returns a list of labels e.g. ['B','I','O','O','B']
		"""
		self.featurize(instance)
		backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)

		best_sequence = []
		num_features = len(instance.feature_vector)

		# get the index to start with
		prob = 0.0
		j = 0
		for state in self.index2label:
			# get state for the last observation
			if backtrace_pointers[0][state][num_features] > prob:
				j = state
				prob = backtrace_pointers[0][state][num_features]

		# previous states
		for i in reversed(range(1,num_features+1)):
			best_sequence.append(self.index2label[backtrace_pointers[1][j][i]])
			j = int(backtrace_pointers[1][j][i])

		best_sequence = list(reversed(best_sequence))

		# write sequences to text file
		with open(self.seq_file, 'a') as f:
			f.write(str(instance.features())+'\n')
			f.write(str(instance.feature_vector)+'\n')
			f.write(str(instance.label)+'\n')
			f.write(str(best_sequence)+'\n')
			f.write("***********\n")

		return best_sequence

	def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
		"""Run Forward algorithm or Viterbi algorithm

		This function uses the trellis to implement dynamic
		programming algorithm for obtaining the best sequence
		of labels given the observations

		Initialization: get probability of transition from start to that state * prob of the first observation emitting that state
		Recursion: find the max prob of alpha * transition prob * likelihood of state observation
			for backtrace pointers, return a pointer to the index of the label associated with that prob
		Termination: alpha * prob transition to final state
			for backtrace pointers, return a pointer to the index of the label associated with that prob

		Returns trellis filled up with the forward probabilities 
		and backtrace pointers for finding the best sequence
		"""
		# Initialize trellis and backtrace pointers 
		observations = instance.feature_vector
		matrix_size = (len(self.labels),len(observations) + 1)

		trellis = numpy.zeros(matrix_size)
		backtrace_pointers = numpy.zeros(matrix_size)

		# initialize values
		for i in self.index2label:
			# transition prob = going from start to that state * first ob emitting that state
			first_ob_index = self.features.index(observations[0])
			trellis[i][0] = self.transition_matrix[0][i] * self.emission_matrix[first_ob_index][i]

			if not run_forward_alg:
				backtrace_pointers[i][0] = 0

		# recursion
		for t in range(1, len(observations)):
			for s in self.index2label:
				max_viterbi_prob = 0.0
				for j in self.index2label:
					alpha = trellis[j][t-1]
					transition_prob = self.transition_matrix[j][s]
					ob_index = self.features.index(observations[t])
					state_ob_likelihood = self.emission_matrix[ob_index][s]

					prob = alpha * transition_prob * state_ob_likelihood
					if prob > trellis[s][t]:	
						trellis[s][t] = prob

					if not run_forward_alg:
						prob_viterbi = alpha * transition_prob
						if prob_viterbi > max_viterbi_prob:
							max_viterbi_prob = prob_viterbi
							backtrace_pointers[s][t] = j # store the index/'pointer' instead of the prob

		# termination
		for i in self.index2label:
			for j in self.index2label:
				# terminal prob = last * transition to final state
				last_ob_index = self.features.index(observations[-1])
				prob = trellis[j][len(observations)-1] * self.transition_matrix[j][i]
				if prob > trellis[i][len(observations)]:
					trellis[i][len(observations)] = prob

				if not run_forward_alg:
					max_viterbi_prob = 0.0
					prob_viterbi = trellis[i][len(observations)]
					if prob_viterbi > max_viterbi_prob:
						max_viterbi_prob = prob_viterbi
						backtrace_pointers[i][len(observations)] = j
		
		return (trellis, backtrace_pointers)

