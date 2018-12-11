from colorama import Fore, Back, Style
from multiprocessing import Process
import numpy as np
import os

from cy_wrangle import process_data_file
from exceptions import SolverException
from logistic_regression import LogisticRegression
from random_forest import DecisionTree, RandomForest
from measures import get_accuracy, get_f1_score

def init_model(model_type, delta, area_width):
	if model_type == 'LR':
		return LogisticRegression(delta, area_width)
	elif model_type == 'DT':
		return DecisionTree(delta)
	elif model_type == 'RF':
		return RandomForest(delta)
	else:
		raise SolverException('Invalid model type: ' + Fore.MAGENTA + model_type + Fore.RESET)

def generate_train_cv_sets(X, Y, percent_train=0.7):
	split_index = int(X.shape[0] * percent_train)

	X_train = X[ :split_index, : ]
	X_cv = X[ split_index:, : ]

	Y_train = Y[ :split_index, : ]
	Y_cv = Y[ split_index:, : ]
	
	return X_train, Y_train, X_cv, Y_cv

def predict_process(X, predictions, delta, model):
	row_indices = X[:, 0] == delta
	predictions[ row_indices ] = model.predict( X[ row_indices, 1: ] )

def train_process(X, Y, delta, model, is_cv):
	row_indices = X[:, 0] == delta
	X_delta_i = X[ row_indices, 1: ]
	Y_delta_i = Y[ row_indices, ]
	if is_cv:
		X_train, Y_train, X_cv, Y_cv = generate_train_cv_sets(X_delta_i, Y_delta_i)
	else:
		X_train = X_delta_i
		Y_train = Y_delta_i
		X_cv = None
		Y_cv = None
	model.fit(X_train, Y_train, X_cv, Y_cv)

def write_predictions_to_file(predictions, filename='submission.csv'):
	print('Writing predictions in ' + Fore.BLUE + filename + Fore.RESET)
	with open(filename, 'wb') as file:
		file.write(b'id,start.1,start.2,start.3,start.4,start.5,start.6,start.7,start.8,start.9,start.10,start.11,start.12,start.13,start.14,start.15,start.16,start.17,start.18,start.19,start.20,start.21,start.22,start.23,start.24,start.25,start.26,start.27,start.28,start.29,start.30,start.31,start.32,start.33,start.34,start.35,start.36,start.37,start.38,start.39,start.40,start.41,start.42,start.43,start.44,start.45,start.46,start.47,start.48,start.49,start.50,start.51,start.52,start.53,start.54,start.55,start.56,start.57,start.58,start.59,start.60,start.61,start.62,start.63,start.64,start.65,start.66,start.67,start.68,start.69,start.70,start.71,start.72,start.73,start.74,start.75,start.76,start.77,start.78,start.79,start.80,start.81,start.82,start.83,start.84,start.85,start.86,start.87,start.88,start.89,start.90,start.91,start.92,start.93,start.94,start.95,start.96,start.97,start.98,start.99,start.100,start.101,start.102,start.103,start.104,start.105,start.106,start.107,start.108,start.109,start.110,start.111,start.112,start.113,start.114,start.115,start.116,start.117,start.118,start.119,start.120,start.121,start.122,start.123,start.124,start.125,start.126,start.127,start.128,start.129,start.130,start.131,start.132,start.133,start.134,start.135,start.136,start.137,start.138,start.139,start.140,start.141,start.142,start.143,start.144,start.145,start.146,start.147,start.148,start.149,start.150,start.151,start.152,start.153,start.154,start.155,start.156,start.157,start.158,start.159,start.160,start.161,start.162,start.163,start.164,start.165,start.166,start.167,start.168,start.169,start.170,start.171,start.172,start.173,start.174,start.175,start.176,start.177,start.178,start.179,start.180,start.181,start.182,start.183,start.184,start.185,start.186,start.187,start.188,start.189,start.190,start.191,start.192,start.193,start.194,start.195,start.196,start.197,start.198,start.199,start.200,start.201,start.202,start.203,start.204,start.205,start.206,start.207,start.208,start.209,start.210,start.211,start.212,start.213,start.214,start.215,start.216,start.217,start.218,start.219,start.220,start.221,start.222,start.223,start.224,start.225,start.226,start.227,start.228,start.229,start.230,start.231,start.232,start.233,start.234,start.235,start.236,start.237,start.238,start.239,start.240,start.241,start.242,start.243,start.244,start.245,start.246,start.247,start.248,start.249,start.250,start.251,start.252,start.253,start.254,start.255,start.256,start.257,start.258,start.259,start.260,start.261,start.262,start.263,start.264,start.265,start.266,start.267,start.268,start.269,start.270,start.271,start.272,start.273,start.274,start.275,start.276,start.277,start.278,start.279,start.280,start.281,start.282,start.283,start.284,start.285,start.286,start.287,start.288,start.289,start.290,start.291,start.292,start.293,start.294,start.295,start.296,start.297,start.298,start.299,start.300,start.301,start.302,start.303,start.304,start.305,start.306,start.307,start.308,start.309,start.310,start.311,start.312,start.313,start.314,start.315,start.316,start.317,start.318,start.319,start.320,start.321,start.322,start.323,start.324,start.325,start.326,start.327,start.328,start.329,start.330,start.331,start.332,start.333,start.334,start.335,start.336,start.337,start.338,start.339,start.340,start.341,start.342,start.343,start.344,start.345,start.346,start.347,start.348,start.349,start.350,start.351,start.352,start.353,start.354,start.355,start.356,start.357,start.358,start.359,start.360,start.361,start.362,start.363,start.364,start.365,start.366,start.367,start.368,start.369,start.370,start.371,start.372,start.373,start.374,start.375,start.376,start.377,start.378,start.379,start.380,start.381,start.382,start.383,start.384,start.385,start.386,start.387,start.388,start.389,start.390,start.391,start.392,start.393,start.394,start.395,start.396,start.397,start.398,start.399,start.400\n')
		id_column = np.arange(predictions.shape[0]) + 1				# insert 'id' at column 0
		predictions = np.c_[id_column, predictions]
		np.savetxt(file, predictions.astype(int), fmt='%i', delimiter=',')

class Solver:
	np.random.seed(42)
	__DELTA_RANGE = 5

	__IS_CV = os.getenv('RGOL_CV') == 'TRUE'
	__IS_MP = os.getenv('RGOL_MP') == 'TRUE'
	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'

	def __init__(self, model_type):
		self.__half_stride = 3
		area_width = self.__half_stride * 2 + 1
		self.__models = []
		for i in range(self.__DELTA_RANGE):
			self.__models.append( init_model(model_type, i + 1, area_width) )

	def load_param(self):
		for model in self.__models:
			model.load_param()

	def train(self, filename):
		X, Y = process_data_file(filename, self.__half_stride, is_training_data=True)
		if self.__IS_MP:
			self.__train_mp(X, Y)
		else:
			self.__train_sp(X, Y)
		if self.__IS_VERBOSE:
			self.__measure_performance(X, Y)

	def __train_sp(self, X, Y):
		for i in range(self.__DELTA_RANGE):
			delta = i + 1
			print(Style.BRIGHT + 'Training model for delta = %d' % delta + Style.RESET_ALL)
			row_indices = X[:, 0] == delta
			X_delta_i = X[ row_indices, 1: ]
			Y_delta_i = Y[ row_indices, ]
			if self.__IS_CV:
				X_train, Y_train, X_cv, Y_cv = generate_train_cv_sets(X_delta_i, Y_delta_i)
			else:
				X_train = X_delta_i
				Y_train = Y_delta_i
				X_cv = None
				Y_cv = None
			self.__models[i].fit(X_train, Y_train, X_cv, Y_cv)

	def __train_mp(self, X, Y):
		processes = []
		for i in range(self.__DELTA_RANGE):
			delta = i + 1
			print(Style.BRIGHT + 'Training model for delta = %d' % delta + Style.RESET_ALL)
			processes.append(Process(target=train_process, args=(X, Y, delta, self.__models[i], self.__IS_CV)))
		for p in processes:
			p.start()
		for p in processes:
			p.join()

	def __measure_performance(self, X, Y):
		predictions = self.__get_predictions(X)
		accuracy = get_accuracy(predictions, Y)
		f1_score = get_f1_score(predictions, Y)
		print(Style.BRIGHT + 'Performance on entire training dataset: ' +Style.RESET_ALL +
			'Accuracy = %.6f, F1 Score = %.6f' % (accuracy, f1_score))

	def predict(self, filename):
		X_test, _ = process_data_file(filename, self.__half_stride, is_training_data=False)
		predictions = self.__get_predictions(X_test)
		predictions = predictions.reshape(-1, 400)
		write_predictions_to_file(predictions)

	def __get_predictions(self, X):
		predictions = np.empty((X.shape[0], 1))
		if self.__IS_MP:
			self.__get_predictions_mp(X, predictions)
		else:
			self.__get_predictions_sp(X, predictions)
		return predictions

	def __get_predictions_sp(self, X, predictions):
		for i in range(self.__DELTA_RANGE):
			delta = i + 1
			print(Style.BRIGHT + 'Predicting where delta = %d' % delta + Style.RESET_ALL)
			row_indices = X[:, 0] == delta
			predictions[ row_indices ] = self.__models[i].predict( X[ row_indices, 1: ] )

	def __get_predictions_mp(self, X, predictions):
		processes = []
		for i in range(self.__DELTA_RANGE):
			delta = i + 1
			print(Style.BRIGHT + 'Predicting where delta = %d' % delta + Style.RESET_ALL)
			processes.append(Process(target=predict_process, args=(X, predictions, delta, self.__models[i])))
		for p in processes:
			p.start()
		for p in processes:
			p.join()
