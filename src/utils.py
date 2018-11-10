# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integrating exosomal microRNA and electronic health data improves tuberculosis diagnosis.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def delete_nan_columns(df):
	"""Rename the NaN columns in dataframe

	Args:
		df: raw dataframe

	Returns: renamed dataframe
	"""
	for key in df.keys():
		if 'Unnamed' in key and key in df.keys():
			del df[key]
	return df


def rename_columns_dataframe(df):
	"""Rename the specific columns in dataframe

	Args:
		df: raw dataframe

	Returns: renamed dataframe
	"""

	names = {
		"Hct": "hematocrit",
		"L": "lymphocyte",
		"N": "neutrophil",
		"M": "monocyte",
		"Hb": "hemoglobin",
		"Alb": "albumin"
	}
	df = df.rename(columns=names)
	return df


def rename_values_dataframe(df):
	"""Rename the specific values in dataframe

	Args:
		df: raw dataframe

	Returns: renamed dataframe
	"""
	names = {
		"meningitis": "other meningitis"
	}

	df.replace(names)
	return df


def preprocess_dataframe(df):
	"""Integrate the steps for preprocessing

	Args:
		df: raw dataframe

	Returns: renamed dataframe
	"""
	df = rename_columns_dataframe(df)
	df = delete_nan_columns(df)
	df = rename_values_dataframe(df)
	return df


def get_sensitivity_and_specificity(predictions, actual, cutoff):
	"""Returns sensitivity and specificity given specific cutoff

	Args:
		predictions: list of predicted probabilities
		actual: list of ground truth, same order as predictions
		cutoff: the threshold to calculate sensitively and specificity

	Returns: sensitively and specificity
	"""

	predictions_label = np.zeros(len(predictions))
	predictions_label[predictions > cutoff] = 1
	cm = confusion_matrix(actual, predictions_label, labels=[1, 0])
	sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[1, 0])
	specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[0, 1])
	return sensitivity, specificity


def get_max_youden_index_threshold(actual, predictions):
	"""Returns cutoff with max youden index

	Args:
		predictions: list of predicted probabilities
		actual: list of ground truth, same order as predictions

	Returns: cutoff
	"""

	sorted_predictions = np.sort(predictions)
	max_youden, max_cutoff = -1, -1
	for i in range(len(sorted_predictions)):
		sensitivity, specificity = get_sensitivity_and_specificity(predictions, actual, sorted_predictions[i])
		tmp_youden = sensitivity + specificity
		if tmp_youden > max_youden:
			max_youden = tmp_youden
			max_cutoff = sorted_predictions[i]

	return max_cutoff


def get_best_sensitivity_and_specificity(actual, predictions):
	"""Returns sensitivity and specificity with best youden index

	Args:
		predictions: list of predicted probabilities
		actual: list of ground truth, same order as predictions

	Returns: sensitively and specificity
	"""

	best_youden_index_cutoff = get_max_youden_index_threshold(actual, predictions)
	sensitivity, specificity = get_sensitivity_and_specificity(predictions, actual, best_youden_index_cutoff)
	return sensitivity, specificity

