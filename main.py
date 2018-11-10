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

import argparse
import os
import pandas as pd

from src.utils import preprocess_dataframe
from src.modelling import best_hp_select, test_model
from src.utils import get_best_sensitivity_and_specificity


miRNAs = ["miR-20a", "miR-20b", "miR-26a", "miR-106a", 'miR-191', 'miR-486']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-folder", type=str, help="input data folder")
	parser.add_argument("--positive-label", type=str, help="label as positive, eg. TBM or PTB")
	args = parser.parse_args()

	train_csv = os.path.join(args.data_folder, 'train_df.csv')
	test_csv = os.path.join(args.data_folder, 'test_df.csv')
	train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)
	train_df, test_df = preprocess_dataframe(train_df), preprocess_dataframe(test_df)

	both_features = train_df.keys()
	mirna_features = miRNAs + [key for key in both_features if 'group' in key]
	ehr_only_features = [key for key in both_features if key not in miRNAs]

	label = args.positive_label

	print("0. EHR + miRNAs modelling")
	best_clf, (hps, aucs), (best_valid_actual, best_valid_predictions) = best_hp_select(train_df[both_features], label)
	test_auc, test_actual, test_predictions = test_model(test_df[both_features], best_clf, label)
	sensitivity, specificity = get_best_sensitivity_and_specificity(best_valid_actual, best_valid_predictions)
	print("Validation: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))
	sensitivity, specificity = get_best_sensitivity_and_specificity(test_actual, test_predictions)
	print("Testing: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))

	print("1. EHR only modelling")
	best_clf, (hps, aucs), (best_valid_actual, best_valid_predictions) = best_hp_select(train_df[ehr_only_features], label)
	test_auc, test_actual, test_predictions = test_model(test_df[ehr_only_features], best_clf, label)
	sensitivity, specificity = get_best_sensitivity_and_specificity(best_valid_actual, best_valid_predictions)
	print("Validation: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))
	sensitivity, specificity = get_best_sensitivity_and_specificity(test_actual, test_predictions)
	print("Testing: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))

	print("2. miRNAs only modelling")
	best_clf, (hps, aucs), (best_valid_actual, best_valid_predictions) = best_hp_select(train_df[mirna_features], label)
	test_auc, test_actual, test_predictions = test_model(test_df[mirna_features], best_clf, label)
	sensitivity, specificity = get_best_sensitivity_and_specificity(best_valid_actual, best_valid_predictions)
	print("Validation: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))
	sensitivity, specificity = get_best_sensitivity_and_specificity(test_actual, test_predictions)
	print("Testing: Sensitivity:\t%.3f Specificity:\t%.3f" % (sensitivity, specificity))
