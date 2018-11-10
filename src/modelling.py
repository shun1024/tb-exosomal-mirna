import os

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


def k_fold_validate(train_df, clf, positive_label, k=3, labels=['group', 'group property']):
	"""Returns the best hyper-parameter according its performance in validation dataset

	Args:
		train_df: training dataframe
		clf: basic used model
		positive_label: positive label
		k: number of fold
		labels: unused features for modelling

	Returns: accuracy, AUC, actual label, predictions
	"""
	keys = [key for key in train_df.keys() if key not in labels]
	X, y = train_df[keys].as_matrix(), train_df['group'].as_matrix()
	y = np.array([1 if tmp == positive_label else 0 for tmp in y])

	scores, pred_y = [], []
	indices = range(len(X))
	kf = KFold(k)
	for train, test in kf.split(indices):
		train_X, train_y = X[train], y[train]
		valid_X, valid_y = X[test], y[test]
		clf.fit(train_X, train_y)
		scores.append(clf.score(valid_X, valid_y))
		probs = pd.DataFrame(clf.predict_proba(valid_X), columns=clf.classes_)
		pred_y.extend(probs[1].as_matrix().tolist())

	valid_acc = np.mean(np.array(scores))
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pred_y)
	valid_auc = auc(false_positive_rate, true_positive_rate)

	return valid_acc, valid_auc, y, np.array(pred_y)


def best_hp_select(train_df, positive_label):
	"""Returns the best hyper-parameter according its performance in validation dataset

	Args:
		train_df: training dataframe
		positive_label: positive label

	Returns: best model, tuple of hyper-parameters list and their corresponding AUC in validation, best model performance tuple
	"""

	hps, aucs = [], []
	best_auc = 0
	best_valid_actual, best_valid_predictions = None, None

	for i in range(30):
		hp = 1.5 ** (i - 15)
		clf = SVC(C=hp, probability=True, kernel='linear', class_weight='balanced')
		valid_acc, valid_auc, valid_actual, valid_predictions = k_fold_validate(train_df, clf, positive_label)
		aucs.append(valid_auc)
		hps.append(hp)

		if valid_auc > best_auc:
			best_auc = valid_auc
			best_valid_actual = valid_actual
			best_valid_predictions = valid_predictions

	best_hp = hps[np.argmax(aucs)]
	best_clf = SVC(C=best_hp, probability=True, kernel='linear')
	return best_clf, (hps, aucs), (best_valid_actual, best_valid_predictions)


def test_model(test_df, clf, positive_label, labels=['group', 'group property']):
	"""Returns the predictions on testing data with given model

	Args:
		test_df: testing dataframe
		clf: positive label
		positive_label: label as positive
		labels: unused features for modelling

	Returns: best model, tuple of hyper-parameters list and their corresponding AUC in validation, best performance tuple
	"""

	keys = [key for key in test_df.keys() if key not in labels]
	X, y = test_df[keys].as_matrix(), test_df['group'].as_matrix()
	y = np.array([1 if tmp == positive_label else 0 for tmp in y])
	pred_y = clf.predict_proba(X)[:, 1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pred_y)
	return auc(false_positive_rate, true_positive_rate), y, pred_y
