from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from src.preprocess import scale_transform
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

class Model(ABC):
    model_name = ""
    target_second_part = 'time_diff_to_close'

    def __init__(self, X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part_data, target, update_col, guessed_win_probabilities_for_test_data):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.second_part_data = second_part_data
        self.index_test = index_test
        self.index_train = index_train
        self.target = target
        self.index_won = second_part_data[target][second_part_data[second_part_data[target] == 1].first_valid_index()]
        self.model = None
        self.update_col = update_col
        self.periods_model_name = 'dt'
        self.y_predict = None
        self.win_probability = None
        self.guessed_win_probabilities_for_test_data = guessed_win_probabilities_for_test_data

    @abstractmethod
    def define_model(self):
        pass

    def fit_score(self):
        self.y_train = np.ravel(self.y_train)
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(self.X_test, self.y_test)
        return acc

    def fit_predict_win_proba(self):
        self.y_train = np.ravel(self.y_train)
        self.model.fit(self.X_train, self.y_train)
        self.y_predict = self.model.predict(self.X_test)
        self.win_probability = self.model.predict_proba(self.X_test)
        self.win_probability = self.win_probability[:, self.index_won]
        y_predict = pd.DataFrame({'predict': self.y_predict, 'probability win': self.win_probability*100})

        predictions_column_name = 'predictions_' + self.model_name
        y_predict = y_predict.set_index(self.index_test, drop=True)
        self.second_part_data.loc[self.index_train, predictions_column_name] = self.second_part_data[self.target]
        self.second_part_data.loc[self.index_test, predictions_column_name] = y_predict['predict']

        self.second_part_data = self.second_part_data.drop(columns=self.target)
        # self.second_part_data = self.second_part_data.rename(columns={predictions_column_name: self.target})
        X, y = self.preprocess_periods()
        return X, y, y_predict

    def predict_closing_dates(self):
        X, y, y_predict = self.fit_predict_win_proba()
        predictions_periods = self.predict_period(X, y, self.index_test, self.index_train, self.second_part_data)
        predictions_periods.index = predictions_periods.index.map(int)
        predictions_periods.loc[self.index_test, 'probability win'] = y_predict['probability win'].map('{:,.2f}%'.format)
        predictions_periods.loc[self.index_test, 'predicted stage'] = y_predict['predict']

        return predictions_periods

    def preprocess_periods(self):
        open_data = self.second_part_data

        # convert data types
        to_change = open_data.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
        to_stay = open_data.select_dtypes(include='number')  # numerical data
        to_stay = to_stay.drop(columns='time_diff_to_close')

        y = open_data[self.target_second_part]  # writing the target column in y
        d = defaultdict(LabelEncoder)  # to be able to decode later
        X = scale_transform(to_change, to_stay, d)
        return X, y

    def predict_period(self, X, y, index_test, index_train, data):
        if index_test.empty:
            print('Empty data')
            data['time to close'] = 0
        else:
            X_train = X.loc[index_train]
            X_test = X.loc[index_test]
            y_train = y.loc[index_train]
            y_train = np.ravel(y_train)

            if self.periods_model_name == 'nn':
                model = MLPRegressor(hidden_layer_sizes=(100, 80), max_iter=1000)
                model.fit(X_train, y_train)
            else:  # model = decision trees
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            predictions = np.around(predictions).astype(int)
            predictions = np.where(predictions < 0, 0, predictions)

            predictions_periods = pd.DataFrame(predictions)
            period_columns_name = 'predicted days to close'
            predictions_periods.columns = [period_columns_name]
            predictions_periods = predictions_periods.set_index(index_test, drop=True)
            data[period_columns_name] = 0
            data.loc[index_test, period_columns_name] = predictions_periods[period_columns_name]

            data.drop(columns=period_columns_name)
        return data

    def make_predicitons(self):
        predictions_periods = self.predict_closing_dates()
        # acc = self.fit_score()
        print(classification_report(self.y_test, self.y_predict))
        print(confusion_matrix(self.y_test, self.y_predict))
        # roc
        ns_probs = self.guessed_win_probabilities_for_test_data
        # calculate scores
        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, self.win_probability)
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Model: ROC AUC=%.3f' % (lr_auc))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, self.win_probability)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Trained Model')
        plt.show()

        self.calculate_revenue_forecast(predictions_periods.loc[self.index_test])

        return predictions_periods

    def calculate_revenue_forecast(self, predictions):
        pass