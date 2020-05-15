from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from src.preprocess import scale_transform
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class Model(ABC):
    model_name = ""
    target_second_part = 'time_diff_to_close'

    def __init__(self, X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part_data, target, period, update_col):
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
        self.period = period
        self.update_col = update_col

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
        y_predict = self.model.predict(self.X_test)
        win_probability = self.model.predict_proba(self.X_test)

        win_probability = win_probability[:, self.index_won]
        y_predict = pd.DataFrame({'predict': y_predict, 'probability win': win_probability})

        predictions_column_name = 'predictions_' + self.model_name
        y_predict = y_predict.set_index(self.index_test, drop=True)
        self.second_part_data.loc[self.index_train, predictions_column_name] = self.second_part_data[self.target]
        self.second_part_data.loc[self.index_test, predictions_column_name] = y_predict['predict']

        self.second_part_data = self.second_part_data.drop(columns=self.target)
        self.second_part_data = self.second_part_data.rename(columns={predictions_column_name: self.target})
        X, y, index_test_new, index_train_new = self.preprocess_periods()
        return X, y, index_test_new, index_train_new, y_predict

    def predict_closing_dates(self):
        X, y, index_test_new, index_train_new, y_predict = self.fit_predict_win_proba()
        predictions_periods = self.predict_period(X, y, index_test_new, index_train_new, self.second_part_data)
        predictions_periods.index = predictions_periods.index.map(int)
        predictions_periods.loc[self.index_test, 'probability win'] = y_predict['probability win']
        predictions_periods.loc[self.index_test, 'predicted stage'] = y_predict['predict']

        return predictions_periods

    def preprocess_periods(self):
        # # make predictions only for data which predicted to be be won
        # open_data = self.second_part_data.loc[self.second_part_data['future stage'] == 1]
        open_data = self.second_part_data

        index_train = open_data.index[open_data[self.update_col] < self.period]
        index_test = open_data.index[open_data[self.update_col] >= self.period]

        # delete unnecessary columns
        open_data = open_data.drop(columns=[self.update_col, 'future stage'])

        # convert data types
        to_change = open_data.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
        to_stay = open_data.select_dtypes(include='number')  # numerical data
        to_stay = to_stay.drop(columns='time_diff_to_close')

        y = open_data[self.target_second_part]  # writing the target column in y
        d = defaultdict(LabelEncoder)  # to be able to decode later
        X = scale_transform(to_change, to_stay, d)
        return X, y, index_test, index_train

    def predict_period(self, X, y, index_test, index_train, data, model_name='dt'):
        if index_test.empty:
            print('Empty data')
            data['time to close'] = 0
        else:
            X_train = X.loc[index_train]
            X_test = X.loc[index_test]
            y_train = y.loc[index_train]
            y_train = np.ravel(y_train)

            X_test_orig = X_test
            # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, shuffle=True)

            if model_name == 'nn':
                model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000)
                model.fit(X_train, y_train)
            else:  # model = decision trees
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)

            predictions = model.predict(X_test_orig)
            predictions = np.around(predictions).astype(int)
            predictions = np.where(predictions < 0, 0, predictions)

            predictions_periods = pd.DataFrame(predictions)
            period_columns_name = 'days to close'
            predictions_periods.columns = [period_columns_name]
            predictions_periods = predictions_periods.set_index(index_test, drop=True)
            data[period_columns_name] = 0
            data.loc[index_test, period_columns_name] = predictions_periods[period_columns_name]
            # data.loc[index_test, 'time to close'] = data[self.update_col] + pd.to_timedelta(data[period_columns_name],
            #                                                                          unit='D')
            data.drop(columns=period_columns_name)
        return data

    def make_predicitons(self):
        predictions_periods = self.predict_closing_dates()
        acc = self.fit_score()
        return predictions_periods, acc
