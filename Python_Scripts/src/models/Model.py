from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error

from src.utils.preprocess import timer

import matplotlib.pyplot as plt

import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
model_logger = logging.getLogger('Training Model')


class Model(ABC):
    """
    Model for classification task
    """
    model_name = ""
    target_second_part = 'time_diff_to_close'

    def __init__(self, X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part_data, target, update_col, guessed_win_probabilities_for_test_data, updates,
                 data_won, data_name, regression_model):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.second_part_data = second_part_data.copy()
        self.index_test = index_test
        self.index_train = index_train
        self.target = target
        self.index_won = second_part_data[target][second_part_data[second_part_data[target] == 1].first_valid_index()]
        self.model = None
        self.update_col = update_col
        self.y_predict = None
        self.win_probability = None
        self.y_probability_guessed = guessed_win_probabilities_for_test_data.copy()
        # self.y_guessed = [1 if x >= 0.5 else 0 for x in self.y_probability_guessed]
        self.updates = updates.copy()
        self.data_won = data_won
        self.plot_name = ""
        self.description = ""
        self.data_name = data_name
        self.regression_model = regression_model
        self.predict_col_name = 'predict_class'
        self.period_column_name = 'predicted_days_to_close'

    @abstractmethod
    def define_model(self):
        pass

    def predict_classification(self):
        """
        Fits the model and makes predictions for class and its probabilities
        :return: predictions for classification task
        """
        self.y_train = np.ravel(self.y_train)
        self.model.fit(self.X_train, self.y_train)
        self.y_predict = self.model.predict(self.X_test)
        self.win_probability = self.model.predict_proba(self.X_test)
        # take only probabilities where prediction is "Won"
        self.win_probability = self.win_probability[:, self.index_won]
        y_predict_classification = pd.DataFrame({'predict': self.y_predict, 'probability win': self.win_probability})
        y_predict_classification = y_predict_classification.set_index(self.index_test, drop=True)

        return y_predict_classification

    def prepare_data_for_second_part(self, classification_results):
        """
        Set values for classification task column for regression task. Train data gets real values, in the test data
        the values are set to the predictions made
        :param classification_results: predictions from the first part
        :return:
        """
        self.second_part_data.loc[self.index_train, self.predict_col_name] = self.second_part_data[self.target]
        self.second_part_data.loc[self.index_test, self.predict_col_name] = classification_results['predict']
        self.second_part_data = self.second_part_data.drop(columns=self.target)

    def predict_closing_period(self):
        """
        makes predictions for closing dates with regression model after classification step
        :return:
        """
        data = self.second_part_data

        X = data.drop(columns=self.target_second_part)
        to_change = X.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
        to_stay = X.select_dtypes(include='number')  # numerical data

        d = defaultdict(LabelEncoder)  # to be able to decode later
        x = to_change.apply(lambda f: d[f.name].fit_transform(f))
        X = x.join(to_stay, how='outer')
        # scaling values
        scaler = MinMaxScaler(copy=False)
        scaled_values = scaler.fit_transform(X)
        X.loc[:, :] = scaled_values
        X_test = X.loc[self.index_test]

        predictions = self.regression_model.predict(X_test)
        predictions = np.around(predictions).astype(int)
        predictions = np.where(predictions < 0, 0, predictions)  # period can't be negative

        predictions_periods = pd.DataFrame(predictions)
        predictions_periods.columns = [self.period_column_name]
        predictions_periods = predictions_periods.set_index(self.index_test, drop=True)
        data[self.period_column_name] = 0
        data.loc[self.index_test, self.period_column_name] = predictions_periods[self.period_column_name]
        return data

    def predict(self):
        """
        makes predictions first for the class and then for closing dates
        :return: data frame with complete predictions
        """
        classification_results = self.predict_classification()
        self.prepare_data_for_second_part(classification_results)
        predictions_final = self.predict_closing_period()
        predictions_final.index = predictions_final.index.map(int)
        predictions_final.loc[self.index_test, 'probability win'] = classification_results['probability win']
        return predictions_final

    def fit_predict(self):
        """
        makes complete predictions and prints/plots the results
        :return: model accuracy metrics and description
        """
        model_logger.info('Making predictions for ' + self.description)
        start = time.time()
        print(self.description.upper())

        predictions_periods = self.predict()
        print('Predictions'.upper())
        print(classification_report(self.y_test, self.y_predict))
        print('Confusion matrix'.upper())
        print(confusion_matrix(self.y_test, self.y_predict))

        # roc
        # calculate score
        model_auc = roc_auc_score(self.y_test, self.win_probability)
        print('Model: ROC AUC=%.3f' % model_auc)
        # calculate curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, self.y_probability_guessed)
        model_fpr, model_tpr, _ = roc_curve(self.y_test, self.win_probability)

        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(model_fpr, model_tpr, marker='.', label='Trained Model')
        plt.legend()
        title = 'ROC Curve for ' + self.data_name + ' with ' + self.plot_name
        plt.title(title)
        plot_name = '../../Plots/roc_' + self.description.lower().replace(" ", "_") + self.data_name.lower().replace(
            " ", "_") + '.svg'
        plt.savefig(plot_name)
        plt.show()

        mpe_guessed, mpe_predicted, mpe_predicted_strict = self.calculate_revenue_forecast(
            predictions_periods.loc[self.index_test])
        end = time.time()
        model_logger.info(('Making predictions finished. Execution time: ' + str(timer(start, end))))
        print('Making predictions finished. Execution time: ' + str(timer(start, end)))

        return model_auc, mpe_guessed, mpe_predicted, mpe_predicted_strict, self.description

    @staticmethod
    def __group_values_by_update(table, sum_values_column, group_col, values_col, period='M', upd_column='Update'):
        """
        Groups table by defined time period
        :param table: input table
        :param sum_values_column: Column with values which will be grouped
        :param period: grouping period
        :param upd_column: Datetime column
        :return: grouped table with new index
        """
        df = table.groupby(group_col)[values_col].sum().reset_index()
        df.columns = [upd_column, sum_values_column]
        res = df.groupby(df[upd_column].dt.to_period(period))[sum_values_column].sum().reset_index()
        res.index = res[upd_column]
        return res.drop(columns=upd_column)

    def __plot_errors(self, guessed, weighted, unweighted, frequency):
        """
        Plots error graphs
        :param guessed: guessed values
        :param weighted: weighted predictions
        :param unweighted: unweighted predictions
        :param frequency: time frequency
        :return:
        """
        plt.plot(guessed, color='red', label='Guessed Revenue')
        plt.plot(weighted, color='blue', label='Predicted Revenue')
        plt.plot(unweighted, color='green', label='Strictly Predicted Revenue')
        plt.legend()
        plt.ylabel('Mean Root Square Error')
        title = 'Compare ' + frequency + ' errors for ' + self.data_name + ' with\n' + self.plot_name
        plt.title(title)
        plot_name = '../../Plots/' + frequency[0] + '_err_' + self.description.lower().replace(" ", "_") + \
                    self.data_name.lower().replace(" ", "_") + '.svg'
        plt.savefig(plot_name)
        plt.show()

    def calculate_revenue_forecast(self, predictions):
        """

        :param predictions:
        :return:
        """
        predictions = predictions[['Opportunity_Name', 'Stage', 'Expected_closing', 'Volume', self.target_second_part,
                                   self.predict_col_name, self.period_column_name, 'probability win']]
        predictions['Update'] = self.updates
        predictions['Guessed probabilities'] = self.y_probability_guessed
        predictions['Guessed_closing'] = pd.to_datetime(predictions['Update']) + pd.to_timedelta(
            predictions['Expected_closing'], unit='D')
        predictions['Predicted_closing'] = pd.to_datetime(predictions['Update']) + pd.to_timedelta(
            predictions[self.period_column_name], unit='D')
        predictions['Guessed_revenue'] = predictions['Volume'] * predictions['Guessed probabilities']
        predictions['Predicted_revenue'] = predictions['Volume'] * predictions['probability win']
        predictions['Predicted_revenue_strict'] = np.where(predictions['probability win'] < 0.5, 0,
                                                           predictions['Volume'])

        # Closing dates MAE:
        mae_dates_guessed = mean_absolute_error(predictions[self.target_second_part], predictions['Expected_closing'])
        mae_dates_predicted = mean_absolute_error(predictions[self.target_second_part],
                                                  predictions[self.period_column_name])

        print('MAE for guessed closing dates: {:.2f}'.format(mae_dates_guessed))
        print('MAE for predicted closing dates: {:.2f}'.format(mae_dates_predicted))

        updates = predictions['Update'].unique()
        monthly_errors_guessed = []
        monthly_errors_weighted = []
        monthly_errors_unweighted = []
        quarterly_errors_guessed = []
        quarterly_errors_weighted = []
        quarterly_errors_unweighted = []
        for u in updates:
            df = predictions[predictions['Update'] == u]

            test_opps = df['Opportunity_Name'].unique()
            actual_won_opps = self.data_won[self.data_won['Opportunity_Name'].isin(test_opps)]
            actual_won_opps = actual_won_opps[['Opportunity_Name', 'Upload_date', 'Volume']]

            actual_revenue = self.__group_values_by_update(actual_won_opps, 'Actual_sum', 'Upload_date', 'Volume')
            guessed_revenue = self.__group_values_by_update(df, 'Guessed_sum', 'Guessed_closing', 'Guessed_revenue')
            weighted_revenue = self.__group_values_by_update(df, 'Weighted_sum', 'Predicted_closing', 'Predicted_revenue')
            unweighted_revenue = self.__group_values_by_update(df, 'Unweighted_sum', 'Predicted_closing', 'Predicted_revenue_strict')

            res_montly = pd.concat([guessed_revenue, weighted_revenue, unweighted_revenue], axis=1)
            res_montly = pd.concat([res_montly, actual_revenue], axis=1)
            res_montly = res_montly.fillna(0)

            monthly_errors_guessed.append(mean_absolute_error(res_montly['Actual_sum'], res_montly['Guessed_sum']))
            monthly_errors_weighted.append(
                mean_absolute_error(res_montly['Actual_sum'], res_montly['Weighted_sum']))
            monthly_errors_unweighted.append(mean_absolute_error(res_montly['Actual_sum'], res_montly['Unweighted_sum']))

            res_quarterly = res_montly[['Guessed_sum', 'Weighted_sum', 'Unweighted_sum', 'Actual_sum']].resample(
                'Q-JAN',
                convention='end').agg(
                'sum')

            quarterly_errors_guessed.append(mean_absolute_error(res_quarterly['Actual_sum'],
                                                                res_quarterly['Guessed_sum']))
            quarterly_errors_weighted.append(mean_absolute_error(res_quarterly['Actual_sum'],
                                                                  res_quarterly['Weighted_sum']))
            quarterly_errors_unweighted.append(mean_absolute_error(res_quarterly['Actual_sum'],
                                                                         res_quarterly['Unweighted_sum']))

        mean_monthly_mae_guessed = sum(monthly_errors_guessed) / len(monthly_errors_guessed)
        mean_monthly_mae_predicted = sum(monthly_errors_weighted) / len(monthly_errors_weighted)
        mean_monthly_mae_predicted_strict = sum(monthly_errors_unweighted) / len(monthly_errors_unweighted)

        print('Mean monthly MAE for guessed data: {:.2f}'.format(mean_monthly_mae_guessed))
        print('Mean monthly MAE for predicted data: {:.2f}'.format(mean_monthly_mae_predicted))
        print('Mean monthly MAE for strictly predicted data: {:.2f}'.format(mean_monthly_mae_predicted_strict))

        mean_quarterly_mae_guessed = sum(quarterly_errors_guessed) / len(quarterly_errors_guessed)
        mean_quarterly_mae_predicted = sum(quarterly_errors_weighted) / len(quarterly_errors_weighted)
        mean_quarterly_mae_predicted_strict = sum(quarterly_errors_unweighted) / len(
            quarterly_errors_unweighted)

        print('Mean quarterly MAE for guessed data: {:.2f}'.format(mean_quarterly_mae_guessed))
        print('Mean quarterly MAE for predicted data: {:.2f}'.format(mean_quarterly_mae_predicted))
        print('Mean quarterly MAE for strictly predicted data: {:.2f}'.format(mean_quarterly_mae_predicted_strict))

        self.__plot_errors(monthly_errors_guessed, monthly_errors_weighted, monthly_errors_unweighted, 'monthly')
        self.__plot_errors(quarterly_errors_guessed, quarterly_errors_weighted, quarterly_errors_unweighted,
                           'quarterly')

        return mean_quarterly_mae_guessed, mean_quarterly_mae_predicted, mean_quarterly_mae_predicted_strict
