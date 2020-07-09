import pandas as pd
import numpy as np

import sys

from utils.preprocess import preprocess

from models.NeuralNetModel import NeuralNetModel
from models.LogRegModel import LogRegModel
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def train_regression_model(regression_model, data, index_train, target):
    """
    Trains model for predicting closing dates

    :param regression_model: defined but not trained regression model
    :param data: data prepared for the second part
    :param index_train: training index, array
    :param target: name of column to predict
    :return:
    """
    X_periods = data.drop(columns=target)
    to_change = X_periods.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
    to_stay = X_periods.select_dtypes(include='number')  # numerical data
    d = defaultdict(LabelEncoder)  # to be able to decode later
    X_periods = to_change.apply(lambda f: d[f.name].fit_transform(f))
    X_periods = X_periods.join(to_stay, how='outer')

    # scaling values
    scaler = MinMaxScaler(copy=False)
    scaled_values = scaler.fit_transform(X_periods)
    X_periods.loc[:, :] = scaled_values
    X_periods_train = X_periods.loc[index_train]
    y_periods = data[target]
    y_periods_train = y_periods.loc[index_train]

    regression_model.fit(X_periods_train, y_periods_train)


def initialize_metrics():
    """
    Sets metrics to start values
    :return: initialized metrics
    """
    best_auc = 0
    best_mae_predicted_weighted = np.inf
    best_mae_predicted_unweighted = np.inf

    best_model_auc = ""
    best_model_mae_predicted_weighted = ""
    best_model_mae_predicted_unweighted = ""
    return best_auc, best_mae_predicted_weighted, best_mae_predicted_unweighted, best_model_auc, \
           best_model_mae_predicted_weighted, best_model_mae_predicted_unweighted


def make_predictions(data_raw, data_name, regression_model):
    """
    makes predictions for NN and LR models
    :param data_raw: input data not preprocessed
    :param data_name: name of data
    :param regression_model: initialized model for regression task
    :return:
    """
    # define column names in data. Usually would be a console input
    key_columns = '(Opportunity_Name,Product)'
    update_col = 'Upload_date'
    created_col = 'Created'
    opp_name_col = 'Opportunity_Name'
    product_name_col = 'Product'

    key_columns = key_columns[1:-1].split(',')
    target = 'future stage'  # target for the first part --> stage Won or Lost
    target_second_part = 'time_diff_to_close'  # target for the second part --> time till closing

    # Models parameters
    nn_activation_list = ['identity', 'logistic', 'tanh', 'relu']
    nn_solver_list = ['lbfgs', 'sgd', 'adam']
    nn_nodes_list = ['(2, 2, 2)', '(100, 100)', '(20, 16, 10, 4)', '(100, 80, 60, 40)']
    lr_solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    c_list = [x * .1 for x in range(1, 11)]

    # preprocess data
    X, y, index_train, index_test, second_part, y_proba_guessed, updates, data_won = preprocess(
        data_raw, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

    # train model for periods prediction first, because the same model will be used each time
    train_regression_model(regression_model, second_part, index_train, target_second_part)

    X_train = X.loc[index_train]
    X_test = X.loc[index_test]
    y_train = y.loc[index_train]
    y_test = y.loc[index_test]

    y_guessed = [1 if x >= 0.5 else 0 for x in y_proba_guessed]
    print('Guessed probabilities'.upper())
    print(classification_report(y_test, y_guessed))
    print('Confusion matrix'.upper())
    print(confusion_matrix(y_test, y_guessed))
    ns_auc = roc_auc_score(y_test, y_proba_guessed)
    print('No Skill: ROC AUC=%.3f' % ns_auc)

    best_auc, best_mae_weighted, best_mae_unweighted, best_model_auc, best_model_mae_weighted, best_model_mae_unweighted = initialize_metrics()

    for nn_activation in nn_activation_list:
        for nn_solver in nn_solver_list:
            for nn_nodes in nn_nodes_list:
                nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                                    index_train, second_part, target, update_col,
                                    y_proba_guessed,
                                    updates,
                                    data_won, data_name, regression_model)
                nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
                auc, mae_guessed, mae_weighted, mae_unweighted, model = nn.fit_predict()
                if auc > best_auc:
                    best_auc = auc
                    best_model_auc = model
                if mae_weighted < best_mae_weighted:
                    best_mae_weighted = mae_weighted
                    best_model_mae_weighted = model
                if mae_unweighted < best_mae_unweighted:
                    best_mae_unweighted = mae_unweighted
                    best_model_mae_unweighted = model
    print(
        'best nn model by AUC: '.upper() + best_model_auc.upper() + " with AUC {:.2f}".format(best_auc))
    print(
        'best guessed revenue MAE: '.upper() + '{:.2f}'.format(mae_guessed))
    print(
        'best nn model by predicted revenue MAE: '.upper() + best_model_mae_weighted.upper() +
        " with MAE {:.2f}".format(best_mae_weighted))
    print(
        'best nn model by strictly predicted revenue MAE: '.upper() + best_model_mae_unweighted.upper() +
        " with MAE {:.2f}".format(best_mae_unweighted))

    best_auc, best_mae_weighted, best_mae_unweighted, best_model_auc, best_model_mae_weighted, best_model_mae_unweighted = initialize_metrics()

    for lr_solver in lr_solver_list:
        for c in c_list:
            lr = LogRegModel(X_train, X_test, y_train, y_test, index_test,
                             index_train, second_part, target, update_col, y_proba_guessed,
                             updates, data_won, data_name, regression_model)
            lr.define_model(solver=lr_solver, c=c)
            auc, mae_guessed, mae_weighted, mae_unweighted, model = lr.fit_predict()
            if auc > best_auc:
                best_auc = auc
                best_model_auc = model
            if mae_weighted < best_mae_weighted:
                best_mae_weighted = mae_weighted
                best_model_mae_weighted = model
            if mae_unweighted < best_mae_unweighted:
                best_mae_unweighted = mae_unweighted
                best_model_mae_unweighted = model
    print(
        'best lr model by AUC: '.upper() + best_model_auc.upper() + " with AUC {:.2f}".format(best_auc))
    print(
        'best guessed revenue MAE: '.upper() + '{:.2f}'.format(mae_guessed))
    print(
        'best lr model by predicted revenue MAE: '.upper() + best_model_mae_weighted.upper() +
        " with MAE {:.2f}".format(best_mae_weighted))
    print(
        'best lr model by strictly predicted revenue MAE: '.upper() + best_model_mae_unweighted.upper() +
        " with MAE {:.2f}".format(best_mae_unweighted))


def main():
    sys.stdout = open('../../Outputs/console_output.txt', 'w')  # write output there

    # optimal config for real data
    regression_model = RandomForestRegressor(n_estimators=100, random_state=42, criterion='mae', n_jobs=-1,
                                             min_samples_leaf=0.01, min_samples_split=0.3, max_samples=0.8)
    print('\n' + '*' * 30 + ' real data '.upper() + '*' * 30)
    data_real = pd.read_csv('../../Data/real_data.csv', index_col=0)
    make_predictions(data_real, "Real data", regression_model)

    # optimal config for generated data
    regression_model = RandomForestRegressor(n_estimators=100, random_state=42, criterion='mae', n_jobs=-1,
                                             min_samples_leaf=0.02, min_samples_split=0.01)
    print('*' * 30 + ' generated data '.upper() + '*' * 30)
    data_generated = pd.read_csv('../../Data/generated_data.csv', index_col=0)
    make_predictions(data_generated, "Generated data", regression_model)

    sys.stdout.close()


if __name__ == "__main__":
    main()
