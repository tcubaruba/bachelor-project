import pandas as pd
import logging
import os
import sys

from src.preprocess import preprocess
from models.NeuralNetModel import NeuralNetModel
from models.LogRegModel import LogRegModel

sys.stdout = open('./Outputs/output.txt', 'w')

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Predicting win probability for sales data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

key_columns = '(Opportunity_Name,Product)'
update_col = 'Upload_date'
created_col = 'Created'
opp_name_col = 'Opportunity_Name'
product_name_col = 'Product'

key_columns = key_columns[1:-1].split(',')
target = 'future stage'
target_second_part = 'time_diff_to_close'

nn_activation_list = ['identity', 'logistic', 'tanh', 'relu']
nn_solver_list = ['lbfgs', 'sgd', 'adam']
nn_nodes_list = ['(2, 2, 2)', '(100, 100)', '(20, 16, 10, 4)', '(100, 80, 60, 40)']
lr_solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
c_list = [x * .1 for x in range(1, 11)]


def make_predictions(data_raw, data_name):
    X, y, index_train, index_test, second_part, guessed_win_probabilities_for_test_data, updates, data_won = preprocess(
        data_raw, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

    X_train = X.loc[index_train]
    X_test = X.loc[index_test]
    y_train = y.loc[index_train]
    y_test = y.loc[index_test]

    best_auc = 0
    best_mrse_guessed = 1000000
    best_mrse_predicted = 1000000
    best_mrse_predicted_strict = 1000000

    best_model_auc = ""
    best_model_mrse_predicted = ""
    best_model_mrse_predicted_strict = ""

    for nn_activation in nn_activation_list:
        for nn_solver in nn_solver_list:
            for nn_nodes in nn_nodes_list:
                nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                                    index_train, second_part, target, update_col,
                                    guessed_win_probabilities_for_test_data,
                                    updates,
                                    data_won, data_name)
                nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
                auc, mrse_guessed, mrse_predicted, mrse_predicted_strict, model = nn.make_predicitons()
                if auc > best_auc:
                    best_auc = auc
                    best_model_auc = model
                if mrse_guessed < best_mrse_guessed:
                    best_mrse_guessed = mrse_guessed
                if mrse_predicted < best_mrse_predicted:
                    best_mrse_predicted = mrse_predicted
                    best_model_mrse_predicted = model
                if mrse_predicted_strict < best_mrse_predicted_strict:
                    best_mrse_predicted_strict = mrse_predicted_strict
                    best_model_mrse_predicted_strict = model
    print(
        'best nn model by AUC: '.upper() + best_model_auc.upper() + " with AUC {:.2f}".format(best_auc))
    print(
        'best guessed revenue MRSE:'.upper() + '{:.2f}'.format(best_mrse_guessed))
    print(
        'best nn model by predicted revenue MRSE: '.upper() + best_model_mrse_predicted.upper() + " with MRSE {:.2f}".format(
            best_mrse_predicted))
    print(
        'best nn model by strictly predicted revenue MRSE: '.upper() + best_model_mrse_predicted_strict.upper() + " with MRSE {:.2f}".format(
            best_mrse_predicted_strict))

    best_auc = 0
    best_mrse_guessed = 1000000
    best_mrse_predicted = 1000000
    best_mrse_predicted_strict = 1000000

    best_model_auc = ""
    best_model_mrse_predicted = ""
    best_model_mrse_predicted_strict = ""

    for lr_solver in lr_solver_list:
        for c in c_list:
            lr = LogRegModel(X_train, X_test, y_train, y_test, index_test,
                             index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data,
                             updates,
                             data_won, data_name)
            lr.define_model(solver=lr_solver, c=c)
            auc, mrse_guessed, mrse_predicted, mrse_predicted_strict, model = lr.make_predicitons()
            if auc > best_auc:
                best_auc = auc
                best_model_auc = model
            if mrse_guessed < best_mrse_guessed:
                best_mrse_guessed = mrse_guessed
            if mrse_predicted < best_mrse_predicted:
                best_mrse_predicted = mrse_predicted
                best_model_mrse_predicted = model
            if mrse_predicted_strict < best_mrse_predicted_strict:
                best_mrse_predicted_strict = mrse_predicted_strict
                best_model_mrse_predicted_strict = model
    print(
        'best lr model by AUC: '.upper() + best_model_auc.upper() + " with AUC {:.2f}".format(best_auc))
    print(
        'best guessed revenue MRSE:'.upper() + '{:.2f}'.format(best_mrse_guessed))
    print(
        'best lr model by predicted revenue MRSE: '.upper() + best_model_mrse_predicted.upper() + " with MRSE {:.2f}".format(
            best_mrse_predicted))
    print(
        'best lr model by strictly predicted revenue MRSE: '.upper() + best_model_mrse_predicted_strict.upper() + " with MRSE {:.2f}".format(
            best_mrse_predicted_strict))


print('*' * 30 + ' generated data '.upper() + '*' * 30)
data_generated = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)
make_predictions(data_generated, "Generated data")

print('\n' + '*' * 30 + ' real data '.upper() + '*' * 30)
data_real = pd.read_csv('./Data/real_data_cleaned.csv', index_col=0)
make_predictions(data_real, "Real data")

sys.stdout.close()
