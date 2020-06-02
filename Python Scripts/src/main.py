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

    nn_best_auc = 0
    lr_best_auc = 0

    nn_best_mpe_guessed = 1000000
    lr_best_mpe_guessed = 1000000

    nn_best_mpe_predicted = 1000000
    lr_best_mpe_predicted = 1000000

    nn_best_model_auc = ""
    lr_best_model_auc = ""

    nn_best_model_mpe__guessed = ""
    lr_best_model_mpe_guessed = ""

    nn_best_model_mpe__predicted = ""
    lr_best_model_mpe_predicted = ""

    for nn_activation in nn_activation_list:
        for nn_solver in nn_solver_list:
            for nn_nodes in nn_nodes_list:
                nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                                    index_train, second_part, target, update_col,
                                    guessed_win_probabilities_for_test_data,
                                    updates,
                                    data_won, data_name)
                nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
                nn_auc, nn_mpe_guessed, nn_mpe_predicted, nn_model = nn.make_predicitons()
                if nn_auc > nn_best_auc:
                    nn_best_auc = nn_auc
                    nn_best_model_auc = nn_model
                if nn_mpe_guessed < nn_best_mpe_guessed:
                    nn_best_mpe_guessed = nn_mpe_guessed
                    nn_best_model_mpe__guessed = nn_model
                if nn_mpe_predicted < nn_best_mpe_predicted:
                    nn_best_mpe_predicted = nn_mpe_predicted
                    nn_best_model_mpe__predicted = nn_model
    print(
        'best nn model by AUC: '.upper() + nn_best_model_auc.upper() + " with AUC {:.2f}".format(nn_best_auc))
    print(
        'best nn model by guessed revenue MPE: '.upper() + nn_best_model_mpe__guessed.upper() + " with MPE {:.2f}".format(
            nn_best_mpe_guessed))
    print(
        'best nn model by predicted revenue MPE: '.upper() + nn_best_model_mpe__predicted.upper() + " with MPE {:.2f}".format(
            nn_best_mpe_predicted))

    for lr_solver in lr_solver_list:
        for c in c_list:
            lr = LogRegModel(X_train, X_test, y_train, y_test, index_test,
                             index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data,
                             updates,
                             data_won, data_name)
            lr.define_model(solver=lr_solver, c=c)
            lr_auc, lr_mpe_guessed, lr_mpe_predicted, lr_model = lr.make_predicitons()
            if lr_auc > lr_best_auc:
                lr_best_auc = lr_auc
                lr_best_model_auc = lr_model
            if lr_mpe_guessed < lr_best_mpe_guessed:
                lr_best_mpe_guessed = lr_mpe_guessed
                lr_best_model_mpe_guessed = lr_model
            if lr_mpe_predicted < lr_best_mpe_predicted:
                lr_best_mpe_predicted = lr_mpe_predicted
                lr_best_model_mpe_predicted = lr_model
    print(
        'best lr model by AUC: '.upper() + lr_best_model_auc.upper() + " with AUC {:.2f}".format(lr_best_auc))
    print(
        'best lr model by guessed revenue MPE: '.upper() + lr_best_model_mpe_guessed.upper() + " with MPE {:.2f}".format(
            lr_best_mpe_guessed))
    print(
        'best lr model by predicted revenue MPE: '.upper() + lr_best_model_mpe_predicted.upper() + " with MPE {:.2f}".format(
            lr_best_mpe_predicted))


print('*' * 30 + ' generated data '.upper() + '*' * 30)
data_generated = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)
make_predictions(data_generated, "Generated data")

print('\n' + '*' * 30 + ' real data '.upper() + '*' * 30)
data_real = pd.read_csv('./Data/real_data_cleaned.csv', index_col=0)
make_predictions(data_real, "Real data")

sys.stdout.close()
