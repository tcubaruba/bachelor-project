import pandas as pd
import logging
import os
import sys

from src.preprocess import preprocess
from models.NeuralNetModel import NeuralNetModel
from models.LogRegModel import LogRegModel
from models.SVCModel import SVCModel

sys.stdout = open('./Outputs/output.txt', 'w')

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Predicting win probability for sales data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print('*' * 30 + ' generated data '.upper() + '*' * 30)
data_raw = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)

# Read parameters
# _, test_period, nn_nodes, lr_solver, nn_solver, nn_activation, svc_kernel, key_columns, update_col, created_col, opp_name_col, product_name_col = sys.argv
lr_solver = 'default'
svc_kernel = 'default'
key_columns = '(Opportunity_Name,Product)'
update_col = 'Upload_date'
created_col = 'Created'
opp_name_col = 'Opportunity_Name'
product_name_col = 'Product'

key_columns = key_columns[1:-1].split(',')
target = 'future stage'
target_second_part = 'time_diff_to_close'

X, y, index_train, index_test, second_part, guessed_win_probabilities_for_test_data, updates, data_won = preprocess(
    data_raw, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

X_train = X.loc[index_train]
X_test = X.loc[index_test]
y_train = y.loc[index_train]
y_test = y.loc[index_test]

nn_activation_list = ['identity', 'logistic', 'tanh', 'relu']
nn_solver_list = ['lbfgs', 'sgd', 'adam']
nn_nodes_list = ['(2, 2, 2)', '(100, 100)', '(20, 10, 4)', '(100, 80, 60)']

for nn_activation in nn_activation_list:
    for nn_solver in nn_solver_list:
        for nn_nodes in nn_nodes_list:
            nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                                index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data,
                                updates,
                                data_won, 'Generated data')
            nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
            nn.make_predicitons()

lr_solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
for lr_solver in lr_solver_list:
    lr = LogRegModel(X_train, X_test, y_train, y_test, index_test,
                     index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data, updates,
                     data_won, 'Generated data')
    lr.define_model(solver=lr_solver, penalty='')
    lr.make_predicitons()


# svc = SVCModel(X_train, X_test, y_train, y_test, index_test,
#                  index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data, updates,
#                  data_won, 'Generated data')
# svc.define_model(kernel=svc_kernel)
# data_after_svc = svc.make_predicitons()


print('\n' + '*' * 30 + ' real data '.upper() + '*' * 30)
data_raw = pd.read_csv('./Data/real_data_cleaned.csv', index_col=0)

X, y, index_train, index_test, second_part, guessed_win_probabilities_for_test_data, updates, data_won = preprocess(
    data_raw, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

X_train = X.loc[index_train]
X_test = X.loc[index_test]
y_train = y.loc[index_train]
y_test = y.loc[index_test]

for nn_activation in nn_activation_list:
    for nn_solver in nn_solver_list:
        for nn_nodes in nn_nodes_list:
            nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                                index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data,
                                updates,
                                data_won, 'Real data')
            nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
            nn.make_predicitons()


for lr_solver in lr_solver_list:
    lr = LogRegModel(X_train, X_test, y_train, y_test, index_test,
                     index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data, updates,
                     data_won, 'Generated data')
    lr.define_model(solver=lr_solver, penalty='')
    lr.make_predicitons()

# svc = SVCModel(X_train, X_test, y_train, y_test, index_test,
#                  index_train, second_part, target, update_col, guessed_win_probabilities_for_test_data, updates,
#                  data_won, 'Real Data')
# svc.define_model(kernel=svc_kernel)
# data_after_svc = svc.make_predicitons()

sys.stdout.close()
