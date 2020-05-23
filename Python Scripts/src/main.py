import pandas as pd
import logging
import os
import sys

from src.preprocess import preprocess
from models.NeuralNetModel import NeuralNetModel
from models.LogRegModel import LogRegModel
from models.SVCModel import SVCModel

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Predicting win probability for sales data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print('generated data'.upper())
data_raw = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)

# Read parameters
_, test_period, nn_nodes, lr_solver, nn_solver, nn_activation, svc_kernel, key_columns, update_col, created_col, opp_name_col, product_name_col = sys.argv
key_columns = key_columns[1:-1].split(',')
target = 'future stage'
target_second_part = 'time_diff_to_close'

X, y, index_train, index_test, dict_LE, second_part = preprocess(data_raw, test_period, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

X_train = X.loc[index_train]
X_test = X.loc[index_test]
y_train = y.loc[index_train]
y_test = y.loc[index_test]

nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part, target, test_period, update_col)
nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
data_after_nn, acc_nn = nn.make_predicitons()

print(acc_nn)
# print(data_after_nn.loc[index_test])

lr = LogRegModel(X_train, X_test, y_train, y_test,index_test,
                 index_train, second_part, target, test_period, update_col)
lr.define_model(solver=lr_solver, penalty='')
data_after_lr, acc_lr = lr.make_predicitons()

print(acc_lr)
# print(data_after_lr.loc[index_test])

svc = SVCModel(X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part, target, test_period, update_col)
svc.define_model(kernel=svc_kernel)
data_after_svc, acc_svc = svc.make_predicitons()

print(acc_svc)
# print(data_after_svc.loc[index_test])


print('real data'.upper())
data_raw = pd.read_csv('./Data/real_data_cleaned.csv', index_col=0)

# Read parameters
_, test_period, nn_nodes, lr_solver, nn_solver, nn_activation, svc_kernel, key_columns, update_col, created_col, opp_name_col, product_name_col = sys.argv
key_columns = key_columns[1:-1].split(',')
target = 'future stage'
target_second_part = 'time_diff_to_close'

X, y, index_train, index_test, dict_LE, second_part = preprocess(data_raw, test_period, target, key_columns, update_col, created_col, opp_name_col, product_name_col)

X_train = X.loc[index_train]
X_test = X.loc[index_test]
y_train = y.loc[index_train]
y_test = y.loc[index_test]

nn = NeuralNetModel(X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part, target, test_period, update_col)
nn.define_model(solver=nn_solver, activation=nn_activation, n_nodes=nn_nodes)
data_after_nn, acc_nn = nn.make_predicitons()

print(acc_nn)
# print(data_after_nn.loc[index_test])

lr = LogRegModel(X_train, X_test, y_train, y_test,index_test,
                 index_train, second_part, target, test_period, update_col)
lr.define_model(solver=lr_solver, penalty='')
data_after_lr, acc_lr = lr.make_predicitons()

print(acc_lr)
# print(data_after_lr.loc[index_test])

svc = SVCModel(X_train, X_test, y_train, y_test, index_test,
                 index_train, second_part, target, test_period, update_col)
svc.define_model(kernel=svc_kernel)
data_after_svc, acc_svc = svc.make_predicitons()

print(acc_svc)
# print(data_after_svc.loc[index_test])
