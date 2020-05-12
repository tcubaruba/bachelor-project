import pandas as pd
import pypyodbc
import numpy as np
import logging
import os
import sys

from src.preprocess import prepare_data
from src.preprocess import preprocess

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Predicting win probability for sales data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_raw = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)

# Read parameters
_, test_period, nn_nodes, lr_solver, nn_solver, nn_activation, svc_kernel, key_columns, update_col, created_col, opp_name_col, product_name_col = sys.argv
key_columns = key_columns[1:-1].split(',')
target = 'future stage'
target_second_part = 'time_diff_to_close'

data = prepare_data(data_raw)
print(data.head())

X, y, index_train, index_test, dict_LE, second_part = preprocess(data, test_period, target, key_columns, update_col, created_col, opp_name_col, product_name_col)
