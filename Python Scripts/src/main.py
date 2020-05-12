import pandas as pd
import pypyodbc
import numpy as np
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Predicting win probability for sales data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('./Data/data_complete_no_closed_duplicates.csv', index_col=0)

print(data.head())
