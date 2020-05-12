from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging
import time
import os
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Preprocessing data')
pd.options.mode.chained_assignment = None  # default='warn'


def prepare_data(data):
    data['Stage_simple'] = np.where(~data['Stage'].isin(['Won', 'Lost']), 'Open', data['Stage'])
    return data.drop(columns=['Price', 'Amount', 'last_stage', 'Stage'])


def timer(start, end):
    """
    Calculates the time difference and outputs it in formatted way
    :param start: start time
    :param end: end time
    :return: difference
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def scale_transform(to_change, to_stay, dict):
    """
    Transforms non-numerical values to numerical and scales the values
    :param to_change: table with non-numerical values
    :param to_stay: table with numerical values
    :param dict: dictionary for encoding
    :return: X with transformed values
    """
    df = to_change
    df_encoded = df.apply(lambda f: dict[f.name].fit_transform(f))
    x = df_encoded
    cat_vars = x.columns.tolist()
    dummies = x
    for var in cat_vars:
        cat_list = pd.get_dummies(df_encoded[var], prefix=var, drop_first=True)
        data1 = dummies.join(cat_list)
        dummies = data1
    data_vars = dummies.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]

    dummies = dummies[to_keep]
    X = dummies.join(to_stay, how='outer')
    # scaling values
    scaler = MinMaxScaler(copy=False)
    scaled_values = scaler.fit_transform(X)
    X.loc[:, :] = scaled_values
    return X


def preprocess(data, test_period, target, key_columns, update_col, created_col, opp_name_col, product_name_col):
    log.info('Start preprocessing data')
    stage_col = 'Stage_simple'
    start = time.time()
    data = data.copy()
    key_columns.append(stage_col)
    # sort data by update
    data = data.sort_values(by=update_col)
    data[created_col] = data[created_col].astype('datetime64[ns]')
    data[update_col] = data[update_col].astype('datetime64[ns]')
    # time difference to first update
    data['timediff'] = np.where(data[stage_col] != 'Open', (data[update_col] - data.groupby(key_columns)[
        update_col].transform('first')), (data[update_col] - data.groupby(key_columns)[
        update_col].transform('last')))

    # time difference from create datum
    data['timediff_since_create'] = data[update_col] - data[created_col]

    # time differences in days
    data['timediff'] = (data['timediff'].dt.components['days']).astype(int)
    data['timediff_since_create'] = (data['timediff_since_create'].dt.components['days']).astype(int)

    # time to close (won or lost)
    data['time_diff_to_close'] = np.where(data[stage_col] == 'Open', data.groupby(key_columns)['timediff'].transform(
        'last') - data['timediff'], data.groupby(key_columns)['timediff'].transform(
        'first') - data['timediff'])

    # take only first occurrence of an opportunity
    data_no_duplicates = data.loc[data['timediff'] == 0]

    # create key value
    data_no_duplicates['key_val'] = data_no_duplicates[opp_name_col].map(str) + ' ' + data_no_duplicates[
        product_name_col]

    # split data and reset indexes
    data_won = data_no_duplicates.loc[data_no_duplicates[stage_col] == 'Won']
    data_lost = data_no_duplicates.loc[data_no_duplicates[stage_col] == 'Lost']
    data_closed = data_no_duplicates.loc[data_no_duplicates[stage_col] != 'Open']

    # create temp tables with only key values and date
    data_won_temp = data_won[['key_val', update_col]]
    data_lost_temp = data_lost[['key_val', update_col]]
    data_closed_temp = data_closed[['key_val', update_col]]

    # define the future stage
    data_no_duplicates['future stage'] = np.where(data_no_duplicates['key_val'].isin(data_won_temp['key_val']), 'Won',
                                                  np.where(
                                                      data_no_duplicates['key_val'].isin(data_lost_temp['key_val']),
                                                      'Lost', 'Open'))

    data_no_duplicates['future stage'] = np.where(data_no_duplicates[stage_col] != 'Open', 'none',
                                                  data_no_duplicates['future stage'])

    # get close date for closed opportunities
    data_no_duplicates['close date'] = data_no_duplicates['key_val'].map(
        data_closed_temp.set_index('key_val')[update_col])

    # set max date to last date of update
    max_date = data[update_col].max()
    max_date = pd.to_datetime(max_date).date()
    data_no_duplicates['close date'] = np.where(data_no_duplicates['close date'].isnull(), max_date,
                                                data_no_duplicates['close date'].dt.date)
    data_no_duplicates['close date'] = data_no_duplicates['close date'].astype('datetime64[ns]')

    # recompute  time difference to closing by days
    data_no_duplicates['time_diff_to_close'] = ((data_no_duplicates['close date'] - data_no_duplicates[update_col])
        .dt.components['days']).astype(int)

    data_no_duplicates[stage_col] = np.where(data_no_duplicates['time_diff_to_close'] < 0,
                                               data_no_duplicates['future stage'], data_no_duplicates[stage_col])

    # delete unnecessary columns
    data_first_part = data_no_duplicates.drop(
        columns=['timediff', 'key_val', 'time_diff_to_close', 'close date'])
    data_second_part = data_no_duplicates.drop(columns=['timediff', 'key_val', 'close date'])

    # FIRST PART
    # data to make predictions for
    open_data = data_first_part.loc[data_first_part[stage_col] == 'Open']

    # change future stage open to lost to make it easier to make the predictions
    open_data.loc[:, 'future stage'] = np.where(open_data['future stage'] == 'Open', 'Lost', open_data['future stage'])

    # change values in target column to numeric
    open_data.loc[:, 'future stage'] = np.where(open_data['future stage'] == 'Lost', 0, 1)

    # delete stage new column, because unnecessary
    open_data = open_data.drop(columns=stage_col)

    # get indices of train and test data
    index_train = open_data.index[open_data[update_col] < test_period]
    index_test = open_data.index[open_data[update_col] >= test_period]

    open_data = open_data.drop(columns=update_col)

    # convert datatypes
    to_change = open_data.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
    to_stay = open_data.select_dtypes(include='number')  # numerical data
    dict_LE = defaultdict(LabelEncoder)  # to be able to decode later
    y = open_data[target]
    to_stay = to_stay.drop(columns=target)
    X = scale_transform(to_change, to_stay, dict_LE)

    # SECOND PART
    open_data_second_part = data_second_part.loc[data_second_part[stage_col] == 'Open']
    # change future stage open to lost to make it easier to make the predictions
    open_data_second_part.loc[:, 'future stage'] = np.where(open_data_second_part['future stage'] == 'Open', 'Lost',
                                                            open_data_second_part['future stage'])

    # change values in target column to numeric
    open_data_second_part.loc[:, 'future stage'] = np.where(open_data_second_part['future stage'] == 'Lost', 0, 1)

    end = time.time()
    log.info('Preprocessing data finished. Execution time: ' + str(timer(start, end)))
    return X, y, index_train, index_test, dict_LE, open_data_second_part
