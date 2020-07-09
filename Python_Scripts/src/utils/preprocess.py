from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging
import time
import os
import warnings
import random
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger('Preprocessing data')
pd.options.mode.chained_assignment = None  # default='warn'
random.seed(42)


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


def preprocess(data, target, key_columns, update_col, created_col, opp_name_col, product_name_col):
    """
    Prepare data for predictions
    :param data: pandas data frame to prepare
    :param target: column to predict
    :param key_columns: columns which make key value
    :param update_col: column name for updates
    :param created_col: column name for date of creation
    :param opp_name_col: column name for the name of opportunity
    :param product_name_col: column name for the name of product
    :return:
    """
    log.info('Start preprocessing data')
    start = time.time()

    # First create column "Stage_simple", which has only three values: Open, Won or Lost. This will be predicted
    stage_col = 'Stage_simple'
    data[stage_col] = np.where(~data['Stage'].isin(['Won', 'Lost']), 'Open', data['Stage'])
    # create pivot table for products for more compact view of which products each opportunity has
    pivot_products = pd.pivot_table(data[[opp_name_col, product_name_col, update_col]],
                                    index=[opp_name_col, update_col],
                                    columns=[product_name_col],
                                    aggfunc=[len],
                                    fill_value=0
                                    )
    # The initial column with product name, as well as its price and amount is not needed. Only the entire volume
    # of opportunity is interesting. The volume will be aggregated
    data = data.drop(columns=[product_name_col, 'Price', 'Amount'])
    columns_left = list(data.columns)
    columns_left.remove('Volume')
    columns_right = list(pivot_products.columns)
    columns = columns_left + columns_right
    data = data.join(pivot_products, on=[opp_name_col, update_col]).groupby(columns)['Volume'].sum().to_frame()
    data = pd.DataFrame(data.to_records())

    key_columns.append(stage_col)
    # sort data by update
    data = data.sort_values(by=update_col)
    data[created_col] = data[created_col].astype('datetime64[ns]')
    data[update_col] = data[update_col].astype('datetime64[ns]')
    # time difference to first update
    data['timediff'] = np.where(data[stage_col] != 'Open', (data[update_col] - data.groupby(opp_name_col)[
        update_col].transform('first')), (data[update_col] - data.groupby(opp_name_col)[
        update_col].transform('last')))

    # time difference from create datum
    data['timediff_since_create'] = data[update_col] - data[created_col]

    # time differences in days
    data['timediff'] = (data['timediff'].dt.components['days']).astype(int)
    data['timediff_since_create'] = (data['timediff_since_create'].dt.components['days']).astype(int)

    # time to close (won or lost)
    data['time_diff_to_close'] = np.where(data[stage_col] == 'Open', data.groupby(opp_name_col)['timediff'].transform(
        'last') - data['timediff'] + 7, data.groupby(opp_name_col)['timediff'].transform(
        'first') - data['timediff'])

    # split data and reset indexes
    data_won = data.loc[data[stage_col] == 'Won']
    data_lost = data.loc[data[stage_col] == 'Lost']

    # create temp tables with only key values and date
    data_won_temp = data_won[[opp_name_col, update_col]]
    data_lost_temp = data_lost[[opp_name_col, update_col]]

    # define the future stage
    data['future stage'] = np.where(data[opp_name_col].isin(data_won_temp[opp_name_col]),
                                                  'Won',
                                                  np.where(
                                                      data[opp_name_col].isin(
                                                          data_lost_temp[opp_name_col]),
                                                      'Lost', 'Open'))

    data['future stage'] = np.where(data[stage_col] != 'Open', 'none',
                                                  data['future stage'])

    # drop rows which are still not closed, because we can not check the prediction accuracy for them
    data = data.loc[data['future stage'] != 'Open']

    # delete unnecessary columns
    data_first_part = data.drop(
        columns=['timediff', created_col])

    # FIRST PART
    # data to make predictions for
    open_data = data_first_part.loc[data_first_part[stage_col] == 'Open']

    # change values in target column to numeric
    open_data.loc[:, 'future stage'] = np.where(open_data['future stage'] == 'Lost', 0, 1)

    # need time difference for the second part predictions
    updates = open_data[update_col]  # need later
    open_data_second_part = open_data.drop(columns=[update_col, stage_col])
    open_data = open_data_second_part.drop(columns='time_diff_to_close')

    # get indices of train and test data
    opps_list = open_data[opp_name_col].unique()
    test_opps = random.sample(list(opps_list), int(0.3 * len(opps_list)))
    index_test = open_data[open_data[opp_name_col].isin(test_opps)].index
    index_train = open_data.drop(index_test).index

    # set win probabilities
    probability_dict = {
        '1. Marketing': 0.05,
        '2. Prospect': 0.1,
        '3. Discovery': 0.25,
        '4. Identify Pains': 0.5,
        '5. Value Proposition': 0.8,
        '6. Final bid': 0.9,
        'Won': 1.,
        'Lost': 0.
    }
    data['Estimated_win_probability'] = data['Stage'].replace(probability_dict)
    guessed_win_probabilities = data['Estimated_win_probability']
    guessed_win_probabilities_for_test_data = guessed_win_probabilities.loc[index_test]

    # convert datatypes
    to_change = open_data.select_dtypes(include=['object', 'datetime'])  # data which need to be encoded
    to_stay = open_data.select_dtypes(include='number')  # numerical data
    dict_LE = defaultdict(LabelEncoder)  # to be able to decode later
    y = open_data[target]
    to_stay = to_stay.drop(columns=target)
    X = scale_transform(to_change, to_stay, dict_LE)

    end = time.time()
    log.info('Preprocessing data finished. Execution time: ' + str(timer(start, end)))
    return X, y, index_train, index_test, open_data_second_part, guessed_win_probabilities_for_test_data, updates.loc[
        index_test], data_won
