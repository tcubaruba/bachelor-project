import pandas as pd


def change_names(column):
    """
    Anonymizes the columns in the data by replacing the values with encoded ones
    :param column: column name
    :return:
    """
    old_names_list = real_data[column].unique()
    new_names = []
    for i in range(0, len(old_names_list)):
        name = column + '_' + str(i)
        new_names.append(name)
    col_dict = dict(zip(old_names_list, new_names))
    real_data[column] = real_data[column].replace(col_dict)


# this data has wrong stages, so they will be replaced
real_data_raw = pd.read_csv('./Data/Movement_Data.csv', engine='python', sep=';')

# this dataframe has correct stages
stage_data = pd.read_csv('./Data/Raw_Movement_Data.csv', engine='python', sep=';')

real_data_raw = real_data_raw.merge(stage_data, on=['OP Opportunity', 'DT Date'])

# select only columns that we need
real_data = real_data_raw[
    ['DT Date', 'OP Opportunity', 'Industry', 'Region N', 'Account', 'CA Campaign', 'Customer Type', 'OWN Owner',
     'SG Stage', 'Product Code', 'Price', 'Quantity', 'Value', 'Estimated Close Month (Step 1)', 'Created on']]

# rename columns such that they match to generated data
real_data.columns = ['Upload_date', 'Opportunity_Name', 'Industry', 'Region', 'Customer', 'Campaign', 'Customer_Type',
                     'Owner', 'Stage', 'Product', 'Price', 'Amount', 'Volume', 'Expected_closing', 'Created']

change_names('Industry')
change_names('Region')
change_names('Campaign')
change_names('Owner')
change_names('Product')

# convert stages
stages_dict = {'Closed Won': 'Won',
               'Closed Lost': 'Lost',
               '6. Final bid and negotiation': '6. Final bid',
               '2. Prospect & Account Building': '2. Prospect',
               '5. Validating benefits & value': '5. Value Proposition',
               '4. Identify pains and Building value pro': '4. Identify Pains',
               'On Hold': 'Lost',
               '7. Closing': '6. Final bid',
               '1. Marketing Converter': '1. Marketing'}
real_data['Stage'] = real_data['Stage'].replace(stages_dict)

# drop duplicates
real_data.drop_duplicates(inplace=True)

# compute expected closing in days
date_format = '%d.%m.%Y'
real_data['Upload_date'] = pd.to_datetime(real_data['Upload_date'], format=date_format)
real_data['Expected_closing'] = pd.to_datetime(real_data['Expected_closing'], format=date_format)
real_data['Expected_closing'] = (real_data['Expected_closing'] - real_data['Upload_date']).dt.days

# look for deal which were never opened
# find closed opportunities
closed = real_data[real_data['Stage'].isin(['Won', 'Lost'])]
open = real_data[~real_data['Stage'].isin(['Won', 'Lost'])]

closed_opps = closed['Opportunity_Name'].unique()
open_opps = open['Opportunity_Name'].unique()

# remove duplicates for closed opportunities
closed_no_duplicates = closed.loc[closed.groupby(['Opportunity_Name', 'Product'])['Upload_date'].idxmin()]
closed_no_duplicates['Expected_closing'] = 0
data = pd.concat([open, closed_no_duplicates])

# find and drop rows which were never opened
never_opened = closed_no_duplicates[~closed_no_duplicates['Opportunity_Name'].isin(open_opps)].index
data = data.drop(never_opened)

data.sort_values(by=['Upload_date', 'Opportunity_Name', 'Product'], inplace=True)

# find last stage
data['last_stage'] = data.groupby(['Opportunity_Name', 'Product'])['Stage'].shift()
# set first stage to '1.Marketing'
data['last_stage'] = data['last_stage'].fillna('1. Marketing')

data.to_csv('./Data/real_data_cleaned.csv')
