# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechat: whatshowlove
@ software: PyCharm
@ file: data_exploratory
@ time: 18-3-17
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
import seaborn as sns

DATAPATH = '../data/train.csv'
ECONOMICALDATA = '../data/macro.csv'

train_df = pd.read_csv(DATAPATH)
macro_df = pd.read_csv(ECONOMICALDATA)
rows, cols = train_df.shape
read_columns = ['timestamp', 'oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', \
                'salary_growth', 'unemployment', 'average_provision_of_build_contract_moscow', 'mortgage_rate', \
                'deposits_rate', 'deposits_growth', 'rent_price_3room_eco', \
                'rent_price_3room_bus']
# 统计信息
print train_df.describe()


# house price distribute
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.savefig("price_dis.png")
plt.show()

# missing value count
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.savefig("miss_value.png")
plt.show()

# importance feature calculate
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))

train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.savefig("feature_importance.png")
plt.show()

# correlation analysis
###### Service Read routines ###
def condition_train(value, col):
    vals = (macro_df[macro_df['mo_ye'] == value])
    ret = vals[col].asobject
    ret = ret[0]
    return ret


def condition_test(value, col):
    vals = (macro_df[macro_df['mo_ye'] == value])
    ret = vals[col].asobject
    ret = ret[0]
    return ret


def condition(value,col):
    vals = (macro_df[macro_df['timestamp'] == value])
    ret = vals[col].asobject
    # if ret.shape[0]>0:
    ret = ret[0]
    return ret


def init_anlz_file():

    anlz_df = train_df
    for clmn in read_columns:
        if clmn == 'timestamp':
            continue
        anlz_df[clmn] = np.nan
        anlz_df[clmn] = anlz_df['timestamp'].apply(condition, col=clmn)
        print(clmn)
    return anlz_df

### Read Data for macro analysis
anlz_df=init_anlz_file()
anlz_df['timestamp']=pd.to_datetime(anlz_df['timestamp'])
anlz_df['mo_ye']=anlz_df['timestamp'].apply(lambda x: x.strftime('%m-%Y'))
anlz_df['price_per_sqm']=anlz_df['price_doc']/anlz_df['full_sq']


macro_columns = ['price_doc','price_per_sqm','full_sq','oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', \
                'salary_growth', 'unemployment', 'average_provision_of_build_contract_moscow', 'mortgage_rate', \
                 'deposits_rate','deposits_growth','rent_price_3room_eco',\
                 'rent_price_3room_bus']
macro_df=pd.DataFrame(anlz_df.groupby('mo_ye')[macro_columns].mean())
macro_df.reset_index(inplace=True)


macro_df['mo_ye']=pd.to_datetime(macro_df['mo_ye'])
macro_df=macro_df.sort_values(by='mo_ye')

# draw heatmap
macro_df.reset_index(inplace=True)
macro_df.drop(['index'],axis=1,inplace=True)
corr = macro_df[:].corr(method='spearman')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
plt.savefig("feature_relative.png")
plt.show()