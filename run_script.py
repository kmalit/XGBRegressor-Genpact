import pandas as pd
import numpy as np
# Fit models
import xgboost as xgb
# Helper functions
from helperfuncs.feature_eng import fit_categorical #helperfuncs.

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

###############################################################################
# load data
###############################################################################
df1 = pd.read_csv('./data/train.csv')
df2 = pd.read_csv('./data/fulfilment_center_info.csv')
df3 = pd.read_csv('./data/meal_info.csv')

###############################################################################
# Data Merge
###############################################################################

def merge(df1,df2,df3):
    df = df1.merge(df2, how='left', on=['center_id']).merge(df3, how='left', on=['meal_id'])
    return df
    
###############################################################################
# Clean Data
###############################################################################
    
def clean_df(df):
    # Drop the columns as explained above
    df = df.drop(['week',"id", "center_id", "meal_id"], axis = 1)
    # Add variables
    ## discount amount
    df['discount'] = 1 - df.checkout_price/df.base_price 
    ## discount indicator
    def dis_ind(row):
        if row['checkout_price'] < row['base_price']:
            return 1
        else:
            return 0
    df['dis_ind'] = df.apply (lambda row: dis_ind (row),axis=1)
    # Arrange columns by data type for easier manipulation
    continuous_vars = ['checkout_price', 'base_price','discount']
    cat_vars = ['dis_ind','emailer_for_promotion', 'homepage_featured','city_code', 'region_code','center_type',
                'op_area','cuisine','category']
    df = df[['num_orders'] + continuous_vars + cat_vars]
    return df

def clean_testdf(df):
    # Drop the columns as explained above
    df = df.drop(['week',"id", "center_id", "meal_id"], axis = 1)
    # Add variables
    ## discount amount
    df['discount'] = 1 - df.checkout_price/df.base_price 
    ## discount indicator
    def dis_ind(row):
        if row['checkout_price'] < row['base_price']:
            return 1
        else:
            return 0
    df['dis_ind'] = df.apply (lambda row: dis_ind (row),axis=1)
    continuous_vars = ['checkout_price', 'base_price','discount']
    cat_vars = ['dis_ind','emailer_for_promotion', 'homepage_featured','city_code', 'region_code','center_type',
                'op_area','cuisine','category']
    df = df[continuous_vars + cat_vars]
    return df

###############################################################################
# Data Transformation 
###############################################################################

# Clean Data
df =  merge(df1,df2,df3)
df = clean_df(df)


# VARIABLE ENCODING

cat_var = ['dis_ind','emailer_for_promotion', 'homepage_featured','city_code', 
           'region_code','center_type','op_area','cuisine','category']

cat_enc = fit_categorical(df,cat_var,'num_orders')
# Label Encoding
df_labelenc = cat_enc.label_encoding()

# frequency encoding
df_freq = cat_enc.freq_encoding()
        
# target mean Encoding
df_tme = cat_enc.target_mean_encoding()

###############################################################################
# MODELS TO FIT
###############################################################################
# Fit XGB Regressor

param = {'objective':['reg:linear'],
         'silent':[1], 
         'nthread': [10],
         'learning_rate': [0.05,0.1],
         'gamma':[0.001],
         'max_depth': [7,10,20], 
         'n_estimators':[100,150],
         'min_child_weight':[5,10,20],
         'reg_lambda':[1,0.1,10,100],
         'subsample': [0.9],
         'colsample_bytree': [1.0],
         'seed': [101]}

def rmsle(real, predicted):
    diffs = np.log((predicted + 1)/(real + 1))
    s_diff = np.square(diffs)
    avg = np.mean(s_diff)
    sqrt = np.sqrt(avg)
    rmsle_100 = 100 * sqrt
    return (rmsle_100)


###############################################################################
# Grid search
###############################################################################
'''
scoring = {'accuracy': make_scorer(rmsle)}
model = xgb.XGBRegressor()

DATA = [df_labelenc,df_freq,df_tme]
DATA_LABEL = ['df_labelenc','df_freq','df_tme']
i = 0
for data in DATA:
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values
    grid = GridSearchCV(estimator=model, param_grid=param, scoring=scoring, 
                        error_score=0.0,cv =5,n_jobs=10,refit=False,verbose=2)
    grid.fit(X,y)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(DATA_LABEL[i]+'_results.csv', index=False)
    i =+1
'''
#Final model
X,y = df_freq.iloc[:,1:], df_freq.iloc[:,0]
X = X.convert_objects(convert_numeric=True)
dtrain = xgb.DMatrix(data = X, label = y,missing=0.0,feature_names = list(X.columns), nthread=-1)
dtrain.save_binary('train.buffer')

modelf = xgb.XGBRegressor(objective='reg:linear',silent=1, nthread=2, 
                          gamma= 0.001, learning_rate= 0.05,
                          max_depth= 7, min_child_weight= 5,
                          n_estimators= 100, reg_lambda= 100,
                          seed= 101, subsample= 0.9)
modelf.fit(X,y)

###############################################################################
# TEST SCORE
###############################################################################

# Data Transformation
test = pd.read_csv('./data/test_QoiMO9B.csv')
test_df = merge(test,df2,df3)
test_df = clean_testdf(test_df)
test_df = cat_enc.ff_req_encoding(test_df)
test_df = test_df.convert_objects(convert_numeric=True)
# Model Scoring
predictions = modelf.predict(test_df)
predictions = np.clip(predictions, a_min = 0, a_max = None)

# Model submission
submission = pd.DataFrame({ 'ID': test['id'],'num_orders': predictions })
submission.to_csv("submission.csv", index=False)
