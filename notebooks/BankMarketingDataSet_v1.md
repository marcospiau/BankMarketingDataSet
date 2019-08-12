
# Initial imports


```
# Numeric and data processing
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from pandas.api.types import CategoricalDtype

# sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.metrics import make_scorer
from scipy.stats import ks_2samp
from sklearn_pandas import DataFrameMapper
from  sklearn.linear_model import Lasso, LassoCV, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, chi2, SelectKBest

# plotting
import matplotlib.pyplot as plt
%matplotlib inline
```


```
# Global random state for reproducibility
random_state_global = 42
```

# Dataprep

## Reading data from csv

8. Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques. 


```
df_full = pd.read_csv('../data/bank-additional/bank-additional-full.csv', sep=';', na_values=['unknown'])
df_full.columns = df_full.columns.str.replace('.', '_')
```


```
df_full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp_var_rate</th>
      <th>cons_price_idx</th>
      <th>cons_conf_idx</th>
      <th>euribor3m</th>
      <th>nr_employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>261</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>NaN</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>149</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>226</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>151</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>307</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```
# Is there missing values on data? YES, I forced it using a_values=['unknown'] using 
df_full.isnull().sum()
```




    age                  0
    job                330
    marital             80
    education         1731
    default           8597
    housing            990
    loan               990
    contact              0
    month                0
    day_of_week          0
    duration             0
    campaign             0
    pdays                0
    previous             0
    poutcome             0
    emp_var_rate         0
    cons_price_idx       0
    cons_conf_idx        0
    euribor3m            0
    nr_employed          0
    y                    0
    dtype: int64



## Categorical and numeric data preprocessing

## Categorical data

Some categorical have a meaningful ordering, like education ('basic.4y' < 'basic.6y',  ...) and default ('yes' > 'no'). For these
we will use this information while creating the category data type; for the others, there is no meaningful order, and we will not specify order while creating the category data type.


```
# Frequency of categories for each categorical feature
for col in df_full.select_dtypes('O').columns:
    print('\n', col)
    print(df_full[col].value_counts(dropna=False))
```

    
     job
    admin.           10422
    blue-collar       9254
    technician        6743
    services          3969
    management        2924
    retired           1720
    entrepreneur      1456
    self-employed     1421
    housemaid         1060
    unemployed        1014
    student            875
    NaN                330
    Name: job, dtype: int64
    
     marital
    married     24928
    single      11568
    divorced     4612
    NaN            80
    Name: marital, dtype: int64
    
     education
    university.degree      12168
    high.school             9515
    basic.9y                6045
    professional.course     5243
    basic.4y                4176
    basic.6y                2292
    NaN                     1731
    illiterate                18
    Name: education, dtype: int64
    
     default
    no     32588
    NaN     8597
    yes        3
    Name: default, dtype: int64
    
     housing
    yes    21576
    no     18622
    NaN      990
    Name: housing, dtype: int64
    
     loan
    no     33950
    yes     6248
    NaN      990
    Name: loan, dtype: int64
    
     contact
    cellular     26144
    telephone    15044
    Name: contact, dtype: int64
    
     month
    may    13769
    jul     7174
    aug     6178
    jun     5318
    nov     4101
    apr     2632
    oct      718
    sep      570
    mar      546
    dec      182
    Name: month, dtype: int64
    
     day_of_week
    thu    8623
    mon    8514
    wed    8134
    tue    8090
    fri    7827
    Name: day_of_week, dtype: int64
    
     poutcome
    nonexistent    35563
    failure         4252
    success         1373
    Name: poutcome, dtype: int64
    
     y
    no     36548
    yes     4640
    Name: y, dtype: int64
    

## Downcasting numeric features to reduce memory usage
(inspired on https://www.kaggle.com/gemartin/load-data-reduce-memory-usage)


```
# Dict storing datatypes of all features
dict_dtypes = {}

# Categorical features with meaningful ordering
dict_dtypes['education'] = CategoricalDtype(categories = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                                          'professional.course', 'university.degree']
                                            , ordered=True)

dict_dtypes['default'] = CategoricalDtype(categories = ['no', 'yes'], ordered=True)
dict_dtypes['housing'] = CategoricalDtype(categories = ['no', 'yes'], ordered=True)
dict_dtypes['loan'] = CategoricalDtype(categories = ['no', 'yes'], ordered=True)
dict_dtypes['poutcome'] = CategoricalDtype(categories = ['failure', 'success'], ordered=True)# nonexistent considered as missing value
dict_dtypes['y'] = CategoricalDtype(categories = ['no', 'yes'], ordered=True)

# Polemic
dict_dtypes['day_of_week'] = CategoricalDtype(categories = ['mon', 'tue', 'wed', 'thu', 'fri'], ordered=True)
dict_dtypes['month'] = CategoricalDtype(categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                                                     'sep', 'oct', 'nov', 'dec'], ordered=True)
# Other categorical features
for col in df_full.select_dtypes('O').columns:
    if col not in dict_dtypes.keys():
        dict_dtypes[col] = CategoricalDtype(categories = sorted(df_full[col].dropna().unique()), ordered=False)
```


```
# Note: pandas alerady have nullable integer datatypes, but I will not use. If a integer column have at least a missing value,
# it will be converted to float32.
for col in df_full.select_dtypes(np.number):
    _vec_min_max = df_full[col].describe()[['min','max']]
    _has_null = df_full[col].isnull().max()
    _has_float = (df_full[col] % 1 != 0).any()
    
    if _has_float or _has_null:
        dict_dtypes[col] = pd.to_numeric(_vec_min_max, downcast='float').dtype
    else:
        if _vec_min_max[0] >=0:
            dict_dtypes[col] = pd.to_numeric(_vec_min_max, downcast='unsigned').dtype
        else:
            dict_dtypes[col] = pd.to_numeric(_vec_min_max, downcast='signed').dtype

start_memory_usage = df_full.memory_usage().sum() / 1024**2
end_memory_usage = df_full.astype(dict_dtypes).memory_usage().sum() / 1024**2

print('Initial memory usage: {:.2f} MB'.format(start_memory_usage))
print('Memory usage after optimization: {:.2f} MB'.format(end_memory_usage))
print('Memory usage decreased by {:.1f}%'.format(100 * (start_memory_usage - end_memory_usage) / start_memory_usage))
```

    Initial memory usage: 6.60 MB
    Memory usage after optimization: 1.49 MB
    Memory usage decreased by 77.3%
    

## Applying new dtypes inplace on initial dataframe


```
df_full = df_full.astype(dict_dtypes)
```

# Modelling

## Train test split


```
# I used pd.get_dummies before splitting instead of including then on pipeline for simplicity
X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(df_full.drop('y', axis=1), dummy_na=True), df_full['y'].cat.codes,
                                                    test_size=0.20, random_state=random_state_global)
```


```
df_full.y
```




    0         no
    1         no
    2         no
    3         no
    4         no
            ... 
    41183    yes
    41184     no
    41185     no
    41186    yes
    41187     no
    Name: y, Length: 41188, dtype: category
    Categories (2, object): [no < yes]




```
num_cols = df_full.select_dtypes(np.number).columns.to_list()
cat_cols = df_full.drop(['y'], axis=1).select_dtypes('category').columns.to_list()
```

## KS and gini functions and sklearn scorers


```
def ks_stat(y_true, y_proba):
#     As seen on https://medium.com/@xiaowei_6531/using-ks-stat-as-a-model-evaluation-metric-in-scikit-learns-gridsearchcv-33135101601c
    return ks_2samp(y_proba[y_true==1], y_proba[y_true!=1]).statistic

ks_scorer = make_scorer(ks_stat, needs_proba=True, greater_is_better=True)

#Remove redundant calls
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

gini_scorer = make_scorer(gini_normalizedc, needs_proba=True, greater_is_better=True)
```

## Basic pipeline


```
# KFold for hyperparameter tuning and acessing model quality
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_global)
```


```
def get_cat_codes(X, return_df=True):
    if return_df:
        return X.apply(lambda x: x.cat.codes)
    else:
        return X.apply(lambda x: x.cat.codes).values

def from_cat_to_str(X):
    return X.astype(str)
    
def f_select_dtypes(X, dtype):
    return X.select_dtypes()

select_num_transformer = FunctionTransformer(lambda x: x.select_dtypes(np.number), validate=False)
select_cat_transformer = FunctionTransformer(lambda x: x.select_dtypes('category'), validate=False)
get_cat_features_transformer = FunctionTransformer(get_cat_codes, validate=False)
from_cat_to_str_transformer = FunctionTransformer(from_cat_to_str, validate=False)
```


```
# If would like to use OneHotEncoder inside pipeline (NOT USED, just for reference!!!)
# Using sklearn pipelines
linear_model_pipeline = FeatureUnion([
    ('num_feat', make_pipeline(select_num_transformer ,StandardScaler())),
    ('cat_feat', make_pipeline(select_cat_transformer ,get_cat_features_transformer, OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')))
     ])

# Using DataFrameMapper from sklearn_pandas (Using OneHotEncoder instead of pd.get_dummies)
# TODO: remove x0 on feature encoded on OneHotEncoder
mapper = DataFrameMapper(
    [([col], StandardScaler()) for col in num_cols] + 
    [([col], [get_cat_features_transformer, OneHotEncoder(dtype=np.int8, sparse=False, handle_unknown='ignore', categories='auto')]) for col in cat_cols]
     , input_df=True, df_out=True)
# mapper.fit_transform(df_full)
```


```
lin_pipe_df_mapper = DataFrameMapper(
    [([col], StandardScaler()) for col in num_cols]
     , input_df=True, df_out=False, default=None)
lin_pipe_df_mapper.fit_transform(X_train)
```




    array([[-1.66930454e-03, -6.31114175e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [-8.64094846e-01, -5.46321353e-01,  5.13675879e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [ 1.81900684e+00, -9.43305926e-01,  1.23359337e+00, ...,
             1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           ...,
           [-4.80794606e-01, -2.45692259e-01, -5.66200360e-01, ...,
             0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [-1.66930454e-03,  1.43583876e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [-1.05574497e+00,  2.90044203e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])



## Feature selection


```
VarianceThreshold()
```




    VarianceThreshold(threshold=0.0)




```
lasso_pipe = SelectFromModel(make_pipeline(linear_dataframe_mapper, VarianceThreshold(0), LassoCV(cv=kf, max_iter=1500)))
lasso_pipe.fit(X_train[num_cols], y_train)
```




    SelectFromModel(estimator=Pipeline(memory=None,
                                       steps=[('dataframemapper',
                                               DataFrameMapper(default=None,
                                                               df_out=False,
                                                               features=[(['age'],
                                                                          StandardScaler(copy=True,
                                                                                         with_mean=True,
                                                                                         with_std=True)),
                                                                         (['duration'],
                                                                          StandardScaler(copy=True,
                                                                                         with_mean=True,
                                                                                         with_std=True)),
                                                                         (['campaign'],
                                                                          StandardScaler(copy=True,
                                                                                         with_mean=True,
                                                                                         with_std=True)),
                                                                         (['pdays'],
                                                                          StandardS...
                                              ('lassocv',
                                               LassoCV(alphas=None, copy_X=True,
                                                       cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
                                                       eps=0.001,
                                                       fit_intercept=True,
                                                       max_iter=1500, n_alphas=100,
                                                       n_jobs=None, normalize=False,
                                                       positive=False,
                                                       precompute='auto',
                                                       random_state=None,
                                                       selection='cyclic',
                                                       tol=0.0001,
                                                       verbose=False))],
                                       verbose=False),
                    max_features=None, norm_order=1, prefit=False, threshold=None)




```
(linear_dataframe_mapper.fit_transform(X_train))
```




    array([[-1.66930454e-03, -6.31114175e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [-8.64094846e-01, -5.46321353e-01,  5.13675879e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [ 1.81900684e+00, -9.43305926e-01,  1.23359337e+00, ...,
             1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           ...,
           [-4.80794606e-01, -2.45692259e-01, -5.66200360e-01, ...,
             0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [-1.66930454e-03,  1.43583876e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [-1.05574497e+00,  2.90044203e-01, -2.06241614e-01, ...,
             0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])




```
lasso_pipe = SelectFromModel(LogisticRegressionCV(cv=kf, max_iter=1000, penalty='elasticnet', solver='saga',
                                                  l1_ratios=[0, 0.5, 1],  class_weight='balanced', n_jobs=10), threshold=1e-5)
lasso_pipe.fit(linear_dataframe_mapper.fit_transform(X_train), y_train)
```




    SelectFromModel(estimator=LogisticRegressionCV(Cs=10, class_weight='balanced',
                                                   cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
                                                   dual=False, fit_intercept=True,
                                                   intercept_scaling=1.0,
                                                   l1_ratios=[0, 0.5, 1],
                                                   max_iter=1000,
                                                   multi_class='warn', n_jobs=10,
                                                   penalty='elasticnet',
                                                   random_state=None, refit=True,
                                                   scoring=None, solver='saga',
                                                   tol=0.0001, verbose=0),
                    max_features=None, norm_order=1, prefit=False, threshold=1e-05)




```
chi2(X_train.drop(num_cols, axis=1), y_train)
```




    (array([2.27720632e+01, 1.34898023e+02, 8.82023835e+00, 2.42593188e+00,
            2.39817213e-03, 2.48189186e+02, 2.12763785e+00, 2.94533736e+01,
            2.72112359e+02, 4.30539700e-02, 6.79948732e+00, 4.57872713e-01,
            3.05747220e+00, 2.37935310e+01, 6.68697317e+01, 2.13840768e-01,
            1.82379370e+00, 2.76232127e+00, 1.89901219e+01, 5.37700465e+01,
            2.33712987e+00, 4.95914083e-03, 6.71829826e+01, 8.13035832e+00,
            6.75414179e+01, 3.80064968e-01, 2.56573109e+02, 2.53651404e+00,
            1.85810550e+00, 2.85690886e-01, 3.01328573e-03, 1.17653917e-01,
            2.85690886e-01, 2.60345684e+02, 4.53390092e+02,            nan,
                       nan,            nan, 7.18647423e+02, 1.77895148e+02,
            2.76751982e+02, 1.95681135e+00, 2.43064024e+01, 2.05290371e+00,
            5.15499552e+02, 6.66905826e+02, 4.75014988e+00, 1.62803424e+02,
                       nan, 1.01191077e+01, 1.74288844e+00, 4.33983392e-01,
            6.60240895e+00, 1.94745344e+00,            nan, 2.76037264e+01,
            3.23738075e+03, 1.68023615e+02]),
     array([1.82397987e-006, 3.47569774e-031, 2.97907749e-003, 1.19342253e-001,
            9.60942299e-001, 6.44494253e-056, 1.44663538e-001, 5.72782484e-008,
            3.93130221e-061, 8.35623683e-001, 9.11840524e-003, 4.98619845e-001,
            8.03666048e-002, 1.07242280e-006, 2.90054419e-016, 6.43773561e-001,
            1.76862365e-001, 9.65081139e-002, 1.31396937e-005, 2.25382775e-013,
            1.26322108e-001, 9.43858444e-001, 2.47440790e-016, 4.35301435e-003,
            2.06307313e-016, 5.37568468e-001, 9.58330476e-058, 1.11240139e-001,
            1.72843612e-001, 5.92995238e-001, 9.56223387e-001, 7.31593068e-001,
            5.92995238e-001, 1.44266500e-058, 1.31930117e-100,             nan,
                        nan,             nan, 2.63496471e-158, 1.39639882e-040,
            3.83186286e-062, 1.61854750e-001, 8.21655047e-007, 1.51916212e-001,
            4.03350032e-114, 4.70434163e-147, 2.92957426e-002, 2.76163417e-037,
                        nan, 1.46740020e-003, 1.86773259e-001, 5.10040804e-001,
            1.01840890e-002, 1.62861531e-001,             nan, 1.48891650e-007,
            0.00000000e+000, 1.99913998e-038]))




```
SelectKBest(chi2, k='all').fit_transform(X_train.drop(num_cols, axis=1), y_train).shape
```




    (32950, 58)


