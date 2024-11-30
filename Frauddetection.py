#working machine learning code
from lightgbm import LGBMClassifier as lgb
import pandas as pd
def load_data(path):
    return pd.read_csv(path,low_memory = False)
client_train = load_data('train/client_train.csv')
client_train.head()
invoice_train = load_data('train/invoice_train.csv')
invoice_train.head(5)
client_test = load_data('test/client_test.csv')
client_test.head()
invoice_test = load_data('test/invoice_test.csv')
invoice_test.head()
print(client_train.shape,invoice_train.shape,client_test.shape,invoice_test.shape)
client_train.info()
client_train.describe().T
invoice_train.describe().T
def client_cleaning(df):

    df.rename(columns = {'disrict':'district'},inplace =True)
    df['region'] = df['region'].astype('object')
    df['district'] = df['district'].astype('object')
    df['client_catg'] = df['client_catg'].astype('object')
    df['creation_date'] = pd.to_datetime(df['creation_date'],
                                               errors='coerce',
                                               infer_datetime_format=True).dt.date
    for col in df.columns:
        print(f"Number of unique values in {col} - {df[col].nunique()}")
    return df
client_cleaning(client_train)
import datetime as dt
today_date = dt.date.today()
client_train[client_train['creation_date'] > dt.date.today()]
client_train.isna().sum()
client_train.duplicated().sum()
train_duplicates = client_train.duplicated(subset = 'client_id', keep = False)
client_train[train_duplicates]
client_cleaning(client_test)
client_test[client_test['creation_date'] > dt.date.today()]
client_test.isna().sum()
client_test.duplicated().sum()
test_duplicates = client_train.duplicated(subset = 'client_id', keep = False)
client_train[test_duplicates]
def invoice_cleaning(df):
    df['invoice_date'] = pd.to_datetime(df['invoice_date'],errors ='coerce').dt.date
    for col in df.columns:
        print(f"Number of unique values in {col} - {df[col].nunique()}")
    
    return df
invoice_cleaning(invoice_train)
invoice_train_duplicates =invoice_train.duplicated()
invoice_train[invoice_train_duplicates]
invoice_train.drop_duplicates(inplace=True)
invoice_train.duplicated().sum()
values = {'ELEC':0,'GAZ':1}
invoice_train['counter_type'] =invoice_train['counter_type'].map(values)
invoice_train['counter_type'].value_counts()
aggs = {}
aggs['consommation_level_1'] = ['mean']
aggs['consommation_level_2'] = ['mean']
aggs['consommation_level_3'] = ['mean']
aggs['consommation_level_4'] = ['mean']
agg_trans = invoice_train.groupby(['client_id']).agg(aggs)
agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
agg_trans.reset_index(inplace=True)
df = (invoice_train.groupby('client_id')
      .size()
      .reset_index(name='transactions_count'))
agg_trans = pd.merge(df, agg_trans, on='client_id', how='left')
agg_trans.head()
train = pd.merge(client_train,agg_trans, on='client_id', how='left')
train.head()
invoice_cleaning(invoice_test)
invoice_test_duplicates =invoice_test.duplicated(keep =False)
invoice_test[invoice_test_duplicates]
invoice_test.drop_duplicates(inplace=True)
invoice_test.duplicated().sum()
invoice_test['counter_type'] =invoice_test['counter_type'].map(values)
invoice_test['counter_type'].value_counts()
agg = {}
agg['consommation_level_1'] = ['mean']
agg['consommation_level_2'] = ['mean']
agg['consommation_level_3'] = ['mean']
agg['consommation_level_4'] = ['mean']
agg_test = invoice_test.groupby(['client_id']).agg(agg)
agg_test.columns = ['_'.join(col).strip() for col in agg_test.columns.values]
agg_test.reset_index(inplace=True)
df_test = (invoice_test.groupby('client_id')
      .size()
      .reset_index(name='{}transactions_count'.format('1')))
agg_test = pd.merge(df_test, agg_test, on='client_id', how='left')
agg_test.head()
test = pd.merge(client_test,agg_test, on='client_id', how='left')
test.head()
test.info()
train.info()
train['target'].value_counts()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
train.drop(['creation_date','client_id'],axis=1,inplace=True)
test.drop(['creation_date','client_id'],axis=1,inplace=True)
X = train.drop(["target"],axis = 1)
y = train.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state =42)
categorical_cols = ['district', 'client_catg', 'region']
x_train_cat = pd.get_dummies(X_train, columns=categorical_cols)
x_test_cat = pd.get_dummies(X_test, columns=categorical_cols)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_cat)
x_test_scaled = scaler.transform(x_test_cat)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
lr = LogisticRegression(random_state = 42,
                            multi_class ="ovr")
lr.fit(X_train_resampled, y_train_resampled)
lr_train_pred = lr.predict(X_train_resampled)
lr_test_pred = lr.predict(x_test_scaled)
lr_train_auc = roc_auc_score(y_train_resampled,lr_train_pred)
lr_test_auc = roc_auc_score(y_test,lr_test_pred)
print(f'baseline logreg Train AUC - {lr_train_auc} \nbaseline logreg Test AUC - {lr_test_auc}')
best_logreg =LogisticRegression(random_state = 42,multi_class ="ovr",tol =  0.0001,
                                 C = 1,solver = "newton-cg",penalty='l2')
best_logreg.fit(X_train_resampled,y_train_resampled)
best_logreg_train_pred = best_logreg.predict(X_train_resampled)
best_logreg_test_pred = best_logreg.predict(x_test_scaled)
best_logreg_train_auc = roc_auc_score(y_train_resampled,best_logreg_train_pred)
best_logreg_test_auc = roc_auc_score(y_test,best_logreg_test_pred)
print(f'Best logreg Train AUC - {best_logreg_train_auc} \nBest logreg Test AUC - {best_logreg_test_auc}')
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train_resampled,y_train_resampled)
dt_train_pred = dt.predict(X_train_resampled)
dt_test_pred = dt.predict(x_test_scaled)
dt_train_auc = roc_auc_score(y_train_resampled,dt_train_pred)
dt_test_auc = roc_auc_score(y_test,dt_test_pred)
print(f'Train AUC - {dt_train_auc} \nTest AUC - {dt_test_auc}')
best_dt = DecisionTreeClassifier(max_depth=350,max_features=4,
                                 min_samples_leaf=500,min_samples_split=4,criterion='entropy')
best_dt.fit(X_train_resampled,y_train_resampled)
best_dt_train_pred = best_dt.predict(X_train_resampled)
best_dt_test_pred = best_dt.predict(x_test_scaled)
best_dt_train_auc = roc_auc_score(y_train_resampled,best_dt_train_pred)
best_dt_test_auc = roc_auc_score(y_test,best_dt_test_pred)
print(f'best dt train AUC - {best_dt_train_auc} \nTest AUC - {best_dt_test_auc}')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
rf_train_pred = rf.predict(X_train_resampled)
rf_test_pred = rf.predict(x_test_scaled)
rf_train_auc = roc_auc_score(y_train_resampled,rf_train_pred)
rf_test_auc = roc_auc_score(y_test,rf_test_pred)
print(f'Train AUC - {rf_train_auc} \nTest AUC - {rf_test_auc}')
best_rf = RandomForestClassifier(random_state = 42,max_depth=None,
                                 max_features='auto',n_estimators=200)
best_rf.fit(X_train_resampled,y_train_resampled)
best_rf_train_pred = best_rf.predict(X_train_resampled)
best_rf_test_pred = best_rf.predict(x_test_scaled)
best_rf_train_auc = roc_auc_score(y_train_resampled,best_rf_train_pred)
best_rf_test_auc = roc_auc_score(y_test,best_rf_test_pred)
print(f'Train AUC - {best_rf_train_auc} \nTest AUC - {best_rf_test_auc}')
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42).fit(X_train_resampled,y_train_resampled)
gb_train_pred = gb.predict(X_train_resampled)
gb_test_pred = gb.predict(x_test_scaled)
gb_train_auc = roc_auc_score(y_train_resampled,gb_train_pred)
gb_test_auc = roc_auc_score(y_test,gb_test_pred)
print(f'Train AUC - {gb_train_auc} \nTest AUC - {gb_test_auc}')
best_gb_clf = GradientBoostingClassifier(random_state=42,n_estimators = 800,
                                         learning_rate=0.007,
                                         max_depth=1,
                                         max_features=4).fit(X_train_resampled,
                                                                  y_train_resampled)
best_gb_train_pred = best_gb_clf.predict(X_train_resampled)
best_gb_test_pred = best_gb_clf.predict(x_test_scaled)
best_gb_train_auc = roc_auc_score(y_train_resampled,best_gb_train_pred)
best_gb_test_auc = roc_auc_score(y_test,best_gb_test_pred)
print(f'Train AUC - {best_gb_train_auc} \nTest AUC - {best_gb_test_auc}')
from lightgbm import LGBMClassifier 
lgb = LGBMClassifier(random_state=42).fit(X_train_resampled,y_train_resampled)
lgb_train_pred = lgb.predict(X_train_resampled)
lgb_test_pred = lgb.predict(x_test_scaled)
lgb_train_auc = roc_auc_score(y_train_resampled,lgb_train_pred)
lgb_test_auc = roc_auc_score(y_test,lgb_test_pred)
print(f'Train AUC - {lgb_train_auc} \nTest AUC - {lgb_test_auc}')
best_lgb = LGBMClassifier(random_state=42,n_estimators=830,
                     num_leaves=454, max_depth=61,
                     learning_rate=0.006,
                     min_split_gain=0.006,
                     bagging_freq=8).fit(X_train_resampled,y_train_resampled)
best_lgb_train_pred = best_lgb.predict(X_train_resampled)
best_lgb_test_pred = best_lgb.predict(x_test_scaled)
best_lgb_train_auc = roc_auc_score(y_train_resampled,best_lgb_train_pred)
best_lgb_test_auc = roc_auc_score(y_test,best_lgb_test_pred)
print(f'Train AUC - {best_lgb_train_auc} \nTest AUC - {best_lgb_test_auc}')
models = {'Model': ['Baseline Logistic Regression', 'Tuned Logistic Regression', 
                    'Decision Tree', 'Tuned Decision Tree', 'Random Forest',
                    'Tuned Random Forest','Gradient Boosting Classifier',
                    'Tuned Gradient Boosting Classifier','LGBMClassifier',
                    'Tuned LGBMClassifier'],
          'Train AUC': [lr_train_auc, best_logreg_train_auc, dt_train_auc, 
                        best_dt_train_auc, rf_train_auc,best_rf_train_auc,
                        gb_train_auc,best_gb_train_auc,lgb_train_auc,best_lgb_train_auc],
          'Test AUC': [lr_test_auc, best_logreg_test_auc, dt_test_auc, 
                       best_dt_test_auc, rf_test_auc,best_rf_test_auc,
                       gb_test_auc,best_gb_test_auc,lgb_test_auc,best_lgb_test_auc]}
auc_df = pd.DataFrame(models)
auc_df = auc_df.set_index('Model')
auc_df['AUC Difference'] = auc_df['Train AUC'] - auc_df['Test AUC']
auc_df = auc_df.sort_values('AUC Difference', ascending=True)
auc_df
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(best_logreg, f)

