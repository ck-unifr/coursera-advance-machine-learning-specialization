import pandas as pd
import lightgbm as lgb


print('train data loading ...')

X_train = pd.read_csv('./data/X_train.csv')
print(X_train.columns)
print(X_train.shape)

y_train = pd.read_csv('./data/y_train.csv', header=None)
print(y_train.shape)

print('validation data loading ...')
X_validation = pd.read_csv('./data/X_validation.csv')
y_validation = pd.read_csv('./data/y_validation.csv', header=None)
print(X_validation.columns)
print(X_validation.shape)
print(y_validation.shape)


print('lightGBM training...')

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

# https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'reg:linear',
    'metric': {'l2', 'rmse'},
    # 'num_leaves': 31,
    'max_depth': 7,
    'learning_rate': 0.08,
    # 'feature_fraction': 0.9,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'verbose': 0,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)

# print('Save model...')
# save model to file
# gbm.save_model('lightgbm_model.txt')


print('test data loading ...')

df_test = pd.read_csv('./data/X_test.csv')
X_test = df_test.columns[1:]

print('lightGBM predicting ...')

pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

df_test['item_cnt_month'] = pred.clip(0, 40)
df_test[['ID', 'item_cnt_month']].to_csv('lightgbm_submission.csv', index=False)
print('Start training...')



