import pandas as pd
import xgboost as xgb

print('train data loading ...')
X_train = pd.read_csv('./data/X_train.csv')
print(X_train.columns)
print(X_train.shape)
print(type(X_train))

y_train = pd.read_csv('./data/y_train.csv', header=None)
print(y_train.shape)

print('validation data loading ...')
X_validation = pd.read_csv('./data/X_validation.csv')
y_validation = pd.read_csv('./data/y_validation.csv', header=None)
print(X_validation.columns)
print(X_validation.shape)
print(y_validation.shape)



print('xgb training ...')

params = {
        'eta': 0.08,
        'max_depth': 7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 3,
        'gamma':1,
        'silent': True
    }

watchlist = [
    (xgb.DMatrix(X_train, y_train), 'train'),
    (xgb.DMatrix(X_validation, y_validation), 'validation')
]

n_trees = 1

model = xgb.train(params, xgb.DMatrix(X_train, y_train), n_trees,  watchlist, maximize=False,
                  verbose_eval=5, early_stopping_rounds=50)



print('test data loading ...')
df_test = pd.read_csv('./data/X_test.csv')
X_test = df_test.copy()
X_test.drop(['ID'], axis=1, inplace=True)
print(X_test.columns)
print(X_test.shape)
print(type(X_test))
print(type(xgb.DMatrix(X_test)))

print('xgb predicting ...')
pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

print('prediction saving ...')
df_test['item_cnt_month'] = pred.clip(0, 40)
df_test[['ID', 'item_cnt_month']].to_csv('xgboost_submission.csv', index=False)

print('save prediction to {}'.format('xgboost_submission.csv'))