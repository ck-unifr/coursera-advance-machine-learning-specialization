import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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

print('test data loading ...')
df_test = pd.read_csv('./data/X_test.csv')
X_test = df_test.copy()
X_test.drop(['ID'], axis=1, inplace=True)
print(X_test.columns)
print(X_test.shape)
print(type(X_test))
print(type(xgb.DMatrix(X_test)))


X_train_stack = X_train.copy()
X_validation_stack = X_validation.copy()
X_test_stack = X_test.copy()

print('xgb training ...')


n_trees = 3500

params1 = {
        'eta': 0.08,
        'max_depth': 7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 3,
        'gamma':1,
        'silent': True
    }

params2 = {
        'eta': 0.08,
        'max_depth': 8,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 4,
        'gamma':1,
        'silent': True
    }

params3 = {
        'eta': 0.08,
        'max_depth': 6,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 5,
        'gamma':1,
        'silent': True
    }

list_params = [params1, params2, params3]

watchlist = [
    (xgb.DMatrix(X_train, y_train), 'train'),
    (xgb.DMatrix (X_validation, y_validation), 'validation')
]
for i, params in enumerate(list_params):
    print('{}/{}'.format(i, len(list_params)))
    print(params)
    model = xgb.train(params, xgb.DMatrix(X_train, y_train), n_trees,  watchlist, maximize=False,
                      verbose_eval=50, early_stopping_rounds=50)

    X_train_stack['xgboost_item_cnt_month_'+str(i)] = model.predict(xgb.DMatrix(X_train), ntree_limit=model.best_ntree_limit)
    X_validation_stack['xgboost_item_cnt_month_'+str(i)] = model.predict(xgb.DMatrix(X_validation), ntree_limit=model.best_ntree_limit)
    X_test_stack['xgboost_item_cnt_month_'+str(i)] = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)


# X_train_stack.to_csv('./data/X_train_stack.csv', index=False)
# X_validation_stack.to_csv('./data/X_validation_stack.csv', index=False)
# X_test_stack.to_csv('./data/X_test_stack.csv', index=False)
#
# X_train_stack = pd.read_csv('./data/X_train_stack.csv')
# X_validation_stack = pd.read_csv('./data/X_validation_stack.csv')
# X_test_stack = pd.read_csv('./data/X_test_stack.csv')


# print('knn training ...')
#
# X_train_sample, _, y_train_sample, __ = train_test_split(X_train, y_train, train_size=.05, random_state=10)
#
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(X_train_sample)
#
# # list_k = [2, 3, 4]
# list_k = [2]
#
# for k in list_k:
#     print("train knn model "+str(k))
#     neigh = KNeighborsRegressor(n_neighbors=k, n_jobs=4, algorithm='kd_tree')
#     neigh.fit(scaler.transform(X_train_sample), y_train_sample)
#
#     print("using "+str(k)+" to predict")
#     X_train_stack[str(k)+'_neighbors'] = neigh.predict(scaler.transform(X_train))
#     X_validation_stack[str(k)+'_neighbors'] = neigh.predict(scaler.transform(X_validation))
#     X_test_stack[str(k)+'_neighbors'] = neigh.predict(scaler.transform(X_test))


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# X_train_transform = scaler.fit_transform(X_train)
# X_validation_transform = scaler.transform(X_validation)
# X_test_transform = scaler.transform(X_test)
#
# for kernel in 'poly', 'rbf', 'sigmoid':
#     clf = SVR(kernel=kernel, max_iter=500)
#     print("train the "+kernel+" SVR model")
#     clf.fit(X_train_transform, y_train)
#     print("Using the "+kernel+" model to predict")
#     X_train['svm_'+kernel] = clf.predict(X_train_transform)
#     X_validation['svm_'+kernel] = clf.predict(X_validation_transform)
#     X_test['svm_'+kernel] = clf.predict(X_test_transform)
#
#
# for df in X_train, X_validation, X_test:
#     df.drop(['svm_poly', 'svm_rbf', 'svm_sigmoid'], axis=1, inplace=True)

X_train = X_train_stack
X_validation = X_validation_stack
X_test = X_test_stack

print('ridge training ...')
n_ridge_iter = 1000

model = Ridge(alpha=1, copy_X=True, normalize=True, max_iter=n_ridge_iter)
model.fit(X_train, y_train)

print(mean_squared_error(y_validation, model.predict(X_validation)))

pred = model.predict(X_test)
# df_test['item_cnt_month'] = pred.clip(0, 40)
df_test['item_cnt_month'] = pred.clip(0, 20)
df_test[['ID', 'item_cnt_month']].to_csv('stacking_submission.csv', index=False)