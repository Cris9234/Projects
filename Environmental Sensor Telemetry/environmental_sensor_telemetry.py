import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn import metrics
import seaborn as sns


# %%
# PREPROCESSING

data = pd.read_csv('iot_telemetry_data.csv')
print(data.isnull().sum())
devices = data.device.unique()
data = data.replace(devices, ['device 1', 'device 2', 'device 3'])
data['time'] = pd.to_datetime(data['ts'], unit='s')
data.drop('ts', inplace=True, axis=1)
data['light'] = data['light'].astype('int')
data['motion'] = data['motion'].astype('int')
print(data.corr())


# %%

dev_1 = data[data.device == 'device 1']
dev_2 = data[data.device == 'device 2']
dev_3 = data[data.device == 'device 3']


# %%
# FUNCTIONS

def FindLocalMin(numbers):
    minima = []
    length = len(numbers)
    for n in range(1, length - 1):
        if numbers[n] <= numbers[n - 1] and numbers[n] <= numbers[n + 1]:
            minima.append(numbers[n])
        if numbers[length - 1] <= numbers[length - 2]:
            minima.append(numbers[length - 1])
    return min(minima)

def data_clean(x, y, x_label, y_label, graphs):
    MSE = []
    rn = range(1000, 2200, 100)
    for s in rn:
        EMA = y.ewm(span=s, adjust=False).mean()
        weight = abs(1 / (.1 + y - EMA))
        mse = round(sm.mean_squared_error(y, EMA, sample_weight=weight), 2)
        MSE.append(mse)

    loc_min = FindLocalMin(MSE)
    idx_first_loc_min = MSE.index(loc_min)
    best_ema = y.ewm(span=rn[idx_first_loc_min], adjust=False).mean()

    if graphs == True:
        plt.figure(figsize=(25, 10))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, color='lime', label='Data')
        plt.plot(x, best_ema, color='blue', linewidth=3, label='EMA')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right', prop={'size': 20})
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.plot(rn, MSE)
        plt.plot(rn[idx_first_loc_min], loc_min, marker='o', color='red')
        plt.xlabel('number of data')
        plt.ylabel('MSE')

    plt.subplots_adjust(wspace=.1)
    plt.show()

    return best_ema


def conf_mat(test, pred):
    print('Accuracy=', metrics.accuracy_score(test, pred))
    c_m = metrics.confusion_matrix(test, pred)
    sns.heatmap(c_m, annot=True, cmap='Blues_r',
                xticklabels=['absence', 'presence'],
                yticklabels=['absence', 'presence'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


# %%

# dc_1_co = data_clean(dev_1.time, dev_1.co, x_label='time', y_label='co', graphs=True)
dc_1_hum = data_clean(dev_1.time, dev_1.humidity, x_label='time', y_label='humidity [ppm]', graphs=True)
# dc_1_lpg = data_clean(dev_1.time, dev_1.lpg, x_label='time', y_label='lpg', graphs=True)
dc_1_smoke = data_clean(dev_1.time, dev_1.smoke, x_label='time', y_label='smoke [ppm]', graphs=True)
dc_1_temp = data_clean(dev_1.time, dev_1.temp, x_label='time', y_label='temp [°F]', graphs=True)


# %%

df = pd.DataFrame({'time': dev_1.time,
                   # 'co': dc_1_co,
                   'humidity': dc_1_hum,
                   # 'lpg': dc_1_lpg,
                   'smoke': dc_1_smoke,
                   'temp': dc_1_temp,
                   'light': dev_1.light,
                   'motion': dev_1.motion})

df['pres'] = np.zeros(len(df), dtype=int)
df = df.sample(frac=.05, random_state=154)
df = df.sort_values(by='time', ascending=True)
df.index = range(0, len(df))


# %%

name_col = df.columns[1:]

for i in name_col[:-1]:
    if df[i].dtype == 'float':
        for j in df.index:
            if j == 0 or df.pres[j] == 1:
                pass
            elif df[i][j - 1] < df[i][j] and df[i][j] - df[i][j - 1] > .0008:
                # .0008 is a random number but it can be replaced with the error
                # of each sensors error_device[i].
                df.loc[j, 'pres'] = 1
    else:
        for j in df.index:
            if df.pres[j] == 1:
                pass
            elif df[i][j] == 1:
                df.loc[j, 'pres'] = 1


#%%
# DATA SPLIT

X, y = df[name_col[:-1]], df[name_col[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)


# %%
# RANDOM FORES

oob_error = []
rn_est = list(range(30, 205, 5))
for i in rn_est:
    clf = RandomForestClassifier(max_features='sqrt', oob_score=True, n_estimators=i,
                                 random_state=123)
    clf.fit(X_train, y_train)
    oob_err = 1 - clf.oob_score_
    oob_error.append(oob_err)

best_n_est = rn_est[oob_error.index(min(oob_error))]


# %%

plt.plot(rn_est, oob_error)
plt.plot(best_n_est, min(oob_error), marker='o', color='red')
plt.xlabel('n estimators')
plt.ylabel('OOB error rate')
plt.show()


# %%

RF_classifier = RandomForestClassifier(max_features='sqrt', oob_score=True,
                                       n_estimators=best_n_est, random_state=123)

# %%

RF_classifier.fit(X_train, y_train)
y_RF_pred = RF_classifier.predict(X_test)
conf_mat(y_test, y_RF_pred)


# %%
# FEATURE IMPORTANCE

feature_importance = RF_classifier.feature_importances_
feature_names = name_col[:-1]
feature_importance = 100 * (feature_importance / max(feature_importance))

index_sorted = np.flipud(np.argsort(feature_importance))
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.bar(pos, feature_importance[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative importance')
plt.title('Feature Importance')
plt.show()


# %%
# APPLICATION OF THE CLASSIFIER ON THE SECOND DATASET

# dc_2_co = data_clean(dev_2.time, dev_2.co, x_label='time', y_label='co', graphs=False)
dc_2_hum = data_clean(dev_2.time, dev_2.humidity, x_label='time', y_label='humidity', graphs=False)
# dc_2_lpg = data_clean(dev_2.time, dev_2.lpg, x_label='time', y_label='lpg', graphs=False)
dc_2_smoke = data_clean(dev_2.time, dev_2.smoke, x_label='time', y_label='smoke', graphs=False)
dc_2_temp = data_clean(dev_2.time, dev_2.temp, x_label='time', y_label='temp', graphs=False)


# %%

df_2 = pd.DataFrame({'time': dev_2.time,
                     # 'co': dc_2_co,
                     'humidity': dc_2_hum,
                     # 'lpg': dc_2_lpg,
                     'smoke': dc_2_smoke,
                     'temp': dc_2_temp,
                     'light': dev_2.light,
                     'motion': dev_2.motion})

df_2['pres'] = np.zeros(len(df_2), dtype=int)
df_2 = df_2.sample(frac=0.5, random_state=154)
df_2 = df_2.sort_values(by='time', ascending=True)
df_2.index = range(0, len(df_2))


# %%

y_RF_pred_2 = RF_classifier.predict(df_2[name_col[:-1]])


# %%

plt.figure(figsize=(20, 10))
plt.plot(df_2.time, y_RF_pred_2, linewidth=3)
plt.xlabel('time')
plt.ylabel('presence')
plt.yticks([0, 1])
plt.ylim(0, 1.1)
plt.show()


# %%
# APPLICATION OF THE CLASSIFIER ON THE THIRD DATASET

# dc_3_co = data_clean(dev_3.time, dev_3.co, x_label='time', y_label='co', graphs=False)
dc_3_hum = data_clean(dev_3.time, dev_3.humidity, x_label='time', y_label='humidity', graphs=False)
# dc_3_lpg = data_clean(dev_3.time, dev_3.lpg, x_label='time', y_label='lpg', graphs=False)
dc_3_smoke = data_clean(dev_3.time, dev_3.smoke, x_label='time', y_label='smoke', graphs=False)
dc_3_temp = data_clean(dev_3.time, dev_3.temp, x_label='time', y_label='temp', graphs=False)


# %%

df_3 = pd.DataFrame({'time': dev_3.time,
                     # 'co': dc_3_co,
                     'humidity': dc_3_hum,
                     # 'lpg': dc_3_lpg,
                     'smoke': dc_3_smoke,
                     'temp': dc_3_temp,
                     'light': dev_3.light,
                     'motion': dev_3.motion})

df_3['pres'] = np.zeros(len(df_3), dtype=int)
df_3 = df_3.sample(frac=0.5, random_state=154)
df_3 = df_3.sort_values(by='time', ascending=True)
df_3.index = range(0, len(df_3))


# %%

y_RF_pred_3 = RF_classifier.predict(df_3[name_col[:-1]])


# %%

plt.figure(figsize=(20, 10))
plt.plot(df_3.time, y_RF_pred_3, linewidth=3)
plt.xlabel('time')
plt.ylabel('presence')
plt.yticks([0, 1])
plt.ylim(0, 1.1)
plt.show()

