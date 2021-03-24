import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn import metrics
import seaborn as sns


# %%
data = pd.read_csv('iot_telemetry_data.csv')
print(data.isnull().sum())
devices = data.device.unique()
data = data.replace(devices, ['device 1', 'device 2', 'device 3'])
data['time'] = pd.to_datetime(data['ts'], unit='s')
data.drop('ts', inplace=True, axis=1)
data['light'] = data['light'].astype('int32')
data['motion'] = data['motion'].astype('int32')
print(data.corr())

dat_v2 = data.sample(frac=.05, random_state=1)
dat_v2 = dat_v2.sort_values(by='time', ascending=True)


# %%
dev_1 = dat_v2[dat_v2.device == 'device 1']
dev_2 = dat_v2[dat_v2.device == 'device 2']
dev_3 = dat_v2[dat_v2.device == 'device 3']

# %%
plt.figure(figsize=(20, 10))
plt.plot(dev_1.time, dev_1.co, color='blue')
plt.plot(dev_1.time, dev_1.smoke, color='red')
plt.plot(dev_1.time, dev_1.lpg, color='green')
plt.xlabel('time')
plt.ylabel('CO, smoke, lpg')
plt.show()


# %%


def FindLocalMin(numbers):
    minima = []
    length = len(numbers)
    for n in range(1, length - 1):
        if numbers[n] <= numbers[n - 1] and numbers[n] <= numbers[n + 1]:
            minima.append(numbers[n])
        if numbers[length - 1] <= numbers[length - 2]:
            minima.append(numbers[length - 1])
    return min(minima)

#%%


def data_clean(x, y, x_label, y_label):
    MSE = []
    rn = range(400, 2200, 100)
    for s in rn:
        EMA = y.ewm(span=s, adjust=False).mean()
        weight = abs(1 / (.1 + y - EMA))
        # 0.1 is a small constant offset used to avoid division by zero
        mse = round(sm.mean_squared_error(y, EMA, sample_weight=weight), 2)
        MSE.append(mse)

    loc_min = FindLocalMin(MSE)
    idx_first_loc_min = np.array(np.where(MSE == loc_min)).min()

    best_ema = y.ewm(span=rn[idx_first_loc_min], adjust=False).mean()

    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, color='blue')
    plt.plot(x, best_ema, color='green', linewidth=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.subplot(1, 2, 2)
    plt.plot(rn, MSE)
    plt.plot(rn[idx_first_loc_min], loc_min, marker='o', color='red')
    plt.xlabel('number of data')
    plt.ylabel('MSE')

    plt.show()

    return best_ema

#%%


def D(x, y):
    y_prime = np.diff(y) / np.diff(x)
    return y_prime


#%%


# dc_1_co = data_clean(dev_1.time, dev_1.co, x_label='time', y_label='co')
dc_1_hum = data_clean(dev_1.time, dev_1.humidity, x_label='time', y_label='humidity')
# dc_1_lpg = data_clean(dev_1.time, dev_1.lpg, x_label='time', y_label='lpg')
dc_1_smoke = data_clean(dev_1.time, dev_1.smoke, x_label='time', y_label='smoke')
dc_1_temp = data_clean(dev_1.time, dev_1.temp, x_label='time', y_label='temp')

#%%
df = pd.DataFrame({'time': dev_1.time,
                   # 'co': dc_1_co,
                   'humidity': dc_1_hum,
                   # 'lpg': dc_1_lpg,
                   'smoke': dc_1_smoke,
                   'temp': dc_1_temp,
                   'light': dev_1.light,
                   'motion': dev_1.motion})
df.index = range(0, len(df))
df['pres'] = np.zeros(len(df), dtype=int)

# %%

name_col = df.columns[1:]

for i in name_col[:-1]:
    if df[i].dtype == 'float':
        for j in df.index:
            if j == 0 or df.pres[j] == 1:
                pass
            elif df[i][j - 1] < df[i][j]:
                df.loc[j, 'pres'] = 1
    else:
        for j in df.index:
            if df.pres[j] == 1:
                pass
            elif df[i][j] == 1:
                df.loc[j, 'pres'] = 1

# %%
X, y = df[name_col[:-1]], df[name_col[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=543)

#%%

RF_classifier = RandomForestClassifier()
RF_classifier.fit(X_train, y_train)
y_RF_pred = RF_classifier.predict(X_test)

# %%
print('Accuracy=', metrics.accuracy_score(y_test, y_RF_pred))
RF_conf_mat = metrics.confusion_matrix(y_test, y_RF_pred)
sns.heatmap(RF_conf_mat, annot=True, cmap='Blues_r')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# %%

feature_importance = RF_classifier.feature_importances_
feature_names = name_col[:-1]
feature_importance = 100 * (feature_importance / max(feature_importance))

# %%
index_sorted = np.flipud(np.argsort(feature_importance))
pos = np.arange(index_sorted.shape[0]) + 0.5

# %%

plt.bar(pos, feature_importance[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative importance')
plt.title('Feature Importance')
plt.show()

