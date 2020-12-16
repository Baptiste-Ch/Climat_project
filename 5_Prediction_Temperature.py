# ---------- LIBRARIES ----------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm


# ---------- IMPORTING & TRANSFORMING -------
france = pd.read_csv('france.csv')
france.DATE = pd.to_datetime(france.DATE)

france_global = france.groupby('DATE')['TAVG'].mean()


# ---------- STATIONARITY ------------------- 

# By Moving Average :
plt.figure(figsize = (10,7))
rolling_mean = france_global.rolling(window = 12).mean()
rolling_std = france_global.rolling(window = 12).std()
plt.plot(france_global, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show()


# Or with ADF test :
result = adfuller(france_global)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# The critical value is beyond 5% so we can assume at 95% that our 
    # time series is stationnary
        

# ---------- ACF/PACF -----------------------
data_df = france_global.diff(12).dropna()
 

# ACF :
plot_acf(data_df.diff().dropna(), lags=60, alpha = 0.01)


# PACF :
plot_pacf(data_df.diff().dropna(), lags=60, alpha = 0.01)

# From these plot we can say :
    # p = 1
    # d = 0
    # q = 1
    # P = 1 
    # D = 1
    # Q = 1
    
# ---------- SARIMA -------------------------   
mod = sm.tsa.statespace.SARIMAX(france_global,
                                order = (1, 0, 1),
                                trend = 'n',
                                seasonal_order = (1, 1, 1, 12),
                                enforce_stationarity = False,
                                enforce_invertibility = False).fit()
print(mod.summary())


# Distribution to the error :
mod.plot_diagnostics(figsize = (13, 9))

# From top to bottom, left to right :
    # Residuals variation looks random
    # Residuals follows normal distribution
    # They fit with the model distribution
    # No Autocorrelated residuals so no pattern unexplained


# ---------- FORECASTING --------------------
    # 36 months :
sns.set_style('darkgrid')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(20,10))
forecast_values = mod.get_forecast(steps=36)
forecast_ci = forecast_values.conf_int()

ax = france_global.plot()
forecast_values.predicted_mean.plot(ax=ax, label = 'Forecasts')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:,0],
                forecast_ci.iloc[:,1], color = 'g', alpha = 0.5)
ax.set_xlim('1975-01-01', '2025-01-01')
ax.set_xlabel('Time')
ax.set_ylabel('TAVG')
ax.set_title('Prévision de la température moyenne en France \n à partir du test SARIMA')

plt.legend()
plt.savefig('forecast_france.png')