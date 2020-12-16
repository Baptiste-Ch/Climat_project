# ---------- LIBRARIES ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------- IMPORTING & TRANSFORMING -------
france = pd.read_csv('france.csv')
france.DATE = pd.to_datetime(france.DATE)
france_global = france.groupby('DATE')['TAVG'].mean()


# ---------- MOVING AVERAGE -----------------
fig, ax = plt.subplots(figsize = (12, 5))
france_global.plot(y = 'TAVG',
                   use_index = True,
                   ax=ax)
# Because of seasonality we can't visually infer on the evolution of the temperature


# Moving average :
plt.figure(figsize = (20,10))
rolling_mean = france_global.rolling(window = 12).mean()
rolling_std = france_global.rolling(window = 12).std()
plt.plot(france_global, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.legend(loc = 'best')
plt.yticks(np.arange(0, 25, 1))
plt.title('Moyenne et Ecart-type mobiles')
plt.show()


# ---------- LINEAR REGRESSION --------------

# We keep the trend of our Time Series :
result_mul = seasonal_decompose(france_global,
                                model='multiplicative')
trend = result_mul.trend
trend = trend.dropna()

plt.figure(figsize = (12, 7))
plt.plot(trend)
plt.title('TAVG désaisonalisé', fontsize = 16)
plt.plot()

trend = trend.reset_index()
trend['days_since'] = (trend.DATE - pd.to_datetime('1975-01-01')).astype('timedelta64[M]')


# Building of the model :
results = smf.ols('trend ~ days_since', data = trend).fit()
print(results.summary())


# ---------- RESIDUALS TESTS ----------------

# Q-Q plot :
fig = plt.figure(figsize = (12, 7))
ax = fig.add_subplot()

sm.qqplot(results.resid, dist = stats.norm, line = 's', ax = ax)

ax.set_title("Q-Q Plot")
plt.show()

# The distribution of the residuals fit with the model distribution


# Plot of normal distribution : 
fig = plt.figure(figsize = (12, 7))

ax = sns.distplot(results.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot des Residus du Model (Bleu) et de la Distribution Normale (Noir)")
ax.set_xlabel("Residus")

# Residuals distribution close to normal 

   
# Homoscedasticity test :
fig = plt.figure(figsize = (12, 7))

ax = sns.scatterplot(y = results.resid, x = results.fittedvalues)

ax.set_title("RVF Plot")
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")

# There is a serious problem of homoscedasticity. There is a seasonal pattern


# ---------- ANNUALIZATION ------------------
france_annual = france_global.resample('Y', closed = 'left').mean()

plt.figure(figsize = (12, 7))
plt.plot(france_annual)
plt.title('TAVG désaisonalisé', fontsize = 16)
plt.plot()

france_annual = france_annual.reset_index()
france_annual['days_since'] = (france_annual.DATE - pd.to_datetime('1975-01-01')).astype('timedelta64[Y]')


# ---------- LINEAR REGRESSION --------------
results_annual = smf.ols('TAVG ~ days_since', data = france_annual).fit()
print(results_annual.summary())


# ---------- RESIDUALS TESTS ----------------

# Q-Q plot :
fig = plt.figure(figsize = (12, 7))
ax = fig.add_subplot()

sm.qqplot(results_annual.resid, dist = stats.norm, line = 's', ax = ax)

ax.set_title("Q-Q Plot")
plt.show()

# The distribution of the residuals still pretty fits with the model distribution


# Normal distribution :  
fig = plt.figure(figsize = (12, 7))

ax = sns.distplot(results_annual.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot des Residus du Model (Bleu) et de la Distribution Normale (Noir)")
ax.set_xlabel("Residus")

# The plot doesn't look really good, we must do a proper test :

# Shapiro test :   
labels = ["Statistic", "p-value"]

norm_res = stats.shapiro(results_annual.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)

# The distribution of the residuals is finally normal


# Homoscedasticity test :  
fig = plt.figure(figsize = (12, 7))

ax = sns.scatterplot(y = results_annual.resid, x = results_annual.fittedvalues)

ax.set_title("RVF Plot")
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")

# Homoscedasticity is validated