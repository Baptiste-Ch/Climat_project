# ----------- LIBRARIES ------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from itertools import cycle, islice
from sklearn import decomposition
from sklearn import preprocessing
from functions import *
from matplotlib import rcParams
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy.stats as stats


# ----------- IMPORTING ------------------
france = pd.read_csv('france.csv')


# ----------- TRANSFORMING DATA ----------
france.DATE = pd.to_datetime(france.DATE)


# We group variables by season :
france_season = france.groupby('NAME').resample('3M', closed = 'left', on = 'DATE') \
    .mean().reset_index()
season = cycle(['Winter', 'Spring', 'Summer', 'Autumn'])
france_season['Seasons'] = list(islice(season, len(france_season)))


# ----------- CHECKING VARIABLES ---------

# PCA :
fr_pca = france_season[['EMXP', 'PRCP', 'TAVG', 'TMAX', 'TMIN']]

X = fr_pca.values
names = fr_pca.index
features = fr_pca.columns

std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)


# We choose Principal Component = 2 :  
n_comp = 2

pca = decomposition.PCA(n_components = n_comp)
pca.fit(X_scaled)
display_scree_plot(pca)


# Correlation circle :
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1)], labels = np.array(features))
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, (0,1), labels = np.array(names))

plt.show()
plt.savefig('pca.png')

# The ACP shows :
    # TMIN, TAVG and TMAX are confounding
    # PRCP and EMXP are closely correlated


# ----------- LINEAR REGRESSION ----------
fig, ax = plt.subplots()
ax.hist(france_season.PRCP, bins=40)
plt.show()


# Building of the model :          
model = smf.ols(formula = 'np.sqrt(PRCP) ~ TAVG + EMXP + C(Seasons) + C(NAME)',
                data = france_season).fit()
print(model.summary())     


# ----------- RESIDUALS TESTS ------------
   
# Q-Q plot :
fig = plt.figure(figsize = (12, 9))
ax = fig.add_subplot()

sm.qqplot(model.resid, dist = stats.norm, line = 's', ax = ax)

ax.set_title("Q-Q Plot")
plt.show()
# The distribution of the residuals fit with the model distribution


# Plot of normal distribution :
fig = plt.figure(figsize = (12, 9))

ax = sns.distplot(model.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot des Residus du Model (Bleu) et de la Distribution Normale (Noir)")
ax.set_xlabel("Residus")
# Residuals distribution close to normal 

   
# Homoscedasticity test :
    
fig = plt.figure(figsize = (12, 9))

ax = sns.scatterplot(y = model.resid, x = model.fittedvalues)

ax.set_title("RVF Plot")
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
# Homoscedasticity is validated