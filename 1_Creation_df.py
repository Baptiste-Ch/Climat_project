# ------------ LOAD LIBRARIES ---------------
import pandas as pd


# ------------ IMPORTATION ------------------
dt = pd.read_csv('data_france.csv')


# ------------ DATA CHECKING ----------------
dt.head()
dt.info()
dt.dtypes


# Here, DATE column has not the correct format. This should be on DateTime :
dt['DATE'] = pd.to_datetime(dt['DATE'])
print(dt.dtypes)


# Three columns are useless for the study :
dt = dt.drop(['STATION', 'LATITUDE', 'LONGITUDE'], axis=1)


# ------------ DEALING WITH NA's ------------
dt = dt.set_index('NAME')
print(dt.isna().groupby('NAME').sum())


# Two cities contain too much NA's, we can't work with them :
france_inter = dt.drop(['ANTICHAN', 'BARBEREY'], axis= 0)


# Same for 2020 rows :
france = france_inter[france_inter['DATE'] < '2020']


# We substitute the remaining NA's by their mean column :
avg = france.mean()
france.fillna(avg, inplace = True)
print(france.mean())


# ------------ CONVERTING TO °C -------------
column = ['TAVG', 'TMIN', 'TMAX']

for col in column: 
    france[col] = (france[col] - 32) * 5/9


# ------------ CHECKING WRONG VALUES --------
print(france[france['TMIN'] > france['TAVG']])
print(france[france['TAVG'] > france['TMAX']])
print(france[france['EMXP'] > france['PRCP']])
   
# No suspect values appears


# ------------ SAVE DATA --------------------
france.to_csv('france.csv', index = True)
