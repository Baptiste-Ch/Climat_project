# ----------- LOAD LIBRARIES -----------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------- IMPORTATION --------------
france = pd.read_csv('france.csv')
france['DATE'] = pd.to_datetime(france['DATE'])


# ----------- PAIRPLOT -----------------
a = sns.pairplot(france,
             x_vars = ['EMXP', 'PRCP', 'TAVG', 'TMAX', 'TMIN'],
             y_vars = ['EMXP', 'PRCP', 'TAVG', 'TMAX', 'TMIN'],
             hue="NAME",
             height = 2)
a.fig.suptitle("Pair Grid of the whole \nvariables of the dataset \nfor each city",
               y = 0.95,
               x = 0.805,
               size = 15,
               weight = 'bold',
               ha = 'left')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.savefig('PairG_Fr.png')

# Some inferences can be done :
    # Temperature variables are linearly correlated
    # The more the Temperature increases, the more the variation of PRCP and EMXP increases
    # PRCP variation is more important in the North than South


# ----- PLOT TEMPERATURE BY STATION -----

# Data creation part :
france = france.set_index('DATE').groupby('NAME').resample('Y', closed = 'left') \
    ['ELEVATION','EMXP', 'PRCP', 'TAVG', 'TMAX', 'TMIN'].bfill().reset_index()


tb_tmin = france.iloc[:,[0, 1, 2, 3, 4, 7]].rename(columns={"TMIN" : "TEMP"})
tb_tmax = france.iloc[:,[0, 1, 2, 3, 4, 6]].rename(columns={"TMAX" : "TEMP"})
tb_tavg = france.iloc[:,[0, 1, 2, 3, 4, 5]].rename(columns={"TAVG" : "TEMP"})


tb_tmin['TMIN'] = ('TMIN')
tb_tmax['TMAX'] = ('TMAX')
tb_tavg['TAVG'] = ('TAVG')

tb_new = pd.concat([tb_tmin, tb_tavg, tb_tmax], ignore_index=False).fillna('')
tb_new['TYPE']=tb_new.apply(lambda x:'%s%s%s' % (x['TMIN'],x['TMAX'],x['TAVG']),axis=1)


end_tb = tb_new.iloc[:,[0, 1, 2, 3, 4, 6, 9]]
end_tb[["ELEVATION",
        "EMXP",
        "PRCP",
        "TEMP"]] = end_tb[["ELEVATION", "EMXP", 'PRCP', 'TEMP']].apply(pd.to_numeric)



# Plotting part : 
sns.set_style('darkgrid')
fig, axes = plt.subplots(2, 3, figsize = (15, 10), sharey = True)

sns.lineplot(ax = axes[0, 0],
             x='DATE', 
             y='TEMP', 
             data = end_tb[end_tb['NAME'] == 'BREST GUIPAVAS'])
sns.lineplot(ax = axes[0, 1], 
             x='DATE', 
             y='TEMP', 
             data = end_tb[end_tb['NAME'] == 'CAEN CARPIQUET'])
sns.lineplot(ax= axes[0, 2],
             x='DATE', 
             y='TEMP', 
             data = end_tb[end_tb['NAME'] == 'LYON ST EXUPERY'])
sns.lineplot(ax = axes[1, 0],
             x='DATE', 
             
             y='TEMP', 
             data = end_tb[end_tb['NAME'] == 'MARSEILLES MARIGNANE'])
sns.lineplot(ax = axes[1, 1],
             x='DATE', 
             y='TEMP', 
             data = end_tb[end_tb['NAME'] == 'ST GIRONS'])

axes[0,0].set(xlabel = 'Time (in years)', ylabel = 'Temperature Max, Mean and Min (°C)') 
axes[0,0].set_title('Brest')
axes[0,1].set(xlabel = 'Time (in years)', ylabel = 'Temperature Max, Mean and Min (°C)') 
axes[0,1].set_title('Caen')
axes[0,2].set(xlabel = 'Time (in years)', ylabel = 'Temperature Max, Mean and Min (°C)') 
axes[0,2].set_title('Lyon')
axes[1,0].set(xlabel = 'Time (in years)', ylabel = 'Temperature Max, Mean and Min (°C)') 
axes[1,0].set_title('Marseilles')
axes[1,1].set(xlabel = 'Time (in years)', ylabel = 'Temperature Max, Mean and Min (°C)') 
axes[1,1].set_title('St Girons')
axes[1,2].set_visible(False)

fig.suptitle('Evolution of temperature since 1975 \nin 5 cities of France',
             weight = 'bold',
             size = 20)
plt.savefig('evol_city_fr.png')