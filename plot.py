import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

lats = np.linspace(90,-90,32)
lons = np.linspace(0,357.5,64)


X  = np.load('train_data/air.2018-2023_H0.npy')
Y = np.load('train_data/air.2018-2023_H6.npy')

sample_points = [('longitude', lons), ('latitude',lats)]

model = tf.keras.models.load_model('gfgModel')

pred = model.predict(tf.reshape(X[0],[1,32,64]))[0]
pred = tf.reshape(pred, [32,64]).numpy()
print(pred.shape)

ax = plt.subplot(111, projection = ccrs.Robinson())
Z = ax.contourf(lons, lats, Y[0],cmap='viridis' ,transform=ccrs.PlateCarree())
gl = ax.gridlines(linewidth=.7, draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180,-90,0,90,180])
ax.coastlines()
plt.savefig('actual.png')

ax = plt.subplot(111, projection = ccrs.Robinson())
Z = ax.contourf(lons, lats, pred,cmap='viridis' ,transform=ccrs.PlateCarree())
gl = ax.gridlines(linewidth=.7, draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180,-90,0,90,180])
ax.coastlines()
plt.savefig('predicted.png')

exit()