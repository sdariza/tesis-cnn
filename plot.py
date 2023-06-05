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

pred = model.predict(tf.reshape(X[10],[1,32,64]))[0]
pred = tf.reshape(pred, [32,64]).numpy()
print(pred.shape)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})
contour = ax.contourf(lons, lats, Y[10], cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(contour, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
gl = ax.gridlines(linewidth=.7, draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180,-90,0,90,180])
ax.coastlines()
plt.savefig('actual.png')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})
contour = ax.contourf(lons, lats, pred, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(contour, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
gl = ax.gridlines(linewidth=.7, draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180,-90,0,90,180])
ax.coastlines()
plt.savefig('predicted.png')
