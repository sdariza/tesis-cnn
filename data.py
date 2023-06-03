import iris
import numpy as np

cube = iris.load_cube('train_data/data_merged.nc')

query_H0 = iris.Constraint(time=lambda cell: cell.point.hour == 0, air_pressure = lambda cell: cell==1000)
cube_H0 = cube.extract(query_H0)


query_H6 = iris.Constraint(time=lambda cell: cell.point.hour == 6, air_pressure = lambda cell: cell==1000)
cube_H6 = cube.extract(query_H6)

lats = np.linspace(90,-90,32)
lons = np.linspace(0,357.5,64)
sample_points = [('longitude', lons), ('latitude',lats)]

cube_H0 = cube_H0.interpolate(sample_points, iris.analysis.Linear()).data.data
cube_H6 = cube_H6.interpolate(sample_points, iris.analysis.Linear()).data.data

np.save('train_data/air.2018-2023_H0.npy', cube_H0)
np.save('train_data/air.2018-2023_H6.npy', cube_H6)

print('All data stored!')