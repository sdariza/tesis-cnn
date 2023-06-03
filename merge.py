import cdo
import glob
cdo = cdo.Cdo()

nc_paths = glob.glob('data/*.nc')

cdo.mergetime(input=nc_paths, output='train_data/data_merged.nc')