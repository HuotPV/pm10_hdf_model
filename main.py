import numpy as np
import pandas as pd
import xarray as xr
import itertools
import glob
import scipy
import netCDF4
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor 

import matplotlib.pyplot as plt

class Pm10Data:
    id_file = itertools.count()

    def __init__(self, path):

        self.path = path
        self.data = pd.concat(
            map(pd.read_csv, glob.glob(path + '*.csv*')), ignore_index=True)

    def info(self):
        print(self.data.Station.unique())

class Era5Data:
    id_file = itertools.count()

    def __init__(self, path):

        self.path = path
        with xr.open_dataset(path) as ds:
            #data = ds.sel(expver=5)
            self.data = ds
    
    def extract_data_at_station(self,lon_st,lat_st):
        lon = np.copy(self.data.longitude)
        lat = np.copy(self.data.latitude)
        i_station = np.argmin(abs(lon-lon_st))
        j_station = np.argmin(abs(lat-lat_st))

        cropped_ds = self.data.sel(longitude=self.data.longitude[i_station], latitude=self.data.latitude[j_station])
        cropped_ds = cropped_ds.mean(dim='expver',skipna=True)
        cropped_ds_daily = cropped_ds.resample(time='D').mean()
        print(cropped_ds_daily.keys())
        return cropped_ds_daily


class Model:

    def __init__(self,data):
        n1 = round(data.shape[0] * 0.8) # used to delimit training and testing dataset

        u10_train = data["u10"].iloc[0:n1]; u10_test = data["u10"].iloc[n1::]
        v10_train = data["v10"].iloc[0:n1]; v10_test = data["v10"].iloc[n1::]
        t2m_train = data["t2m"].iloc[0:n1]; t2m_test = data["t2m"].iloc[n1::]
        d2m_train = data["d2m"].iloc[0:n1]; d2m_test = data["d2m"].iloc[n1::]
        blh_train = data["blh"].iloc[0:n1]; blh_test = data["blh"].iloc[n1::]
        mbld_train = data["mbld"].iloc[0:n1]; mbld_test = data["mbld"].iloc[n1::]
        tp_train = data["tp"].iloc[0:n1]; tp_test = data["tp"].iloc[n1::]
        sp_train = data["sp"].iloc[0:n1]; sp_test = data["sp"].iloc[n1::]
        tp_train = data["tp"].iloc[0:n1]; tp_test = data["tp"].iloc[n1::]
        pm10_train = data["Particules PM10  (µg.m-3)"].iloc[0:n1]; pm10_test = data["Particules PM10  (µg.m-3)"].iloc[n1::]
        date_train = data["Date"].iloc[0:n1]; date_test = data["Date"].iloc[n1::]

        self.X_train = np.concatenate([np.expand_dims(u10_train, axis=0),
                            np.expand_dims(v10_train, axis=0), 
                            np.expand_dims(sp_train, axis=0), 
                            np.expand_dims(tp_train, axis=0), 
                            np.expand_dims(t2m_train, axis=0), 
                            np.expand_dims(d2m_train, axis=0), 
                            np.expand_dims(blh_train, axis=0), 
                            np.expand_dims(mbld_train, axis=0), 
                            np.expand_dims(np.sqrt(u10_train*u10_train + v10_train*v10_train),axis=0)],axis=0)

        self.X_test = np.concatenate([np.expand_dims(u10_test, axis=0),
                            np.expand_dims(v10_test, axis=0), 
                            np.expand_dims(sp_test, axis=0), 
                            np.expand_dims(tp_test, axis=0), 
                            np.expand_dims(t2m_test, axis=0), 
                            np.expand_dims(d2m_test, axis=0), 
                            np.expand_dims(blh_test, axis=0), 
                            np.expand_dims(mbld_test, axis=0), 
                            np.expand_dims(np.sqrt(u10_test*u10_test + v10_test*v10_test),axis=0)],axis=0)

        self.pm10_train = pm10_train
        self.pm10_test = pm10_test
        self.date_train = date_train
        self.date_test = date_test


class RegModel(Model):

    def __init__(self,data):
        super().__init__(data)

    def predict(self):
        # Create linear regression object
        reg = linear_model.LinearRegression()
        reg.fit(np.moveaxis(self.X_train, 0, -1), self.pm10_train)


        predict_pm10 = reg.predict(np.moveaxis(self.X_test, 0, -1))

        baseline_rmse = np.sqrt(metrics.mean_squared_error(self.pm10_test*0 + np.nanmean(self.pm10_test), self.pm10_test))
        predict_rmse = np.sqrt(metrics.mean_squared_error(predict_pm10, self.pm10_test))

        print('RMSE climatological:', baseline_rmse)
        print('RMSE RegMod:', predict_rmse)

        stats = [baseline_rmse,predict_rmse]

        a = np.concatenate([np.expand_dims(self.date_test, axis=0),
                            np.expand_dims(self.pm10_test, axis=0),
                            np.expand_dims(predict_pm10, axis=0)],axis=0)
        return a,stats


class RandomForestModel(Model):

    def __init__(self,data):
        super().__init__(data)

    def predict(self):
        # Create Decision tree model
        reg = RandomForestRegressor(n_estimators = 10, # The number of trees in the forest
                            max_depth = 4, # the maximum depth any tree is allowed to have
                            )
        reg.fit(np.moveaxis(self.X_train, 0, -1), self.pm10_train)


        predict_pm10 = reg.predict(np.moveaxis(self.X_test, 0, -1))

        baseline_rmse = np.sqrt(metrics.mean_squared_error(self.pm10_test*0 + np.nanmean(self.pm10_test), self.pm10_test))
        predict_rmse = np.sqrt(metrics.mean_squared_error(predict_pm10, self.pm10_test))

        print('RMSE climatological:', baseline_rmse)
        print('RMSE RegMod:', predict_rmse)

        stats = [baseline_rmse,predict_rmse]

        a = np.concatenate([np.expand_dims(self.date_test, axis=0),
                            np.expand_dims(self.pm10_test, axis=0),
                            np.expand_dims(predict_pm10, axis=0)],axis=0)
        return a,stats
    

def __main__():
    # paths to data
    #path_pm10 = '/home/pv/Documents/pm10_hdf_data/'
    #path_station_info = '/home/pv/Documents/pm10_hdf_data/stations_infos.csv'
    #path_era5 = '/home/pv/Documents/pm10_hdf_data/era5-data.nc'

    path_pm10 = '/app/inputs/'
    path_station_info = '/app/inputs/stations_infos.csv'
    path_era5 = '/app/inputs/era5-data.nc'

    ### LOAD DATA ###
    # pm10, add station info
    station_info = pd.read_csv(path_station_info)
    pm10_dataset = Pm10Data(path_pm10)
    pm10_stations = station_info.merge(pm10_dataset.data,on='Station')
    pm10_dataset.data.dropna()

    # era5
    era5_dataset = Era5Data(path_era5)
    print(era5_dataset.data.keys())

    # get era5 data for each station
    era5_stations = pd.DataFrame()
    for i_station in range(station_info.shape[0]):
        era5_subset = era5_dataset.extract_data_at_station(station_info.Longitude[i_station],station_info.Latitude[i_station]).to_dataframe()
        era5_subset['Station'] = station_info.Station[i_station]
        era5_stations = pd.concat((era5_stations,era5_subset))

    # add a Date column for merging purpose
    era5_stations["Date"]=era5_stations.index

    # merge pm10 and era5 data for each station
    era5_pm10_stations=pd.DataFrame()

    for i_station in range(station_info.shape[0]):
        era5_onestation = era5_stations.loc[era5_stations['Station']==station_info.Station[i_station]]
        pm10_onestation = pm10_stations.loc[pm10_stations['Station']==station_info.Station[i_station]]
        era5_onestation['Date'] = pd.to_datetime(era5_onestation['Date'], utc = True)
        pm10_onestation['Date'] = pd.to_datetime(pm10_onestation['Date'], utc = True)
        df_merged = era5_onestation.merge(pm10_onestation,on='Date')
        era5_pm10_stations = pd.concat((era5_pm10_stations,df_merged))

    print(era5_pm10_stations)
    print(era5_pm10_stations.shape)
    era5_pm10_stations.dropna(inplace=True)

    # build the linear regression model
    stats = np.zeros((station_info.shape[0],3))

    for i_station in range(station_info.shape[0]):
        print(station_info.Station[i_station])
        mymodel_reg = RegModel(era5_pm10_stations.loc[era5_pm10_stations['Station_x']==station_info.Station[i_station]])
        mymodel_rnd = RandomForestModel(era5_pm10_stations.loc[era5_pm10_stations['Station_x']==station_info.Station[i_station]])
        results_reg,stats_reg = mymodel_reg.predict()
        results_rnd,stats_rnd = mymodel_rnd.predict()

        a = [stats_reg[0],stats_reg[1],stats_rnd[1]]
        stats[i_station,:] = a

        plt.plot(results_reg[0,:],results_reg[2,:],color=[0.1,0.4,0.8],label="regression")
        plt.plot(results_rnd[0,:],results_rnd[2,:],color=[0.6,0.2,0.7],label="random forest")
        plt.plot(results_reg[0,:],results_reg[1,:],'k',label="observed")
        plt.ylabel('Particules PM10  (µg.m-3)')
        plt.legend()
        plt.title(station_info.Station[i_station])
        plt.savefig('/app/inputs/results_{:02d}.png'.format(i_station),dpi=300)
        plt.close()
        
    print(stats)
    plt.plot(stats,label=["baseline","regression","random forest"])
    plt.ylabel('PM10 RMSE (µg.m-3)')
    plt.legend()
    plt.title(station_info.Station[i_station])
    plt.savefig('/app/inputs/stats.png'.format(i_station),dpi=300)
    plt.close()
    
__main__()
