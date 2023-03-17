import sys,os,pickle,random
sys.path.append(r'/filepath/weighted-distance-transform/')
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import netCDF4
import h5py
import copy
from numba import jit,prange
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy import config
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import ndimage as ndi
from scipy import stats
import scipy.io
from decimal import Decimal
import seaborn as sns
from PIL import Image
import imageio
import wdt


random.seed()

default_cwm_mask = r'/filepath/mask_cbpm_2021_01_13.txt'
default_cwm_area = r'/filepath/cwm_grid.mat'

seaweed_file_ambient = r'/filepath/Preferred_species_f0.nc'
seaweed_file_flux = r'/filepath/Preferred_species_f1.nc'

d2pfile = r'/filepath/d2port.nc'
depthfile = r'/filepath/gebco_cwm.nc'
wavefile = r'/filepath/wave_data.nc'
seqfracfile = r'/filepath/fseq_cwm_by_100y_linear_interp.nc'

#### Parameter ranges from literature
default_p_bounds = {'capex': [170630, 969626],
                    'linecost': [0.06, 1.45],
                    'labor': [37706, 119579],
                    'harvcost': [124485, 394780],
                    'transportcost': [0.1125, 0.3375],
                    'transportems': [0.0000142, 0.00004518], # calculated +- 50% from 0.00003012 tons CO2/ton-km with data from Aitken et al., 2014
                    'maintenanceboatems': [0, 0.0008715], # tons CO2/km per maintenance trip, assuming +50% uncertainty from Aitken et al., 2014 value of 0.000581 tons CO2/km/trip
                    'insur': [35000, 105000],
                    'license': [1409, 1637],
                    'opex': [63004, 69316],
                    'sinkval': [0, 0],
                    'sequestration_rate': [0.9, 1], #### assuming almost all (at least 90%) of the seaweed makes it below the mixed layer to bottom
                    'removal_rate': [0.5, 1], #### see paper Lydia sent? --> Matt's CESM experiment shows regional heterogeneity, global average of about 0.5
                    'productval': [300, 800],  ##### need more data on this
                    'avoidedems_product': [0.1, 1], # calculated roughly 0.4 from web search of kcal/tonDW and 1gCO2e/kcal
                    'convertcost': [24, 72], ##### $/tonDW, calculated +- 50% from average cost from PNNL macroalgae as a biomass feedstock (2010) assuming full plant feedstock capacity
                    'convertems': [0.0011, 0.0085], # calculated +- 50% from 8kWh/ton (0.005672 tons co2/ton DW) for fuel and 3kWh/ton for feed with data from Aitken et al., 2014
                    # 'convertfrac': [0.2, 0.5], # tons product/ton seaweed converted
                    'depth_mult': [0, 1], # referenced in van den burg
                    'wave_mult': [0, 1],
                    'seaweed_map': [0, 100] # for randomly picking seaweed map from biophysical MC runs
                    } 

#### Rounded parameter ranges assuming renewable electricity, lower capex min, rounded
p_bounds_renelec_rounded = {'capex': [10000, 1000000],
                    'linecost': [0.05, 1.45],
                    'labor': [38000, 120000],
                    'harvcost': [120000, 400000],
                    'transportcost': [0.1, 0.35], # van den burg et al., 2016
                    'transportems': [0, 0.000045], # min of 0 for potential electric or fuel cell ships, then calculated +- 50% from 0.00003012 tons CO2/ton-km with data from Aitken et al., 2014
                    'maintenanceboatems': [0, 0.0035], # tons CO2/km per maintenance trip, assuming +50% uncertainty using boat fuel consumption of 1l/1.14km (seagrant alaska) and methods from Aitken et al., 2014, resulting value of 0.0023653 tons CO2/km/trip
                    'insur': [35000, 105000],
                    'license': [1000, 2000],
                    'opex': [60000, 70000],
                    'sinkval': [0, 0],
                    'sequestration_rate': [1, 1], #### assuming almost all (at least 95%) of the seaweed makes it below the mixed layer to bottom
                    'removal_rate': [0.4, 1], #### see paper Lydia sent? --> Matt's CESM experiment shows regional heterogeneity, global average of about 0.5
                    'productval': [400, 800],  ##### animal feed ranges in value from 500-600/ton
                    'avoidedems_product': [0.7, 6.0], # calculated approximately 3-6 from web search of seaweed kcal/tonDW and 1gCO2e/kcal agricultural emissions (Hong et al), biofuel 0.79 tCO2/tDW from web search of co2 emissions per gallon jet fuel
                    'convertcost': [20, 80], ##### $/tonDW, calculated +- 50% from average cost from PNNL macroalgae as a biomass feedstock (2010) assuming full plant feedstock capacity
                    'convertems': [0, 0.01], # calculated +- 50% from 8kWh/ton (0.005672 tons co2/ton DW) for fuel and 3kWh/ton for feed with data from Aitken et al., 2014
                    # 'convertfrac': [0.2, 0.5], # tons product/ton seaweed converted
                    'depth_mult': [0, 1],
                    'wave_mult': [0, 1],
                    'seaweed_map': [0, 5] # for randomly picking seaweed map from biophysical MC runs
                    } 

p_bounds_seaweedmap = {'seaweed_map': [0, 100] # for randomly picking seaweed map from biophysical MC runs
                    } 

#### Uncomment line below to use renewable electricity/lower capex bounds (was used in paper analysis):
default_p_bounds = p_bounds_renelec_rounded

@jit(nopython=True)
def custom_quantile(arr,quant):
    nx,ny,nz = np.shape(arr)
    q = np.full((nx,ny),np.nan,np.float32)
    for i in range(nx):
        for j in range(ny):
            if not np.any(~np.isfinite(arr[i,j,:])):
                q[i,j] = np.quantile(arr[i,j,:],quant)
    return q

@jit(nopython=True,parallel=True,nogil=True)
def custom_stats(arr):
    nx,ny,nz = np.shape(arr)
    q05 = np.full((nx,ny),np.nan,np.float32)
    q25 = np.full((nx,ny),np.nan,np.float32)
    q75 = np.full((nx,ny),np.nan,np.float32)
    q95 = np.full((nx,ny),np.nan,np.float32)
    med = np.full((nx,ny),np.nan,np.float32)
    mean = np.full((nx,ny),np.nan,np.float32)
    mina = np.full((nx,ny),np.nan,np.float32)
    maxa = np.full((nx,ny),np.nan,np.float32)
    for i in prange(nx):
        for j in range(ny):
            if not np.any(~np.isfinite(arr[i,j,:])):
                q05[i,j] = np.nanquantile(arr[i,j,:],0.05)
                q25[i,j] = np.nanquantile(arr[i,j,:],0.25)
                q75[i,j] = np.nanquantile(arr[i,j,:],0.75)
                q95[i,j] = np.nanquantile(arr[i,j,:],0.95)
                med[i,j] = np.nanmedian(arr[i,j,:])
                mean[i,j] = np.nanmean(arr[i,j,:])
                mina[i,j] = np.nanmin(arr[i,j,:])
                maxa[i,j] = np.nanmax(arr[i,j,:])
    return (q05,q25,q75,q95,med,mean,mina,maxa)

@jit(nopython=True,parallel=True,nogil=True)
def custom_stats_v2(arr,q05,q25,q75,q95,med,mean,mina,maxa):
    nx,ny,nz = np.shape(arr)
    q05[...] = np.nan
    q25[...] = np.nan
    q75[...] = np.nan
    q95[...] = np.nan
    med[...] = np.nan
    mean[...] = np.nan
    mina[...] = np.nan
    maxa[...] = np.nan
    for i in prange(nx):
        for j in range(ny):
            # if not np.any(~np.isfinite(arr[i,j,:])):
                q05[i,j] = np.nanquantile(arr[i,j,:],0.05)
                q25[i,j] = np.nanquantile(arr[i,j,:],0.25)
                q75[i,j] = np.nanquantile(arr[i,j,:],0.75)
                q95[i,j] = np.nanquantile(arr[i,j,:],0.95)
                med[i,j] = np.nanmedian(arr[i,j,:])
                mean[i,j] = np.nanmean(arr[i,j,:])
                mina[i,j] = np.nanmin(arr[i,j,:])
                maxa[i,j] = np.nanmax(arr[i,j,:])

@jit(nopython=True,parallel=True,nogil=True)
def custom_stats_v3(arr,q05,q25,q75,q95,med,mean,mina,maxa):
    nx,ny,nz = np.shape(arr)
    q05[...] = np.nan
    q25[...] = np.nan
    q75[...] = np.nan
    q95[...] = np.nan
    med[...] = np.nan
    mean[...] = np.nan
    mina[...] = np.nan
    maxa[...] = np.nan
    for i in prange(nx):
        arr1 = np.full(nz,np.nan,np.float32)
        for j in range(ny):
            # if not np.any(~np.isfinite(arr[i,j,:])):
                nk = 0
                for k in range(nz):
                    if np.isfinite(arr[i,j,k]):
                        arr1[nk] = arr[i,j,k]
                        nk += 1
                if nk > 0:
                    q05[i,j] = np.quantile(arr1[:nk],0.05)
                    q25[i,j] = np.quantile(arr1[:nk],0.25)
                    q75[i,j] = np.quantile(arr1[:nk],0.75)
                    q95[i,j] = np.quantile(arr1[:nk],0.95)
                    med[i,j] = np.median(arr1[:nk])
                    mean[i,j] = np.mean(arr1[:nk])
                    mina[i,j] = np.min(arr1[:nk])
                    maxa[i,j] = np.max(arr1[:nk])

def ncread(ncfile, varname):
    # this should work but xarray is choking on the dates for some reason
    #ds = xr.open_dataset(ncfile)
    #data = ds[varname][...].values
    #ds.close()
    nc = netCDF4.Dataset(ncfile)
    data = nc.variables[varname][...].filled(np.nan)
    nc.close()
    return data

def get_area(cwm_grid_h5, area_varname = 'area'):
    with h5py.File(cwm_grid_h5, 'r') as fp:
        amask = fp['cwm_grid'][area_varname][...]
    return np.transpose(amask)

def load_cwm_lon_lat():
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)
    return longitude,latitude

def load_cwm_lon_lat_mesh():
    longitude, latitude = load_cwm_lon_lat()
    xlon, ylat = np.meshgrid(longitude, latitude)
    return longitude,latitude,xlon,ylat

def load_cwm_lon_lat_mesh_bounds():
    longitude,latitude,xlon,ylat = load_cwm_lon_lat_mesh()
    lonb = np.arange(4321) * 1/12 - 180
    latb = -1.0*(np.arange(2161) * 1/12 - 90)
    xlonb,ylatb = np.meshgrid(lonb, latb)
    return longitude,latitude,xlon,ylat,lonb,latb,xlonb,ylatb

def load_cwm_grid_mask(mask_fn=default_cwm_mask):
    longitude,latitude,xlon,ylat = load_cwm_lon_lat_mesh()
    mask = np.logical_not(np.loadtxt(mask_fn))
    return longitude,latitude,xlon,ylat,mask

def load_cwm_grid_area_bounds(mask_fn=default_cwm_area):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask(mask_fn)
    lonb = np.arange(4321) * 1/12 - 180
    latb = -1.0*(np.arange(2161) * 1/12 - 90)
    xlonb,ylatb = np.meshgrid(lonb, latb)
    area = get_area(default_cwm_area)
    return longitude,latitude,xlon,ylat,mask,area,lonb,latb,xlonb,ylatb

def random_params(p_bounds=default_p_bounds):
    """
    another way to do this that is more traditional, less "pythonic":

    rp = {} # dict()
    for param,bounds in p_bounds.items():
        rp[param] = random.uniform(bounds[0],bounds[1])

    """
    return {param: random.uniform(bounds[0],bounds[1]) for param,bounds in p_bounds.items() }

def normal_params(p_bounds=default_p_bounds):
    """
    another way to do this that is more traditional, less "pythonic":

    rp = {} # dict()
    for param,bounds in p_bounds.items():
        rp[param] = random.normal(bounds[0],bounds[1])

    """ # 889010
    # mu, sigma = 854.215,539.268  # using mean of q50 map for sigma
    mu, sigma = 50, 16.67 # using mean of q50 map  for sigma
    # mu, sigma = 1,0.4933 # divided each by mu (above) to normalize mean to 1
    return {param: random.gauss(mu, sigma) for param,bounds in p_bounds.items() }

def open_output_nc(outfile,lat,lon,vars,chunksizes=(72,4320,1)):
    nc = netCDF4.Dataset(outfile,'w')
    nc.createDimension('latitude',len(lat))
    nc.createDimension('longitude',len(lon))
    nc.createDimension('mc',None)  # unlimited dimension
    latvar = nc.createVariable('latitude','f8',('latitude'))
    latvar.setncatts({'units': "degrees_north",'long_name': "latitude"})
    lonvar = nc.createVariable('longitude','f8',('longitude'))
    lonvar.setncatts({'units': "degrees_east",'long_name': "longitude"})
    mcvar = nc.createVariable('mc','i4',('mc'))
    for v_name in vars:
        outVar = nc.createVariable(v_name, 'f4', ('latitude','longitude','mc'),zlib=True,
                                   complevel=1,chunksizes=chunksizes)
    latvar[...] = lat
    lonvar[...] = lon
    nc.sync()
    return nc

def create_cwm_annual_dataset(var_list,unit_list):
    # create dataset
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)
    var_dict = {}
    for var,unit in zip(var_list,unit_list):
        var_dict[var] = xr.DataArray(
            dims=['latitude','longitude'],
            coords={'latitude': latitude, 'longitude': longitude},
            attrs={'_FillValue': -999.9,'units': unit}
        )
    return xr.Dataset(var_dict)

def save_TEA_analysis(outfile,var_list,data_dict,unit_list):
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)
    var_dict = {}
    for var,unit in zip(var_list,unit_list):
        var_dict[var] = xr.DataArray(
            dims=['latitude','longitude'],
            coords={'latitude': latitude, 'longitude': longitude},
            attrs={'_FillValue': -999.9,'units': unit},
            data=data_dict[var]
        )
    encoding = {v:{"dtype":np.float32,"zlib": True, "complevel": 4} for v in var_list}
    ds = xr.Dataset(var_dict)
    ds.to_netcdf(outfile,format='netCDF4',encoding=encoding)

def cwm_map(cwm_data,clims,title,cmap):
    fig=plt.figure(figsize=[9,4])
    ax = plt.gca()
    img = plt.imshow(cwm_data,cmap=cmap,clim=clims)
    plt.colorbar(img)
    #plt.grid(linestyle='--',linewidth=0.25)
    #plt.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.title(title)
    print('Creating: %s'%title)
    return fig

def cwm_map_pcar(cwm_data,clims,title,ylat,xlon,cmap):
    # fig1=plt.figure(figsize=[8,6])
    ax = plt.axes(projection=ccrs.PlateCarree())  # h=10km resolution, for coasts, etc
    plt.pcolormesh(xlon,ylat,cwm_data,cmap=cmap,vmin=clims[0],vmax=clims[1])
    plt.colorbar()
    ax.coastlines(linewidth=0.33)
    plt.title(title)
    return ax

def area_weighted_avg(data,seaweed,area,percentile):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    weights = area / np.min(area)
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    seaweedmask = copy.deepcopy(seaweed)
    seaweedmask[seaweedmask >=0] = 1
    
    data = data * landmask * seaweedmask
    data_copy1 = copy.deepcopy(data)
    data_copy2 = copy.deepcopy(data)
    
    num_data = data_copy1.size - (np.isnan(data_copy1).sum()) # - ((data_copy1 >= 10000000).sum())
    num_data_toavg = num_data / (100/percentile)
    k_index = int(num_data_toavg - 1)
    # print(k_index)
    num_data_toavg_row = np.sort(data_copy2.ravel())
    # print(len(num_data_toavg_row))
    num_data_toavg_row = num_data_toavg_row[(~np.isnan(num_data_toavg_row))] # (num_data_toavg_row < 10000000) & 
    # print(len(num_data_toavg_row))
    if len(num_data_toavg_row) > 0 & k_index <=len(num_data_toavg_row):
        percentile_k = num_data_toavg_row[k_index]
        data[data > percentile_k] = np.nan
        data_map = copy.deepcopy(data)
        weights[np.isnan(data)] = 0;
        data[np.isnan(data)] = 0;
        percentile_area_average = np.average(data, weights = weights)
    else:
        data_map = copy.deepcopy(data)
        percentile_area_average = np.nan
    
    return percentile_area_average,data_map

def area_weighted_avg_maxtomin(data,seaweed,area,percentile):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    weights = area / np.min(area)
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    seaweedmask = copy.deepcopy(seaweed)
    seaweedmask[seaweedmask >=0] = 1
    
    data = data * landmask * seaweedmask
    data_copy1 = copy.deepcopy(data)
    data_copy2 = copy.deepcopy(data)
    
    num_data = data_copy1.size - (np.isnan(data_copy1).sum()) # - ((data_copy1 >= 10000000).sum())
    num_data_toavg = num_data / (100/percentile)
    k_index = int(num_data_toavg - 1)
    # print(k_index)
    num_data_toavg_row = np.sort(data_copy2.ravel())
    num_data_toavg_row = num_data_toavg_row[::-1]
    # print(len(num_data_toavg_row))
    num_data_toavg_row = num_data_toavg_row[(~np.isnan(num_data_toavg_row))] # (num_data_toavg_row < 10000000) & 
    # print(len(num_data_toavg_row))
    if len(num_data_toavg_row) > 0 & k_index <=len(num_data_toavg_row):
        percentile_k = num_data_toavg_row[k_index]
        data[data < percentile_k] = np.nan
        data_map = copy.deepcopy(data)
        weights[np.isnan(data)] = 0;
        data[np.isnan(data)] = 0;
        percentile_area_average = np.average(data, weights = weights)
    else:
        data_map = copy.deepcopy(data)
        percentile_area_average = np.nan
    
    return percentile_area_average,data_map

def get_regional_mean(data,region_latlon,area,xlon,ylat):
    # now getting average value for region
    percentile = 1
    weights = area / np.min(area)
    lon_copy = copy.deepcopy(xlon)
    if region_latlon[0] < region_latlon[1]:
        lon_copy[lon_copy <= region_latlon[0]] = np.nan
        lon_copy[lon_copy >= region_latlon[1]] = np.nan
        lon_copy[np.isfinite(lon_copy)] = 1
        lat_copy = copy.deepcopy(ylat)
        lat_copy[lat_copy <= region_latlon[2]] = np.nan
        lat_copy[lat_copy >= region_latlon[3]] = np.nan
        lat_copy[np.isfinite(lat_copy)] = 1
    elif region_latlon[0] > region_latlon[1]:
        lon_copy[(lon_copy <= region_latlon[0]) & (lon_copy >= region_latlon[1])] = np.nan
        lon_copy[np.isfinite(lon_copy)] = 1
        lat_copy = copy.deepcopy(ylat)
        lat_copy[lat_copy <= region_latlon[2]] = np.nan
        lat_copy[lat_copy >= region_latlon[3]] = np.nan
        lat_copy[np.isfinite(lat_copy)] = 1
    lonlat_box = lon_copy * lat_copy
    regional_data_box = data * lonlat_box
    # weights[np.isnan(regional_data_box)] = 0;
    # weights[regional_data_box > 10000000] = 0;
    # regional_data_box[np.isnan(regional_data_box)] = 0;
    num_data = regional_data_box.size - (np.isnan(regional_data_box).sum()) # - ((data_copy1 >= 10000000).sum())
    num_data_toavg = num_data / (100/percentile)
    k_index = int(num_data_toavg - 1)
    num_data_toavg_row = np.sort(regional_data_box.ravel())
    num_data_toavg_row = num_data_toavg_row[(~np.isnan(num_data_toavg_row))]
    if len(num_data_toavg_row) > 0 & k_index <=len(num_data_toavg_row):
        percentile_k = num_data_toavg_row[k_index]
        regional_data_box[regional_data_box > percentile_k] = np.nan
        data_map = copy.deepcopy(regional_data_box)
        weights[np.isnan(regional_data_box)] = 0;
        regional_data_box[np.isnan(regional_data_box)] = 0;
        region_mean = np.average(regional_data_box, weights = weights)
        region_mean = int(region_mean)
    else:
        region_mean = np.nan
    # regional_data_box[regional_data_box > 10000000] = 0;
    # region_mean = np.average(regional_data_box, weights = weights)
    # region_mean = int(region_mean)
    return region_mean, regional_data_box  

def cost_binned_area(data,seaweed,area,bin_size,bin_start):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    weights = area / np.min(area)
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    seaweedmask = copy.deepcopy(seaweed)
    seaweedmask[seaweedmask >=0] = 1
    ocean_area = 361900000 #km2
    
    bin_vars = ['bin_100_area','bin_200_area','bin_300_area','bin_400_area','bin_500_area','bin_600_area',
                'bin_700_area','bin_800_area','bin_900_area','bin_1000_area','bin_1100_area',
                'bin_1200_area','bin_1300_area','bin_1400_area','bin_1500_area','bin_1600_area',
                'bin_1700_area','bin_1800_area','bin_1900_area','bin_2000_area']
    
    data = data * landmask * seaweedmask
    data_copy1 = copy.deepcopy(data)
    
    bin_min = bin_start
    bin_max = bin_start + bin_size
    
    for i in range(len(bin_vars)):
        # print(bin_min)
        # print(bin_max)

        data[data < bin_min] = np.nan
        data[data > bin_max] = np.nan
        data_copy = copy.deepcopy(data)
        num_data_toavg_row = data_copy.ravel()
        num_data_toavg = num_data_toavg_row[(~np.isnan(num_data_toavg_row))]
        # print(len(num_data_toavg))
        
        if len(num_data_toavg) > 0:
            data_map = copy.deepcopy(data)
            # weights[np.isnan(data)] = 0;
            # data[np.isnan(data)] = 0;
            # bin_avg = np.average(data, weights = weights)
            bin_area = copy.deepcopy(data_map)
            bin_area[~np.isnan(bin_area)] = 1
            farmed_area1 = bin_area * area * landmask # tonsCO2/tonDW * tonsDW/km2 * km2/gridcell
            farmed_area_map = copy.deepcopy(farmed_area1)
            farmed_area_flat = farmed_area1.flatten()
            farmed_area_flat = farmed_area_flat[~np.isnan(farmed_area_flat)]
            percent_oceanarea_farmed = (((sum(farmed_area_flat))/ocean_area)*100)
            bin_area_farmed_km2 = (sum(farmed_area_flat))
            
        else:
            bin_area_farmed_km2 = 0
        
        globals()[bin_vars[i]] = bin_area_farmed_km2

        # print(bin_vars[i])
        
        bin_min = bin_max
        bin_max = bin_max + bin_size
        
        data = copy.deepcopy(data_copy1)
        # weights = copy.deepcopy(weights_copy1)
    
    print(bin_2000_area)
    
    return bin_100_area, bin_200_area, bin_300_area, bin_400_area, bin_500_area, bin_600_area, bin_700_area, bin_800_area, bin_900_area, bin_1000_area, bin_1100_area, bin_1200_area, bin_1300_area, bin_1400_area, bin_1500_area, bin_1600_area, bin_1700_area, bin_1800_area, bin_1900_area, bin_2000_area

def percentile_binned_avg(data,seaweed,area,bin_size,bin_start):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    weights = area / np.min(area)
    weights_copy1 = copy.deepcopy(weights)
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    seaweedmask = copy.deepcopy(seaweed)
    seaweedmask[seaweedmask >=0] = 1
    
    data = data * landmask * seaweedmask
    data_copy1 = copy.deepcopy(data)
    data_copy2 = copy.deepcopy(data)
    data_copy3 = copy.deepcopy(data)
    
    bin_vars = ['percentile_point4','percentile_point8','percentile_1.2','percentile_1.6','percentile_2']
    
    num_data = data_copy1.size - (np.isnan(data_copy1).sum()) # - ((data_copy1 >= 10000000).sum())
    num_data_toavg = num_data / (100/bin_size)
    k_start_index = int(((bin_start/bin_size)*num_data_toavg))
    k_stop_index = int((num_data_toavg + ((bin_start/bin_size)*num_data_toavg)) - 1)
    # print(k_index)
    num_data_toavg_row = np.sort(data_copy2.ravel())
    # print(len(num_data_toavg_row))
    num_data_toavg_row = num_data_toavg_row[(~np.isnan(num_data_toavg_row))] # (num_data_toavg_row < 10000000) & 
    for i in range(len(bin_vars)):
        if len(num_data_toavg_row) > 0 & k_stop_index <=len(num_data_toavg_row):
            percentile_k_start = num_data_toavg_row[k_start_index]
            percentile_k_stop = num_data_toavg_row[k_stop_index]
            data[data > percentile_k_stop] = np.nan
            data[data < percentile_k_start] = np.nan
            data_map = copy.deepcopy(data)
            weights[np.isnan(data)] = 0;
            data[np.isnan(data)] = 0;
            percentile_area_average = np.average(data, weights = weights)
        else:
            data_map = copy.deepcopy(data)
            percentile_area_average = np.nan
        globals()[bin_vars[i]] = percentile_area_average
        k_start_index = int(k_start_index + num_data_toavg)
        k_stop_index = int(k_stop_index + num_data_toavg)
        data = copy.deepcopy(data_copy3)
        weights = copy.deepcopy(weights_copy1)
    
    return percentile_point4,percentile_point8,percentile_1p2,percentile_1p6,percentile_2

def cwm_map_fig1(cwm_data1,cwm_data2,cwm_data3,clims,title,area,seaweed,ylat,xlon,cmap):
    
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    
    fig = plt.figure()
    projection_global = ccrs.Robinson()
    projection_regional = ccrs.PlateCarree()


    ax1 = plt.subplot2grid((3,7), (0,0), colspan=3, facecolor = 'k', projection=projection_global)
    ax1.set_title('Global')
    ax2 = plt.subplot2grid((3,7), (0,3), facecolor = 'k', projection=projection_regional)
    ax2.set_title('North\nAtlantic')
    ax3 = plt.subplot2grid((3,7), (0,4), facecolor = 'k', projection=projection_regional)
    ax3.set_title('South\nAtlantic')
    ax4 = plt.subplot2grid((3,7), (0,5), facecolor = 'k', projection=projection_regional)
    ax4.set_title('North\nPacific')
    ax5 = plt.subplot2grid((3,7), (0,6), facecolor = 'k', projection=projection_regional)
    ax5.set_title('South\nPacific')
    ax6 = plt.subplot2grid((3,7), (1,0), colspan=3, facecolor = 'k', projection=projection_global)
    ax7 = plt.subplot2grid((3,7), (1,3), facecolor = 'k', projection=projection_regional)
    ax8 = plt.subplot2grid((3,7), (1,4), facecolor = 'k', projection=projection_regional)
    ax9 = plt.subplot2grid((3,7), (1,5), facecolor = 'k', projection=projection_regional)
    ax10 = plt.subplot2grid((3,7), (1,6), facecolor = 'k', projection=projection_regional)
    ax11 = plt.subplot2grid((3,7), (2,0), colspan=3, facecolor = 'k', projection=projection_global)
    ax12 = plt.subplot2grid((3,7), (2,3), facecolor = 'k', projection=projection_regional)
    ax13 = plt.subplot2grid((3,7), (2,4), facecolor = 'k', projection=projection_regional)
    ax14 = plt.subplot2grid((3,7), (2,5), facecolor = 'k', projection=projection_regional)
    ax15 = plt.subplot2grid((3,7), (2,6), facecolor = 'k', projection=projection_regional)
    
    cwm_data1 = copy.deepcopy(cwm_data1)
    cwm_data2 = copy.deepcopy(cwm_data2)
    cwm_data3 = copy.deepcopy(cwm_data3)
    
    rows = 3
    cols = 5
    i=0
    
    data = [cwm_data1,cwm_data2,cwm_data3]
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]
    north_atlantic_latlon = [-5,5,50,65] # left, right, bottom, top
    south_atlantic_latlon = [-70,-60,-60,-45]
    north_pacific_latlon = [-160,-150,45,60]
    south_pacific_latlon = [170,180,-50,-35]
    # south_pacific_latlon = [170,-170,-50,-35]
    
    for i in range(rows):
    # while i<rows:
        ax_col1=axes[(cols*i)]  # h=10km resolution, for coasts, etc
        # ax_col1.coastlines()
        global_map = ax_col1.pcolormesh(xlon,ylat,data[i],cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
        # ax_col1.coastlines()
        ax_col1.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
        # ax_col1.add_feature(cf.COASTLINE)
        # global_mean = int(np.nanmean(data[i]))
        global_mean, global_mean_map = area_weighted_avg(data[i],seaweed,area,percentile=0.89)
        ax_col1.text(0.5, -0.2, str(int(global_mean)), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax_col1.transAxes)
        
        ax_col2=axes[1+(cols*i)] #lat_0 = -50, lon_0 = -60, 
        ax_col2.set_extent(north_atlantic_latlon, crs=projection_regional)
        ax_col2.pcolormesh(xlon,ylat,data[i],cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto')
        # ax_col2.coastlines()
        ax_col2.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
        # ax_col2.add_feature(cf.COASTLINE)
        region1_mean, region1_data_box = get_regional_mean(data[i],north_atlantic_latlon,area,xlon,ylat)
        ax_col2.text(0.5, -0.2, str(region1_mean), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax_col2.transAxes)
        
        ax_col3=axes[2+(cols*i)] #lat_0 = -50, lon_0 = -60,
        ax_col3.set_extent(south_atlantic_latlon, crs=projection_regional)
        ax_col3.pcolormesh(xlon,ylat,data[i],cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto')
        # ax_col3.coastlines()
        ax_col3.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
        # ax_col3.add_feature(cf.COASTLINE)
        region2_mean, region2_data_box = get_regional_mean(data[i],south_atlantic_latlon,area,xlon,ylat)
        ax_col3.text(0.5, -0.2, str(region2_mean), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax_col3.transAxes)
    
        ax_col4=axes[3+(cols*i)] #lat_0 = 50, lon_0 = -130, 
        ax_col4.set_extent(north_pacific_latlon, crs=projection_regional)
        ax_col4.pcolormesh(xlon,ylat,data[i],cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto')
        # ax_col4.coastlines()
        ax_col4.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
        # ax_col4.add_feature(cf.COASTLINE)
        region3_mean, region3_data_box = get_regional_mean(data[i],north_pacific_latlon,area,xlon,ylat)
        ax_col4.text(0.5, -0.2, str(region3_mean), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax_col4.transAxes)
        
        ax_col5=axes[4+(cols*i)] #lat_0 = 50, lon_0 = -130, 
        ax_col5.set_extent(south_pacific_latlon, crs=projection_regional)
        ax_col5.pcolormesh(xlon,ylat,data[i],cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto')
        # ax_col5.coastlines()
        ax_col5.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
        # ax_col5.add_feature(cf.COASTLINE)
        region4_mean, region4_data_box = get_regional_mean(data[i],south_pacific_latlon,area,xlon,ylat)
        ax_col5.text(0.5, -0.2, str(region4_mean), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax_col5.transAxes)
    
        print('\nstill making figure...')
    
    cb_ax = fig.add_axes([0.94, 0.12, 0.02, 0.77])
    cbar = fig.colorbar(global_map, cax=cb_ax)
    
    print('done, now saving and plotting...')
    
    fig.savefig('seaweed_cultivation_cost.png', dpi=900, format='png', bbox_inches='tight')
    
    return fig, region1_data_box, region2_data_box, region3_data_box, region4_data_box

def cwm_products_map_fig2(mc_data_file,clims,title,area,seaweed,ylat,xlon,cmap):
    fig = plt.figure()
    projection_global = ccrs.Robinson()

    ax1 = plt.subplot2grid((2,2), (0,0), facecolor = 'k', projection=projection_global)
    ax1.set_title('Sinking')
    ax2 = plt.subplot2grid((2,2), (0,1), facecolor = 'k', projection=projection_global)
    ax2.set_title('Products - Food')
    ax3 = plt.subplot2grid((2,2), (1,0), facecolor = 'k', projection=projection_global)
    ax3.set_title('Products - Animal Feed')
    ax4 = plt.subplot2grid((2,2), (1,1), facecolor = 'k', projection=projection_global)
    ax4.set_title('Products - Biofuel')

    meta_file = os.path.expanduser(r'/filepath/ambient_data_params.csv')
    meta = pd.read_csv(meta_file)
    
    # percentiles correspond to percent of seaweed growth pixels - to get ocean area, need to calculate what percent of pixels correspond to 1% ocean area for each map (using CDF plot function) and then adjust percentile input accordingly
    sinking_median_map = ncread(r'/filepath/Preferred_species_f0.nc','Harvest_q50')
    print('plotting sinking map...')
    # sinking_map = ax1.pcolormesh(xlon,ylat,sinking_median_map,cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
    ax1.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
    sinking_mean, sinking_mean_map = area_weighted_avg_maxtomin(sinking_median_map,seaweed,area,percentile=0.2)
    sinking_map = ax1.pcolormesh(xlon,ylat,sinking_mean_map,cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
    ax1.text(0.5, -0.2, str(int(sinking_mean)), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax1.transAxes)
    
    food_data_analysis = TEA_analysis_products_fig2(r'/filepath/flux_data.nc','net_costperton_product',meta_file,[1,6],[500,800],r'/filepath/flux_netcostpertonfood_analysis.nc',chunksizes=chunks)
    print('running food MC analysis...')
    food_median_map = food_data_analysis['tea_q5']
    # food_median_map = ncread(r'/filepath/ambient_netcostpertonfood_analysis.nc','tea_q5') # food_data_analysis['tea_median']
    print('plotting food map...')
    ax2.pcolormesh(xlon,ylat,food_median_map,cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
    ax2.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
    food_mean, food_mean_map = area_weighted_avg(food_median_map,seaweed,area,percentile=1.18)
    ax2.text(0.5, -0.2, str(int(food_mean)), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax2.transAxes)
    
    feed_data_analysis = TEA_analysis_products_fig2(r'/filepath/flux_data.nc','net_costperton_product',meta_file,[1,3.1],[400,500],r'/filepath/flux_netcostpertonfeed_analysis.nc',chunksizes=chunks)
    print('running feed MC analysis...')
    feed_median_map = feed_data_analysis['tea_q5']
    # feed_median_map = ncread(r'/filepath/ambient_netcostpertonfeed_analysis.nc','tea_q5') # feed_data_analysis['tea_median']
    print('plotting feed map...')
    ax3.pcolormesh(xlon,ylat,feed_median_map,cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
    ax3.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
    feed_mean, feed_mean_map = area_weighted_avg(feed_median_map,seaweed,area,percentile=1.16)
    ax3.text(0.5, -0.2, str(int(feed_mean)), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax3.transAxes)
    
    fuel_data_analysis = TEA_analysis_products_fig2(r'/filepath/flux_data.nc','net_costperton_product',meta_file,[0.7,1],[400,500],r'/filepath/flux_netcostpertonfuel_analysis.nc',chunksizes=chunks)
    print('running fuel MC analysis...')
    fuel_median_map = fuel_data_analysis['tea_q5']
    # fuel_median_map = ncread(r'/filepath/ambient_netcostpertonfuel_analysis.nc','tea_q5') # fuel_data_analysis['tea_median']
    print('plotting fuel map...')
    ax4.pcolormesh(xlon,ylat,fuel_median_map,cmap=cmap,vmin=clims[0],vmax=clims[1],shading='auto',transform=ccrs.PlateCarree())
    ax4.add_feature(cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='0.7'))
    fuel_mean, fuel_mean_map = area_weighted_avg(fuel_median_map,seaweed,area,percentile=1.21)
    ax4.text(0.5, -0.2, str(int(fuel_mean)), va='bottom', ha='center',
                     rotation='horizontal', rotation_mode='anchor',
                     transform=ax4.transAxes)
    
    cb_ax = fig.add_axes([0.96, 0.12, 0.02, 0.77])
    cbar = fig.colorbar(sinking_map, cax=cb_ax)
    
    print('done, now saving and plotting.')
    # fig.savefig('SIfigure6_productcost_seaweed_TEA_5000runs_05_16_22.png', dpi=900, format='png', bbox_inches='tight')
    
    return fig

def distance_to_sink(seaweed,species,nharv,outfile,years):
    # load data sources
    longitude,latitude = load_cwm_lon_lat()
    carbon_fraction = 0.3 # tons C/ton DW
    carbon_to_co2 = 3.67 # tons CO2/ton C
    fseq_100y = ncread(seqfracfile,'fseq_bottom_100y')
    fseq_200y = ncread(seqfracfile,'fseq_bottom_200y')
    longitude,latitude = load_cwm_lon_lat()
    d2port = ncread(d2pfile,'d2p')
    depth = ncread(depthfile,'elevation') * (-1.0)
    waves = ncread(wavefile,'wave height (mean)')
    waves = np.transpose(waves)
    
   #  species = ncread(seaweed_file,'index_Hq5')
    
    seaweed_ww = seaweed / 0.1

    equipment = copy.deepcopy(species)
    equipment[equipment == 0] = 1231.87 # eucheuma line spacing 0.2 m
    equipment[equipment == 1] = 185.24 # sargassum line spacing 1.33 m
    equipment[equipment == 2] = 4927.50 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
    equipment[equipment == 3] = 164.25 # saccharina (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m
    equipment[equipment == 4] = 164.25 # macrocystis (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m

    linedensity = copy.deepcopy(species)
    linedensity[linedensity == 0] = 5000000 # m line per km2, eucheuma line spacing 0.2 m
    linedensity[linedensity == 1] = 751880 # sargassum line spacing 1.33 m
    linedensity[linedensity == 2] = 20000000 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
    linedensity[linedensity == 3] = 666667 # saccharina (kelp) line spacing 1.5 m
    linedensity[linedensity == 4] = 666667 # macrocystis (kelp) line spacing 1.5 m
        
    linespacing = copy.deepcopy(species)
    # used to calculate capital cost increase of support ropes/anchoring with depth
    # FOR PORPHYRA: NOT equivalent to net mesh size. Equivalent to spacing between nets.
    linespacing[linespacing == 0] = 0.2 # eucheuma line spacing 0.2 m
    linespacing[linespacing == 1] = 1.33 # sargassum line spacing 1.33 m
    linespacing[linespacing == 2] = 2 # porphyra NET support line spacing, 0.1m grid, total amount of line equal to 0.05m longline spacing with support ropes every 2m
    linespacing[linespacing == 3] = 1.5 # saccharina (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m
    linespacing[linespacing == 4] = 1.5 # macrocystis (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m

    # nharv = ncread(seaweed_file,'nharv_Hq5')
    
    d_cost = depth / 500 # 500m is the threshold above which costs increase as a function of depth because anchorage design changes - see Yu et al., 2020
    d_cost[d_cost < 1] = 0
    # d_cost = 1
    w_cost = waves / 3 # 3m is the global average significant wave height, so we assume that above 3m waves will impact the lifetime of capital 
    w_cost[w_cost < 1] = 0

    # cost, value, emissions median values, using p_bounds_renelec
    mean_capex = 570128
    mean_linecost = 0.755
    mean_labor = 78642.5
    mean_harvcost = 259632.5
    mean_transport_cost = 0.225 # $/t-DW/km
    mean_transport_ems = 0.00002259 # t-CO2/t-DW-km
    mean_maintenanceboat_ems = 0.00177396 # t-CO2/km
    mean_insur = 70000
    mean_license = 1523
    mean_opex = 66160
    depth_mult = 1
    wave_mult = 1
    mean_removal_rate = 0.7
    mean_sequestration_rate = 1 # 0.975
    sinkval = 0
    
    maintenance_distance = 50 # km/km2, travel around each km2 for maintenance
    maintenance_area_pertrip = 0.5 # km2
    maintenance_trips = 6 # amount of times each km2 is visited for maintenance per year
    
    # using mask, and multiplying by area to convert from grid units to km
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    seaweedmask = copy.deepcopy(seaweed)
    seaweedmask[seaweedmask >=0] = 1
    area = get_area(default_cwm_area)
    distance_map = area**(1/2)
    
    #### pseudo-code for testing differnt fseq transport distances
    # fseq_target = 1
    # save fseq original with land mask nans
    # do cost/value calcs with original fseq map
    # save them as fseq_current
    # while fseq_target <= 100:
    #     seq_frac = seq_frac * 100
    #     do d2sink
    #     do cost/value calcs with each fseq_target d2sink map
    #     save 
    #     compare to saved original fseq_current cost/value maps (per grid cell)
    #     if fseq_target s better than fseq_current, update fseq_current
    #     fseq_target+=1
    # output final fseq and d2sink maps
    
    if years==100:
        fseq_original = fseq_100y * landmask
        fseq_previous = fseq_100y * landmask
        # fseq_previous[fseq_previous < 0.001] = 0.001
    elif years==200:
        fseq_original = fseq_200y * landmask
        fseq_previous = fseq_200y * landmask
    # fseq_original = fseq_original_100y_aspercent
    d2sink_previous = 0
    
    start = 0.01
    step = 0.01
    fseq_max = np.nanmax(fseq_original)
    
    targets_list = np.arange(start,1.001,step)
    
    if fseq_max in targets_list:
        stop = fseq_max + (step / 10)
    else:
        stop = fseq_max
    # print(stop)
    fseq_targets_list = np.arange(start,stop,step)
    print(fseq_targets_list)
    #
    for fseq_target in fseq_targets_list: #(start,stop,step):
        print('testing next fseq threshold')
        seq_frac_notransport = fseq_previous
        seq_frac_withtransport = copy.deepcopy(fseq_previous)
        seq_frac_withtransport[seq_frac_withtransport < fseq_target] = fseq_target
        
        # making binary matrix to locate points where target fraction sequestered for at least 500 years
        sink = copy.deepcopy(fseq_previous)
        sink[sink >= fseq_target] = 2
        sink[sink < fseq_target] = 0
        ## sink[np.isnan(sink)] = 0
        sink[np.isnan(sink)] = -1 ##
        ## sink = np.logical_not(sink)
        sink_width, sink_height = sink.shape[1], sink.shape[0]
        sink_rgb = np.zeros([sink_height,sink_width,3], dtype=np.uint8)
        sink_rgb[sink==0] = [255,255,255]
        sink_rgb[sink==2] = [0,128,0]
        sink_rgb[sink==-1] = [0,0,0]
        sink_image = Image.fromarray(sink_rgb)
        sink_image.save('d2sink_map.png')
        sink_field = wdt.map_image_to_costs('d2sink_map.png')
        d2sink1 = wdt.get_weighted_distance_transform(sink_field)
        
        # making another bindary matrix to retrieve seaweed values from shallower cells and move them to deeper cells
        ## shallow = np.logical_not(sink)
        
        # finding distance from each point to nearest point where target fraction sequestered for at least 500 years
        # also returns index of the nearest sinking point for each grid cell
        ## d2sink, sinkidx = ndi.distance_transform_edt(sink, return_indices=True)
        
        ## d2shallow, shallowidx = ndi.distance_transform_edt(shallow, return_indices=True)
        d2sink = copy.deepcopy(d2sink1)
        d2sink = d2sink * distance_map * 255 * landmask #### change this 9 to cwm_grid_area
        ## d2shallow = d2shallow * distance_map * mask
        
        #### resolve prod ems and cost vs post-prod ems and cost with seaweed_ww
        # calculating whether to transport or sink in place
        totlinecost = mean_linecost * linedensity # $/km^2
        w_cost[w_cost >= 1] = wave_mult
        # max wave height is about 6m, so max w_cost is around 5, i.e. capital is replaced every other year (every 2 years)
        d_cost[d_cost >= 1] = depth_mult 
        # only impacts anchoring line cost, so equivalent to = p['depth_mult'] * 0.31 but capital costs overall from depth increase can double, so max d_mult = 1
        cost = (((mean_capex + (mean_capex * d_cost) + (mean_capex * w_cost)) + (mean_opex) + (mean_insur) + (mean_license) + (mean_labor) + (totlinecost) + ((mean_harvcost * nharv) + (mean_transport_cost * equipment * d2port))) / seaweed)
        ems = (((equipment * d2port * mean_transport_ems) / seaweed) + ((((d2port * 2) * mean_maintenanceboat_ems * maintenance_trips / maintenance_area_pertrip) + (maintenance_distance * mean_maintenanceboat_ems)) / seaweed))
        #
        removed_ems_sink_withtransport = (mean_removal_rate * mean_sequestration_rate * carbon_fraction * carbon_to_co2 * seq_frac_withtransport)
        removed_ems_sink_notransport = (mean_removal_rate * mean_sequestration_rate * carbon_fraction * carbon_to_co2 * seq_frac_notransport)
        #
        val_sink_withtransport = (sinkval - (((mean_transport_cost * d2sink * seaweed_ww) + (mean_transport_cost * 2 * d2sink * equipment) + (mean_transport_cost * d2port * equipment)) / seaweed))
        netcost_sink_withtransport = cost - val_sink_withtransport
        #
        val_sink_notransport = (sinkval - (((mean_transport_cost * d2sink_previous * seaweed_ww) + (mean_transport_cost * 2 * d2sink_previous * equipment) + (mean_transport_cost * d2port * equipment)) / seaweed))
        netcost_sink_notransport = cost - val_sink_notransport
        #
        ems_sink_withtransport = (removed_ems_sink_withtransport - (((mean_transport_ems * d2sink * seaweed_ww) + (mean_transport_ems * 2 * d2sink * equipment) + (mean_transport_ems * d2port * equipment)) / seaweed))
        emsnet_sink_withtransport = ems_sink_withtransport - ems
        emsnet_sink_withtransport[emsnet_sink_withtransport <= 0] = np.inf
        #
        ems_sink_notransport = (removed_ems_sink_notransport - (((mean_transport_ems * d2sink_previous * seaweed_ww) + (mean_transport_ems * 2 * d2sink_previous * equipment) + (mean_transport_ems * d2port * equipment)) / seaweed))
        emsnet_sink_notransport = ems_sink_notransport - ems
        emsnet_sink_notransport[emsnet_sink_notransport <= 0] = np.inf
        
        netcostperton_sink_withtransport = netcost_sink_withtransport / emsnet_sink_withtransport
        netcostperton_sink_notransport = netcost_sink_notransport / emsnet_sink_notransport
        
        # comparing with vs without transport net cost per ton to choose transport or no transport for each cell
        sinkhere_or_sinkthere = netcostperton_sink_notransport / netcostperton_sink_withtransport
        sinkhere_or_sinkthere[sinkhere_or_sinkthere <= 1.00001] = 0
        sinkhere_or_sinkthere[emsnet_sink_notransport == np.inf] = 1
        sinkhere_or_sinkthere[emsnet_sink_withtransport == np.inf] = 0
        sinkhere_or_sinkthere = sinkhere_or_sinkthere * landmask
        sinkhere_or_sinkthere[np.isnan(sinkhere_or_sinkthere)] = 0
        # sinkhere_or_sinkthere[sinkhere_or_sinkthere < 0] = -1
        sinkhere_or_sinkthere[sinkhere_or_sinkthere > 1.00001] = 1
        # 0 = no transport, 1 = transport
        ## d2sink, sinkidx = ndi.distance_transform_edt(sinkhere_or_sinkthere, return_indices=True)
        
        fseq_withtransport1 = copy.deepcopy(sinkhere_or_sinkthere)
        # fseq_100y_withtransport1 = np.logical_not(fseq_100y_withtransport1)
        fseq_withtransport1 = fseq_withtransport1.astype(float)
        fseq_withtransport = copy.deepcopy(fseq_withtransport1)
        fseq_withtransport[fseq_withtransport == 0] = np.nan
        # fseq_100y_withtransport[np.isfinite(fseq_100y_withtransport)] = 1
        fseq_withtransport2 = fseq_withtransport * fseq_target
        fseq_withtransport = np.ma.masked_where(~np.isnan(fseq_withtransport2),fseq_withtransport2)
        fseq_withtransport[np.isnan(fseq_withtransport)] = 1
        fseq_withtransport = fseq_withtransport * fseq_previous * landmask # * seaweedmask
        fseq_withtransport = fseq_withtransport.data
        
        d2sink_new1 = copy.deepcopy(sinkhere_or_sinkthere)
        # d2sink_new1 = np.logical_not(d2sink_new1)
        d2sink_new1 = d2sink_new1.astype(float)
        d2sink_new = copy.deepcopy(d2sink_new1)
        d2sink_new[d2sink_new == 0] = np.nan
        # d2sink_new[np.isfinite(d2sink_new)] = 1
        d2sink_new2 = d2sink_new * d2sink
        d2sink_new = np.ma.masked_where(~np.isnan(d2sink_new2),d2sink_new2)
        d2sink_new[np.isnan(d2sink_new)] = 1
        d2sink_new = d2sink_new * d2sink_previous * landmask
        d2sink_new = d2sink_new.data
        
        fseq_previous = fseq_withtransport
        fseq_previous = copy.deepcopy(fseq_previous)
        d2sink_previous = d2sink_new
        d2sink_previous = copy.deepcopy(d2sink_previous)
        print(fseq_target)
        
    fseq_optimizedtransport = copy.deepcopy(fseq_previous)
    # fseq_100y_optimizedtransport = fseq_100y_optimizedtransport # / 100
    d2sink_optimizedtransport = copy.deepcopy(d2sink_previous)
    
    if years==100:
        vars=['d2sink','fseq_100years','fseq_100years_withtransport'] #,'d2sink_200y','fseq_200years','fseq_200years_withtransport']
        nc_d2sink_fseq = netCDF4.Dataset(outfile,'w')
        nc_d2sink_fseq.createDimension('latitude',len(latitude))
        nc_d2sink_fseq.createDimension('longitude',len(longitude))
        latvar = nc_d2sink_fseq.createVariable('latitude','f8',('latitude'))
        latvar.setncatts({'units': "degrees_north",'long_name': "latitude"})
        lonvar = nc_d2sink_fseq.createVariable('longitude','f8',('longitude'))
        lonvar.setncatts({'units': "degrees_east",'long_name': "longitude"})
        for v_name in vars:
            outVar = nc_d2sink_fseq.createVariable(v_name, 'f4', ('latitude','longitude'),zlib=True,
                                                   complevel=1)
        d2sink_var = nc_d2sink_fseq.variables['d2sink']
        d2sink_var.setncatts({'units': "km",'long_name': "distance to nearest sinking point"})
        fseq_var = nc_d2sink_fseq.variables['fseq_100years']
        fseq_var.setncatts({'units': "unitless (fraction)",'long_name': "fraction of sunk carbon remaining after 100 years (seaweed is sunk in same cell as grown)"})
        fseq_var2 = nc_d2sink_fseq.variables['fseq_100years_withtransport']
        fseq_var2.setncatts({'units': "unitless (fraction)",'long_name': "fraction of sunk carbon remaining after 100 years (seaweed grown in cell is transported to area with lowest cost per ton CO2 removed)"})
    
    elif years==200:
        vars=['d2sink','fseq_200years','fseq_200years_withtransport'] #,'d2sink_200y','fseq_200years','fseq_200years_withtransport']
        nc_d2sink_fseq = netCDF4.Dataset(outfile,'w')
        nc_d2sink_fseq.createDimension('latitude',len(latitude))
        nc_d2sink_fseq.createDimension('longitude',len(longitude))
        latvar = nc_d2sink_fseq.createVariable('latitude','f8',('latitude'))
        latvar.setncatts({'units': "degrees_north",'long_name': "latitude"})
        lonvar = nc_d2sink_fseq.createVariable('longitude','f8',('longitude'))
        lonvar.setncatts({'units': "degrees_east",'long_name': "longitude"})
        for v_name in vars:
            outVar = nc_d2sink_fseq.createVariable(v_name, 'f4', ('latitude','longitude'),zlib=True,
                                                   complevel=1)
        d2sink_var = nc_d2sink_fseq.variables['d2sink']
        d2sink_var.setncatts({'units': "km",'long_name': "distance to nearest sinking point"})
        fseq_var = nc_d2sink_fseq.variables['fseq_200years']
        fseq_var.setncatts({'units': "unitless (fraction)",'long_name': "fraction of sunk carbon remaining after 200 years (seaweed is sunk in same cell as grown)"})
        fseq_var2 = nc_d2sink_fseq.variables['fseq_200years_withtransport']
        fseq_var2.setncatts({'units': "unitless (fraction)",'long_name': "fraction of sunk carbon remaining after 200 years (seaweed grown in cell is transported to area with lowest cost per ton CO2 removed)"})
    
    latvar[...] = latitude
    lonvar[...] = longitude
    d2sink_var[...] = d2sink_optimizedtransport
    fseq_var[...] = fseq_original
    fseq_var2[...] = fseq_optimizedtransport
    
    nc_d2sink_fseq.sync()
    nc_d2sink_fseq.close()
    
    return sinkhere_or_sinkthere,nc_d2sink_fseq,fseq_original,fseq_optimizedtransport,sink,d2sink_optimizedtransport,netcostperton_sink_notransport,netcostperton_sink_withtransport,mask

def TEA_cost_test(seaweed_c,outfile,TEA_params=default_p_bounds,linedensity=1.e6,n_runs=100,offshore=18.52,
             chunksizes=(24,4320,1)):
    """An example of a monte carlo type analysis, where the randomized results arrays are written to
     a NetCDF file, with specific (compressed) chunksizes. Knowing/setting the chunksizes allows
     efficient reads and computations later on a pixel-by-pixel basis."""

    # load costing data sources
    longitude,latitude = load_cwm_lon_lat()
    d2port = ncread(d2pfile,'d2p')
    #depth = ncread(depthfile,'elevation')

    # pre-calc data manipulation
    distmult = d2port > offshore
    distmult = distmult.astype(np.int32) + 1  # this would work to: distmult + 1

    # open giant output file
    nc = open_output_nc(outfile,latitude,longitude,vars=['cost'],chunksizes=chunksizes)
    cost_var = nc.variables['cost']
    mc_var = nc.variables['mc']
    prec = None
    for i in range(n_runs):  # varies 0 : n_runs-1
        print('MC calc %i'%i)
        p = random_params(TEA_params)
        totlinecost = p['linecost'] * linedensity # $/km^2
        cost = (((p['capex'] * distmult) + (p['opex']) + (p['insur']) + (p['license']) + (p['labor']) + (totlinecost) + (d2port * p['transportcost'] * seaweed_c) + (p['harvcost'])) / seaweed_c)
        #cost[~np.isfinite()] = np.nan # is this needed? infinites are weird
        mc_var[i] = i  # save index variable to the monte carlo var - ensure compatibility with other programs
        cost_var[...,i] = cost # save to netcdf
        nc.sync() # maybe this will prevent random "HDF error"s?
        # save random params in case we want to examine sensitivity
        # can't just use p because values must be in lists...se we create a new dict with lists of 1 value
        plist = {k:[v,] for k,v, in p.items()}
        if prec is None:
            prec = pd.DataFrame(plist)
        else:
            prec.append(plist,ignore_index=True)

    nc.close()
    outpath,_ = os.path.splitext(outfile)
    prec.to_csv(outpath+'_params.csv')

def TEA_cost_value_net(seaweed,outfile,seq_frac,d2sink,linedensity,n_runs,chunksizes,TEA_params=default_p_bounds,
             seaweed_map_range=p_bounds_seaweedmap,paramfile = None):
    """An example of a monte carlo type analysis, where the randomized results arrays are written to
     a NetCDF file, with specific (compressed) chunksizes. Knowing/setting the chunksizes allows
     efficient reads and computations later on a pixel-by-pixel basis."""

    # load costing data sources
    longitude,latitude,xlon,ylat,lonb,latb,xlonb,ylatb = load_cwm_lon_lat_mesh_bounds()
    d2port = ncread(d2pfile,'d2p')
    depth = ncread(depthfile,'elevation') * (-1.0)
    waves = ncread(wavefile,'wave height (mean)')
    waves = np.transpose(waves)
    area = get_area(default_cwm_area)
    seaweed_file = seaweed

    # nharv = ncread(seaweed_file,'nharv_H')
    
    maintenance_distance = 50 # km/km2, travel around each km2 for maintenance
    maintenance_area_pertrip = 0.5 # km2
    maintenance_trips = 6 # amount of times each km2 is visited for maintenance per year
    
    d_cost = depth / 500 # 500m is the threshold above which costs increase as a function of depth because anchorage design changes - see Yu et al., 2020
    d_cost[d_cost < 1] = 0
    # d_cost = 1
    w_cost = waves / 3 # 3m is the global average significant wave height, so we assume that above 3m waves will impact the lifetime of capital 
    w_cost[w_cost < 1] = 0
    
    carbon_fraction = 0.3 # tons C/ton DW
    carbon_to_co2 = 3.67 # tons CO2/ton C
    
    # open giant output files
    nc = open_output_nc(outfile,latitude,longitude,vars=['prod_cost','prod_cost_low10p_avg','prod_ems',
        'value_sink','removed_ems_sink','value_product','avoided_ems_product','net_cost_sink','net_ems_sink',
        'net_costperton_sink','net_costperton_sink_low10p_avg','net_cost_product','net_ems_product',
        'net_costperton_product','net_costperton_product_low10p_avg','sink_or_ship','seaweed_dw',
        'seaweed_ww','species','nharv','d2port','d2sink','fseq_100y_withtransport','depth','depth_scaling',
        'significant wave height','wave height scaling','linedensity','linespacing',
        'equipment_mass'],chunksizes=chunksizes)
    cost_var = nc.variables['prod_cost']
    cost_var_low1p = nc.variables['prod_cost_low10p_avg']
    ems_var = nc.variables['prod_ems']
    val_var_sink = nc.variables['value_sink']
    ems_var_sink = nc.variables['removed_ems_sink']
    val_var_product = nc.variables['value_product']
    ems_var_product = nc.variables['avoided_ems_product']
    netcost_var_sink = nc.variables['net_cost_sink']
    netems_var_sink = nc.variables['net_ems_sink']
    netcostperton_var_sink = nc.variables['net_costperton_sink']
    netcostperton_var_sink_low1p = nc.variables['net_costperton_sink_low10p_avg']
    netcost_var_product = nc.variables['net_cost_product']
    netems_var_product = nc.variables['net_ems_product']
    netcostperton_var_product = nc.variables['net_costperton_product']
    netcostperton_var_product_low1p = nc.variables['net_costperton_product_low10p_avg']
    sink_or_ship_var = nc.variables['sink_or_ship']
    mc_var = nc.variables['mc']
    prec = None
    
    # defining regions for metadata tracking/random forest analysis
    north_atlantic_latlon = [-75,20,0,75] # left, right, bottom, top
    south_atlantic_latlon = [-70,20,-60,0]
    north_pacific_latlon = [100,-100,20,70]
    central_pacific_latlon = [130,-90,-20,20]
    south_pacific_latlon = [140,-70,-60,0]
    indian_latlon = [20,100,-60,30]
    
    i = 0
    difference = 1
    mean = 0
    mean_previous = 0

    for i in range(n_runs):  # varies 0 : n_runs-1

        print('MC calc %i'%(i+1))
        p = random_params(TEA_params)
        m = normal_params(seaweed_map_range)
        #### Change "ambeint" to "flux" in these file names if using limited nutrient data
        if m['seaweed_map'] <=32.7:
            seaweed = ncread(seaweed_file,'Harvest_q5')
            nharv = ncread(seaweed_file,'nharv_Hq5')
            species = ncread(seaweed_file,'index_Hq5')
            d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q5.nc','d2sink')
            seq_frac = ncread(r'/filepath/d2sink_fseq_maps_ambient_q5.nc','fseq_100years_withtransport')
        elif 32.7 < m['seaweed_map'] <=44.7:
            seaweed = ncread(seaweed_file,'Harvest_q25')
            nharv = ncread(seaweed_file,'nharv_Hq25')
            species = ncread(seaweed_file,'index_Hq25') 
            d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q25.nc','d2sink')
            seq_frac = ncread(r'/filepath/d2sink_fseq_maps_ambient_q25.nc','fseq_100years_withtransport')
        elif 44.7 < m['seaweed_map'] <=55.3:
            seaweed = ncread(seaweed_file,'Harvest_q50')
            nharv = ncread(seaweed_file,'nharv_Hq50')
            species = ncread(seaweed_file,'index_Hq50')
            d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','d2sink')
            seq_frac = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','fseq_100years_withtransport')
        elif 55.3 < m['seaweed_map'] <=67.3:
            seaweed = ncread(seaweed_file,'Harvest_q75')
            nharv = ncread(seaweed_file,'nharv_Hq75')
            species = ncread(seaweed_file,'index_Hq75') 
            d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q75.nc','d2sink')
            seq_frac = ncread(r'/filepath/d2sink_fseq_ambient_flux_q75.nc','fseq_100years_withtransport')
        elif 67.3 < m['seaweed_map'] <=100:
            seaweed = ncread(seaweed_file,'Harvest_q95')
            nharv = ncread(seaweed_file,'nharv_Hq95')
            species = ncread(seaweed_file,'index_Hq95') 
            d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q95.nc','d2sink')
            seq_frac = ncread(r'/filepath/d2sink_fseq_maps_ambient_q95.nc','fseq_100years_withtransport')
    
        equipment = copy.deepcopy(species)
        equipment[equipment == 0] = 1231.87 # eucheuma line spacing 0.2 m
        equipment[equipment == 1] = 185.24 # sargassum line spacing 1.33 m
        equipment[equipment == 2] = 4927.50 # porphyra line spacing net 2mx100m, 0.1m grid (Wu et al, marine pollution bulletin, 2015), total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
        equipment[equipment == 3] = 164.25 # saccharina (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m
        equipment[equipment == 4] = 164.25 # macrocystis (kelp) mass of equipment, tons/km2 (lines, ropes, anchors, buoys, etc) per km2, line spacing 1.5 m

        linedensity = copy.deepcopy(species)
        linedensity[linedensity == 0] = 5000000 # m line per km2, eucheuma line spacing 0.25 m
        linedensity[linedensity == 1] = 751880 # sargassum line spacing 1.33 m
        linedensity[linedensity == 2] = 20000000 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
        linedensity[linedensity == 3] = 666667 # saccharina (kelp) line spacing 1.5 m
        linedensity[linedensity == 4] = 666667 # macrocystis (kelp) line spacing 1.5 m
        
        seaweed[seaweed==0] = np.nan
        seaweed_ww = seaweed / 0.1
        
        #### cost calcs
        totlinecost = p['linecost'] * linedensity # $/km^2
        w_cost[w_cost >= 1] = p['wave_mult'] 
        d_cost[d_cost >= 1] = p['depth_mult'] 
        cost = (((p['capex'] + (p['capex'] * d_cost) + (p['capex'] * w_cost) + (totlinecost)) + ((p['opex']) + (p['insur']) + (p['license']) + (p['labor'])) + ((p['harvcost'] * nharv) + (p['transportcost'] * equipment * d2port))) / seaweed)
        cost_copy = copy.deepcopy(cost)
        # lowpoint5p_prodcost_avg,lowpoint5p_prodcost_avg_map = area_weighted_avg(cost_copy,seaweed,area,percentile=0.5)
        low1p_prodcost_avg,low1p_prodcost_avg_map = area_weighted_avg(cost_copy,seaweed,area,percentile=1)
        percentile_point4_prodcost_avg,percentile_point8_prodcost_avg,percentile_1p2_prodcost_avg,percentile_1p6_prodcost_avg,\
            percentile_2_prodcost_avg = percentile_binned_avg(cost_copy,seaweed,area,bin_size=0.4,bin_start=0)
        # print(low1p_prodcost_avg)
        region1_prodcost_mean, region1_area = get_regional_mean(cost_copy,north_atlantic_latlon,area,xlon,ylat)
        region2_prodcost_mean, region2_area = get_regional_mean(cost_copy,south_atlantic_latlon,area,xlon,ylat)
        region3_prodcost_mean, region3_area = get_regional_mean(cost_copy,north_pacific_latlon,area,xlon,ylat)
        region4_prodcost_mean, region4_area = get_regional_mean(cost_copy,central_pacific_latlon,area,xlon,ylat)
        region5_prodcost_mean, region5_area = get_regional_mean(cost_copy,south_pacific_latlon,area,xlon,ylat)
        region6_prodcost_mean, region6_area = get_regional_mean(cost_copy,indian_latlon,area,xlon,ylat)
        ems = ((((equipment * d2port * p['transportems'])) / seaweed) + ((((d2port * 2) * p['maintenanceboatems'] * maintenance_trips / maintenance_area_pertrip) + (maintenance_distance * p['maintenanceboatems'])) / seaweed))
        mc_var[i] = i  # save index variable to the monte carlo var - ensure compatibility with other programs
        cost_var[...,i] = cost # save to netcdf
        cost_var_low1p[...,i] = low1p_prodcost_avg_map
        ems_var[...,i] = ems # save to netcdf
        
        #### value calcs
        removed_ems_sink = (p['removal_rate'] * p['sequestration_rate'] * carbon_fraction * carbon_to_co2 * seq_frac)
        val_sink = (p['sinkval'] - (((p['transportcost'] * d2sink * seaweed_ww) + (p['transportcost'] * 2 * d2sink * equipment) + (p['transportcost'] * d2port * equipment)) / seaweed))
        ems_sink = (removed_ems_sink - (((p['transportems'] * d2sink * seaweed_ww) + (p['transportems'] * 2 * d2sink * equipment) + (p['transportems'] * d2port * equipment)) / seaweed))
        val_product = (p['productval'] - (((p['transportcost'] * d2port * (seaweed_ww + equipment)) / seaweed) + p['convertcost']))
        ems_product = (p['avoidedems_product'] - (((p['transportems'] * d2port * (seaweed_ww + equipment)) / seaweed) + p['convertems']))
        val_var_sink[...,i] = val_sink # save to netcdf
        ems_var_sink[...,i] = ems_sink # save to netcdf
        val_var_product[...,i] = val_product # save to netcdf
        ems_var_product[...,i] = ems_product # save to netcdf
        
        #### net calcs
        # SINK
        costnet_sink = (cost - val_sink)
        emsnet_sink = (ems_sink - ems)
        emsnet_sink[emsnet_sink <= 0] = np.nan
        netcostperton_sink = costnet_sink / emsnet_sink
        cost_sink_copy = copy.deepcopy(netcostperton_sink)
        # lowpoint5p_sinkcost_avg,lowpoint5p_sinkcost_avg_map = area_weighted_avg(cost_sink_copy,seaweed,area,percentile=0.5)
        low1p_sinkcost_avg,low1p_sinkcost_avg_map = area_weighted_avg(cost_sink_copy,seaweed,area,percentile=1)
        percentile_point4_sinkcost_avg,percentile_point8_sinkcost_avg,percentile_1p2_sinkcost_avg,percentile_1p6_sinkcost_avg,\
            percentile_2_sinkcost_avg = percentile_binned_avg(cost_sink_copy,seaweed,area,bin_size=0.4,bin_start=0)
        region1_sinkcost_mean, region1_area = get_regional_mean(cost_sink_copy,north_atlantic_latlon,area,xlon,ylat)
        region2_sinkcost_mean, region2_area = get_regional_mean(cost_sink_copy,south_atlantic_latlon,area,xlon,ylat)
        region3_sinkcost_mean, region3_area = get_regional_mean(cost_sink_copy,north_pacific_latlon,area,xlon,ylat)
        region4_sinkcost_mean, region4_area = get_regional_mean(cost_sink_copy,central_pacific_latlon,area,xlon,ylat)
        region5_sinkcost_mean, region5_area = get_regional_mean(cost_sink_copy,south_pacific_latlon,area,xlon,ylat)
        region6_sinkcost_mean, region6_area = get_regional_mean(cost_sink_copy,indian_latlon,area,xlon,ylat)
        # netcostperton_sink[netcostperton_sink<0] = np.nan
        netcost_var_sink[...,i] = costnet_sink # save to netcdf
        netems_var_sink[...,i] = emsnet_sink # save to netcdf
        netcostperton_var_sink[...,i] = netcostperton_sink # save to netcdf
        netcostperton_var_sink_low1p[...,i] = low1p_sinkcost_avg_map
        # PRODUCT
        costnet_product = (cost - val_product)
        emsnet_product = (ems_product - ems)
        emsnet_product[emsnet_product <= 0] = np.nan
        netcostperton_product = costnet_product / emsnet_product
        cost_product_copy = copy.deepcopy(netcostperton_product)
        # lowpoint5p_productcost_avg,lowpoint5p_productcost_avg_map = area_weighted_avg(cost_product_copy,seaweed,area,percentile=0.5)
        low1p_productcost_avg,low1p_productcost_avg_map = area_weighted_avg(cost_product_copy,seaweed,area,percentile=1)
        percentile_point4_productcost_avg,percentile_point8_productcost_avg,percentile_1p2_productcost_avg,percentile_1p6_productcost_avg,\
            percentile_2_productcost_avg = percentile_binned_avg(cost_product_copy,seaweed,area,bin_size=0.4,bin_start=0)
        region1_productcost_mean, region1_area = get_regional_mean(cost_product_copy,north_atlantic_latlon,area,xlon,ylat)
        region2_productcost_mean, region2_area = get_regional_mean(cost_product_copy,south_atlantic_latlon,area,xlon,ylat)
        region3_productcost_mean, region3_area = get_regional_mean(cost_product_copy,north_pacific_latlon,area,xlon,ylat)
        region4_productcost_mean, region4_area = get_regional_mean(cost_product_copy,central_pacific_latlon,area,xlon,ylat)
        region5_productcost_mean, region5_area = get_regional_mean(cost_product_copy,south_pacific_latlon,area,xlon,ylat)
        region6_productcost_mean, region6_area = get_regional_mean(cost_product_copy,indian_latlon,area,xlon,ylat)
        # netcostperton_product[netcostperton_product<0] = np.nan
        netcost_var_product[...,i] = costnet_product # save to netcdf
        netems_var_product[...,i] = emsnet_product # save to netcdf
        netcostperton_var_product[...,i] = netcostperton_product # save to netcdf
        netcostperton_var_product_low1p[...,i] = low1p_productcost_avg_map

        
        #compare sinking and products for each iteration to come up with preferred outcome map (1=sinking, 2=products)
        sink_or_ship = netcostperton_sink / netcostperton_product
        sink_or_ship[sink_or_ship <= 1] = 1
        sink_or_ship[sink_or_ship > 1] = 2
        sink_or_ship_var[...,i] = sink_or_ship
        # sink_or_ship 1 = sinking preferred, 2 = products preferred
        
        nc.sync() # maybe this will prevent random "HDF error"s?
        
        # save random params in case we want to examine sensitivity
        # can't just use p because values must be in lists...se we create a new dict with lists of 1 value
        #plist = {k:[v,] for k,v, in p.items()}
        # p['prod_cost_lowpoint5p_avg'] = lowpoint5p_prodcost_avg
        p['prod_cost_low1p_avg'] = low1p_prodcost_avg
        p['percentile_0.4_prodcost_avg'] = percentile_point4_prodcost_avg
        p['percentile_0.8_prodcost_avg'] = percentile_point8_prodcost_avg
        p['percentile_1.2_prodcost_avg'] = percentile_1p2_prodcost_avg
        p['percentile_1.6_prodcost_avg'] = percentile_1p6_prodcost_avg
        p['percentile_2_prodcost_avg'] = percentile_2_prodcost_avg
        p['prod_cost_avg_northatlantic'] = region1_prodcost_mean
        p['prod_cost_avg_southatlantic'] = region2_prodcost_mean
        p['prod_cost_avg_northpacific'] = region3_prodcost_mean
        p['prod_cost_avg_centralpacific'] = region4_prodcost_mean
        p['prod_cost_avg_southpacific'] = region5_prodcost_mean
        p['prod_cost_avg_indian'] = region6_prodcost_mean
        # p['net_costperton_sink_lowpoint5p_avg'] = lowpoint5p_sinkcost_avg
        p['net_costperton_sink_low1p_avg'] = low1p_sinkcost_avg
        p['percentile_0.4_sinkcost_avg'] = percentile_point4_sinkcost_avg
        p['percentile_0.8_sinkcost_avg'] = percentile_point8_sinkcost_avg
        p['percentile_1.2_sinkcost_avg'] = percentile_1p2_sinkcost_avg
        p['percentile_1.6_sinkcost_avg'] = percentile_1p6_sinkcost_avg
        p['percentile_2_sinkcost_avg'] = percentile_2_sinkcost_avg
        p['net_costperton_sink_avg_northatlantic'] = region1_sinkcost_mean
        p['net_costperton_sink_avg_southatlantic'] = region2_sinkcost_mean
        p['net_costperton_sink_avg_northpacific'] = region3_sinkcost_mean
        p['net_costperton_sink_avg_centralpacific'] = region4_sinkcost_mean
        p['net_costperton_sink_avg_southpacific'] = region5_sinkcost_mean
        p['net_costperton_sink_avg_indian'] = region6_sinkcost_mean
        # p['net_costperton_product_lowpoint5p_avg'] = lowpoint5p_productcost_avg
        p['net_costperton_product_low1p_avg'] = low1p_productcost_avg
        p['percentile_0.4_productcost_avg'] = percentile_point4_productcost_avg
        p['percentile_0.8_productcost_avg'] = percentile_point8_productcost_avg
        p['percentile_1.2_productcost_avg'] = percentile_1p2_productcost_avg
        p['percentile_1.6_productcost_avg'] = percentile_1p6_productcost_avg
        p['percentile_2_productcost_avg'] = percentile_2_productcost_avg
        p['net_costperton_product_avg_northatlantic'] = region1_productcost_mean
        p['net_costperton_product_avg_southatlantic'] = region2_productcost_mean
        p['net_costperton_product_avg_northpacific'] = region3_productcost_mean
        p['net_costperton_product_avg_centralpacific'] = region4_productcost_mean
        p['net_costperton_product_avg_southpacific'] = region5_productcost_mean
        p['net_costperton_product_avg_indian'] = region6_productcost_mean

        #
        p['seaweed_map'] = m['seaweed_map']
        if prec is None:
            prec = [p] #pd.DataFrame(plist)
            # prec.append(m)
        else:
            prec.append(p)
            # prec.append(m)
        i += 1
    nc.close()
    outpath,_ = os.path.splitext(outfile)
    prec = pd.DataFrame(prec)
    prec.to_csv(outpath+'_params.csv')
    
def TEA_analysis(TEA_file,TEA_var,out_nc_file,chunksizes):
    """This function reads the montecarlo analysis NetCDF file using the
    chunksize with which it was written. Because the minimum read size is one
    chunk, we can be efficient by reading and processing all data within a series of
    chunks that are written (and compressed) on disk.

    The standard numpy function 'nanquantile" was unbearably slow, so we use a custom
    routine, compiled using numba, that is faster.

    Saves resulting stats/analysis arrays to another netcdf file, and returns the dictionary
    of stats/analysis arrays

    The idea is that one can use this on different montecarlo results for cost, energy, value,
    carbon, etc.
    """

    metrics = ['tea_median','tea_mean','tea_min','tea_max','tea_q5','tea_q25','tea_q75','tea_q95']

    # find bounds and setup array for fastest reads across chunks
    span = chunksizes[0]

    # open mc netcdf file, and find dimensions
    hdf = h5py.File(TEA_file)
    dset = hdf[TEA_var]
    latdimsize = dset.shape[0]
    nruns = dset.shape[2]
    n_reads = int(np.ceil(latdimsize/span))  # number of sequential reads we need to do for per-pixel analysis
    read_buffer = np.full((span,4320,nruns),np.nan,'f4') # an array to quickly load netcdf data reads

    # result arrays
    r = {m : np.full((2160,4320),np.nan,'f4') for m in metrics}

    for i in range(n_reads):
        print('reading runs slice %i of %i...'%(i+1,n_reads))
        ls = i*span
        le = min(ls+span,2159)
        nl = le-ls
        ncslice = np.s_[ls:le,...]
        if i == n_reads-1:
            # last read is a different shape, use slower method
            read_buffer[0:nl,...] = dset[ncslice]
        else:
            dset.read_direct(read_buffer, source_sel=ncslice)

        print('analysis of runs slice %i of %i...'%(i+1,n_reads))

        # 1st try - somewhat fast
        # r['tea_median'][ls:le,...] = np.nanmedian(read_buffer[0:nl,...],axis=2)
        # r['tea_mean'][ls:le,...] = np.nanmean(read_buffer[0:nl,...],axis=2)
        # r['tea_min'][ls:le,...] = np.nanmin(read_buffer[0:nl,...],axis=2)
        # r['tea_max'][ls:le,...] = np.nanmax(read_buffer[0:nl,...],axis=2)
        # r['tea_q5'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.05)
        # r['tea_q25'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.25)
        # r['tea_q75'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.75)
        # r['tea_q95'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.95)

        # 2nd try - parallel and lots faster
        #r['tea_q5'][ls:le,...],r['tea_q25'][ls:le,...],r['tea_q75'][ls:le,...],r['tea_q95'][ls:le,...],\
        #r['tea_median'][ls:le,...],r['tea_mean'][ls:le,...],r['tea_min'][ls:le,...],\
        #r['tea_max'][ls:le,...] = custom_stats(read_buffer[0:nl,...])

        # 3rd try - faster still with less memory less single-core memory management
        #### V2 vs V3 stats switch
        custom_stats_v3(read_buffer[0:nl,...],r['tea_q5'][ls:le,...],r['tea_q25'][ls:le,...],
                        r['tea_q75'][ls:le,...],r['tea_q95'][ls:le,...],r['tea_median'][ls:le,...],
                        r['tea_mean'][ls:le,...],r['tea_min'][ls:le,...],r['tea_max'][ls:le,...])

    hdf.close()
    units = ['$/tonne' for i in range(len(metrics))] # all units the same for now


    # perhaps perform more/add to the analysis in the r dictionary ?
    #

    save_TEA_analysis(out_nc_file,metrics,r,units)

    return r

def TEA_analysis_products_fig2(TEA_file,TEA_var,meta_file,avoidedems_lims,productval_lims,out_nc_file,chunksizes):
    """This function reads the montecarlo analysis NetCDF file using the
    chunksize with which it was written. Because the minimum read size is one
    chunk, we can be efficient by reading and processing all data within a series of
    chunks that are written (and compressed) on disk.

    The standard numpy function 'nanquantile" was unbearably slow, so we use a custom
    routine, compiled using numba, that is faster.

    Saves resulting stats/analysis arrays to another netcdf file, and returns the dictionary
    of stats/analysis arrays

    The idea is that one can use this on different montecarlo results for cost, energy, value,
    carbon, etc.
    """

    metrics = ['tea_median','tea_mean','tea_min','tea_max','tea_q5','tea_q25','tea_q75','tea_q95']

    # find bounds and setup array for fastest reads across chunks
    span = chunksizes[0]

    # open mc netcdf file, and find dimensions
    hdf = h5py.File(TEA_file)
    dset = hdf[TEA_var]
    latdimsize = dset.shape[0]
    meta = pd.read_csv(meta_file)
    mc_range = meta[(meta['avoidedems_product'] >= avoidedems_lims[0]) & (meta['avoidedems_product'] <= avoidedems_lims[1]) & (meta['productval'] >= productval_lims[0]) & (meta['productval'] <= productval_lims[1])]
    print(mc_range)
    mc_range = mc_range['Unnamed: 0'].tolist()
    nruns = len(mc_range) # dset.shape[2]
    n_reads = int(np.ceil(latdimsize/span))  # number of sequential reads we need to do for per-pixel analysis
    read_buffer = np.full((span,4320,nruns),np.nan,'f4') # an array to quickly load netcdf data reads

    # result arrays
    r = {m : np.full((2160,4320),np.nan,'f4') for m in metrics}

    for i in range(n_reads):
        print('reading runs slice %i of %i...'%(i+1,n_reads))
        ls = i*span
        le = min(ls+span,2159)
        nl = le-ls
        ncslice = np.s_[ls:le,...]
        temp_array = dset[ncslice]
        read_buffer[0:nl,:,0:nruns] = temp_array[...,mc_range]

        print('analysis of runs slice %i of %i...'%(i+1,n_reads))

        # 1st try - somewhat fast
        # r['tea_median'][ls:le,...] = np.nanmedian(read_buffer[0:nl,...],axis=2)
        # r['tea_mean'][ls:le,...] = np.nanmean(read_buffer[0:nl,...],axis=2)
        # r['tea_min'][ls:le,...] = np.nanmin(read_buffer[0:nl,...],axis=2)
        # r['tea_max'][ls:le,...] = np.nanmax(read_buffer[0:nl,...],axis=2)
        # r['tea_q5'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.05)
        # r['tea_q25'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.25)
        # r['tea_q75'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.75)
        # r['tea_q95'][ls:le,...] = custom_quantile(read_buffer[0:nl,...],0.95)

        # 2nd try - parallel and lots faster
        #r['tea_q5'][ls:le,...],r['tea_q25'][ls:le,...],r['tea_q75'][ls:le,...],r['tea_q95'][ls:le,...],\
        #r['tea_median'][ls:le,...],r['tea_mean'][ls:le,...],r['tea_min'][ls:le,...],\
        #r['tea_max'][ls:le,...] = custom_stats(read_buffer[0:nl,...])

        # 3rd try - faster still with less memory less single-core memory management
        #### V2 vs V3 stats switch
        custom_stats_v3(read_buffer[0:nl,...],r['tea_q5'][ls:le,...],r['tea_q25'][ls:le,...],
                        r['tea_q75'][ls:le,...],r['tea_q95'][ls:le,...],r['tea_median'][ls:le,...],
                        r['tea_mean'][ls:le,...],r['tea_min'][ls:le,...],r['tea_max'][ls:le,...])

    hdf.close()
    units = ['$/tonne' for i in range(len(metrics))] # all units the same for now


    # perhaps perform more/add to the analysis in the r dictionary ?
    #

    save_TEA_analysis(out_nc_file,metrics,r,units)

    return r

def percent_CDF(percent,percent_step,percent_step_100p,subchunks,totalbars,net_cost,carbon_removed,netcostperton_area_mask,seaweed_biomass,ships,mpas):
    import copy
    carbon_fraction = 0.3
    carbon_to_co2 = 3.67
    longitude,latitude,xlon,ylat,mask = load_cwm_grid_mask()
    mask = mask.astype(float)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = np.nan
    landmask = mask
    mpamask = mpas
    shippingmask = ships
    area = get_area(default_cwm_area)
    ocean_area = 361900000 #km2
    total_seaweed_area = (seaweed_biomass * area) / seaweed_biomass
    total_seaweed_area = np.nansum(total_seaweed_area)
    seaweed_bio = copy.deepcopy(seaweed_biomass)
    percent_blocks = round(percent/percent_step) # getting 0.1% chunks, so need 10x more runs than 1% chunks

    avg_percent_cost = [np.nan] * percent_blocks # pixel_blocks
    sum_carbonremoved = [np.nan] * percent_blocks # pixel_blocks
    area_farmed_km2 = [np.nan] * percent_blocks
    farmed_area_notblocked_km2 = [np.nan] * percent_blocks
    percent_area_blocked = [np.nan] * percent_blocks
    percent_oceanarea_farmed = [np.nan] * percent_blocks
    percent_seaweedarea_farmed = [np.nan] * percent_blocks
    
    print(avg_percent_cost)
    print(avg_percent_cost[0])

    for i in range(0,percent_blocks):
        weights = area / np.min(area)
        k = round(((np.count_nonzero(~np.isnan(netcostperton_area_mask))) / (100/(percent_step))) * (i+1))
        net_cost1 = copy.deepcopy(net_cost) # 
        net_cost_flat = net_cost1.flatten() #
        net_cost_flat = net_cost_flat[~np.isnan(net_cost_flat)] #
        net_cost_flat = np.sort(net_cost_flat)#
        net_cost_sorted = copy.deepcopy(net_cost_flat)
        target_cost = net_cost_sorted[k]
        net_cost2 = copy.deepcopy(net_cost1)
        net_cost2 = net_cost2 * landmask
        net_cost2[net_cost2 > target_cost] = np.nan
        weights[np.isnan(net_cost2)] = 0;
        net_cost2[np.isnan(net_cost2)] = 0;
        percentile_area_average = np.average(net_cost2, weights = weights)
        avg_percent_cost[i] = percentile_area_average
        print(k)
        print(i)
        print(avg_percent_cost[i])
        target_cost = net_cost_sorted[k]

        low_cost_map = copy.deepcopy(net_cost)
        low_cost_map[low_cost_map > target_cost] = np.nan
        low_cost_area = copy.deepcopy(low_cost_map)
        low_cost_area[~np.isnan(low_cost_area)] = 1
        
        carbon_removed1 = carbon_removed * low_cost_area * seaweed_bio * area # tonsCO2/tonDW * tonsDW/km2 * km2/gridcell
        carbon_removed_map = copy.deepcopy(carbon_removed1)
        carbon_removed_flat = carbon_removed1.flatten()
        carbon_removed_flat = carbon_removed_flat[~np.isnan(carbon_removed_flat)]
        sum_carbonremoved[i] = round(sum(carbon_removed_flat))
        seaweed_ref = carbon_removed1 / carbon_removed / area
        print(np.nanmean(seaweed_ref))
        print(sum_carbonremoved[i])
        
        farmed_area1 = low_cost_area * area * landmask # tonsCO2/tonDW * tonsDW/km2 * km2/gridcell
        farmed_area_map = copy.deepcopy(farmed_area1)
        farmed_area_flat = farmed_area1.flatten()
        farmed_area_flat = farmed_area_flat[~np.isnan(farmed_area_flat)]
        percent_oceanarea_farmed[i] = (((sum(farmed_area_flat))/ocean_area)*100)
        area_farmed_km2[i] = (sum(farmed_area_flat))
        print(percent_oceanarea_farmed[i])
        
        seaweed_area1 = (low_cost_area * area * seaweed_biomass) / seaweed_biomass # tonsCO2/tonDW * tonsDW/km2 * km2/gridcell
        # seaweed_area_map = copy.deepcopy(seaweed_area1)
        seaweed_area_flat = seaweed_area1.flatten()
        seaweed_area_flat = seaweed_area_flat[~np.isnan(seaweed_area_flat)]
        percent_seaweedarea_farmed[i] = (((sum(seaweed_area_flat))/total_seaweed_area)*100)
        print(percent_seaweedarea_farmed[i])
        
        area_notblocked = ((low_cost_area * area * seaweed_biomass) / seaweed_biomass) * ships * mpas
        area_notblocked_map = copy.deepcopy(area_notblocked)
        area_notblocked_flat = area_notblocked_map.flatten()
        area_notblocked_flat = area_notblocked_flat[~np.isnan(area_notblocked_flat)]
        farmed_area_notblocked_km2[i] = (sum(area_notblocked_flat))
        percent_area_blocked[i] = ((area_farmed_km2[i] - farmed_area_notblocked_km2[i]) / area_farmed_km2[i])*100
        print(percent_area_blocked[i])
        
    return avg_percent_cost, low_cost_map, sum_carbonremoved, carbon_removed_map, area_farmed_km2, farmed_area_notblocked_km2, percent_area_blocked, percent_oceanarea_farmed, farmed_area_map, percent_seaweedarea_farmed
 

# use this to safely use methods from this file
if __name__ =='__main__':
    import pylab
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    # import proplot as pplt
    run = 1;
    if run==1:
        import copy
        carbon_fraction = 0.3
        carbon_to_co2 = 3.67
        nruns=5000
        
        seaweed_file = seaweed_file_ambient
        
        seaweed_flux_harvest_DW_g_m2 = ncread(seaweed_file_flux,
                                         'Growth') # growth and harvest got switched in the flux file, so growth is actually harvest and vice versa
        # seaweed_ambient_harvest_DW_g_m2 = ncread(seaweed_file_ambient,
        #                                  'Harvest_q95')
        seaweed_ambient_harvest_DW_g_m2 = ncread(seaweed_file_ambient,
                                         'Harvest')
        seaweed_q50 = ncread(seaweed_file,
                             'Harvest_q50')
        seaweed_median = copy.deepcopy(seaweed_q50)
        seaweed_harvest_DW_g_m2 = seaweed_median[:]
        
        # seaweed_harvest_DW_g_m2[seaweed_harvest_DW_g_m2==0] = np.nan 
        seaweed_harvest_DW_tn_km2 = seaweed_harvest_DW_g_m2 * 1e6 / 1e6
        seaweed_harvest_C_tn_km2 = seaweed_harvest_DW_tn_km2 * carbon_fraction 
        seaweed_harvest_C02_tn_km2 = seaweed_harvest_C_tn_km2 * carbon_to_co2
    
        seaweed = copy.deepcopy(seaweed_harvest_DW_tn_km2)
        seaweed_q95 = ncread(seaweed_file,
                             'Harvest_q95')
        seaweed_q75 = ncread(seaweed_file,
                             'Harvest_q75')
        seaweed_q25 = ncread(seaweed_file,
                             'Harvest_q25')
        seaweed_q5 = ncread(seaweed_file,
                             'Harvest_q5')
        seaweed_mean = ncread(seaweed_file,
                             'Harvest_mean')
        # species = ncread(seaweed_file,'index_Hq95')
        species = ncread(seaweed_file,'index_H')
        # nharv = ncread(seaweed_file,'nharv_Hq95')
        nharv = ncread(seaweed_file,'nharv_H')
        harv_stdev = ncread(seaweed_file,'Harvest_std')
        # seaweed = seaweed_file
   
    run = 2;
    #### distance to sink function
    if run==1:
        sinkhere_or_sinkthere,ncfile_d2sink_fseq,fseq_original,fseq_withtransport,sink,d2sink,netcostperton_sink_notransport,netcostperton_sink_withtransport,mask = distance_to_sink(seaweed,species,nharv,r'/filepath/d2sink_fseq_maps_ambient_q95.nc',years=100)
    
    run = 1
    if run==1:
        fseq_withtransport = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','fseq_100years_withtransport')
        d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','d2sink')
        
        # add_seaweed_here,seaweed_change,seaweed_comp,seaweed_comp1,seaweed_comp2,seaweed_comp3 = seaweed_transport2sink(seaweed_harvest_C_tn_km2,sink,d2sink,sinkidx,shallow,d2shallow,shallowidx,mask)
        # seaweed_idx = np.indices((2160,4320))
        longitude,latitude,xlon,ylat,lonb,latb,xlonb,ylatb = load_cwm_lon_lat_mesh_bounds()
        area = get_area(default_cwm_area)
        ## seq_frac = scipy.io.loadmat(seqfracfile) ## NEED TO ASK BEN TO INTERPOLATE TO OUR GRID
        # seq_frac = ncread(seqfracfile,'fseq_bottom_500y')
        seq_frac = copy.deepcopy(fseq_withtransport)
        # seq_frac = 1;

    run = 1;
    if run==1:
        seaweed = copy.deepcopy(seaweed_harvest_DW_tn_km2)
        nharv = ncread(seaweed_file,'nharv_H')
        species = ncread(seaweed_file,'index_Hq50')

        linedensity = copy.deepcopy(species)
        linedensity[linedensity == 0] = 5000000 # m line per km2, eucheuma line spacing 0.2 m
        linedensity[linedensity == 1] = 751880 # sargassum line spacing 1.33 m
        linedensity[linedensity == 2] = 20000000 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
        linedensity[linedensity == 3] = 666667 # saccharina (kelp) line spacing 1.5 m
        linedensity[linedensity == 4] = 666667 # macrocystis (kelp) line spacing 1.5 m
        
        waves = ncread(wavefile,'wave height (mean)')
        waves = np.transpose(waves)
        
        chunks = (72,4320,1)
    
        species = ncread(seaweed_file,'index_Hq50')
        seedweight_filter = copy.deepcopy(seaweed)
        seaweed_ref = copy.deepcopy(seaweed)
        species_key = copy.deepcopy(species)
        seedweight_filter[species_key == 0] = 200 # m line per km2, eucheuma line spacing 0.2 m
        seedweight_filter[species_key == 1] = 50 # sargassum line spacing 1.33 m
        seedweight_filter[species_key == 2] = 10 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
        seedweight_filter[species_key == 3] = 50 # saccharina (kelp) line spacing 1.5 m
        seedweight_filter[species_key == 4] = 50 # macrocystis (kelp) line spacing 1.5 m
        
        seedweight_check = seaweed_ref / seedweight_filter
        seedweight_check[seedweight_check <=1] = np.nan
        seedweight_check[~np.isnan(seedweight_check)] = 1
        seedweight_check = seedweight_check * seaweed_median
        seaweed_aboveseedweight = copy.deepcopy(seedweight_check)
        
        harv_stdev[np.isnan(seaweed_aboveseedweight)] = np.nan
    
    run = 2;
    if run==1:
        #### Running Monte Carlo analysis
        TEA_cost_value_net(seaweed_file_flux,r'/filepath/flux_data.nc',seq_frac,d2sink,linedensity,nruns,chunksizes=chunks)
        
    run = 2;
    if run==1:  
        #### Doing product vs. sinking analysis from Monte Carlo runs (must complete Monte Carlo runs first, see line above)
        production_cost = TEA_analysis(r'/filepath/flux_data.nc','prod_cost',r'/filepath/flux_prodcost_analysis.nc',chunksizes=chunks) # (24,4320,1)
        net_emsremoved_sink = TEA_analysis(r'/filepath/flux_data.nc','net_ems_sink',r'/filepath/flux_netemssink_analysis.nc',chunksizes=chunks) # (24,4320,1)
        net_costperton_sink = TEA_analysis(r'/filepath/flux_data.nc','net_costperton_sink',r'/filepath/flux_netcostpertonsink_analysis.nc',chunksizes=chunks) # (24,4320,1)
        net_emsavoided_product = TEA_analysis(r'/filepath/flux_data.nc','net_ems_product',r'/filepath/flux_netemsproduct_analysis.nc',chunksizes=chunks) # (24,4320,1)
        net_costperton_product = TEA_analysis(r'/filepath/flux_data.nc','net_costperton_product',r'/filepath/flux_netcostpertonproduct_analysis.nc',chunksizes=chunks) # (24,4320,1)
        sink_or_ship = TEA_analysis(r'/filepath/flux_data.nc','sink_or_ship',r'/filepath/flux_sinkorship_analysis.nc',chunksizes=chunks) # (24,4320,1)
      
    run = 1;
    if run==1:
        
        hdf = h5py.File(r'/filepath/ambient_data.nc')
        hdf.close()
        
        sensitive_area_file = r'/filepath/sensitive_area_maps.nc'
        
        d2sink = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','d2sink')
        fseq_withtransport = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','fseq_100years_withtransport')
        seq_frac = copy.deepcopy(fseq_withtransport)
        fseq = copy.deepcopy(fseq_withtransport)
        fseq_original = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','fseq_100years')
        d2port = ncread(d2pfile,'d2p')
        mpas = ncread(r'/filepath/WDPA_CWM.nc','mpa_mask_min_20_sq_km')
        mpas = mpas.astype(float)
        mpas[mpas > 0] = np.nan
        mpas[~np.isnan(mpas)] = 1
        ships = ncread(r'/filepath/global_ship_density.nc','ship_density_cwm')
        ships = ships.astype(float)
        ships[ships > 225000000] = np.nan
        ships[~np.isnan(ships)] = 1
        
        vars=['marine_protected_areas','shipping_lanes'] #,'d2sink_200y','fseq_200years','fseq_200years_withtransport']
        sensitive_areas = netCDF4.Dataset(sensitive_area_file,'w')
        sensitive_areas.createDimension('latitude',len(latitude))
        sensitive_areas.createDimension('longitude',len(longitude))
        latvar = sensitive_areas.createVariable('latitude','f8',('latitude'))
        latvar.setncatts({'units': "degrees_north",'long_name': "latitude"})
        lonvar = sensitive_areas.createVariable('longitude','f8',('longitude'))
        lonvar.setncatts({'units': "degrees_east",'long_name': "longitude"})
        for v_name in vars:
            outVar = sensitive_areas.createVariable(v_name, 'f4', ('latitude','longitude'),zlib=True,
                                                       complevel=1)
        mpas_var = sensitive_areas.variables['marine_protected_areas']
        mpas_var.setncatts({'units': "mask",'long_name': "marine protected areas"})
        ships_var = sensitive_areas.variables['shipping_lanes']
        ships_var.setncatts({'units': "mask",'long_name': "shipping lanes"})
        
        latvar[...] = latitude
        lonvar[...] = longitude
        mpas_var[...] = mpas
        ships_var[...] = ships
        
        sensitive_areas.sync()
        sensitive_areas.close()
        
    run = 2;
    if run==1:
        var1 = ncread(r'/filepath/flux_prodcost_analysis.nc','tea_median')
        var2 = ncread(r'/filepath/flux_prodcost_analysis.nc','tea_min')
        var3 = ncread(r'/filepath/flux_prodcost_analysis.nc','tea_max')
        var4 = ncread(r'/filepath/flux_netcostpertonsink_analysis.nc','tea_median')
        # var4[fseq <= 0.01] = np.nan
        var5 = ncread(r'/filepath/flux_netcostpertonsink_analysis.nc','tea_min')
        # var5[fseq <= 0.01] = np.nan
        var6 = ncread(r'/filepath/flux_netcostpertonsink_analysis.nc','tea_max')
        # var6[fseq <= 0.01] = np.nan
        var7 = ncread(r'/filepath/ambient_sinkorship_analysis.nc','tea_median')
        var8 = ncread(r'/filepath/flux_netcostpertonproduct_analysis.nc','tea_min')
        var9 = ncread(r'/filepath/flux_netcostpertonproduct_analysis.nc','tea_max')
        
        # test_mpas = var7 * mpas
        
        print('\nmaking figure...')
        # my_cmap = sns.cubehelix_palette(start=.5, rot=-.75, light=.99, as_cmap=True)
   
    run = 1;
    if run==1:
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        
        cmap = sns.color_palette('viridis', as_cmap=True)
        new_cmap = truncate_colormap(cmap, 0, 0.75)
        my_cmap = new_cmap

    run = 1;
    if run==1:
        # fig,region1,region2,region3,region4 = cwm_map_fig1(var1,var2,var3,[0,3000],'flux production cost ($/tDW), 5000 runs, 07/21/22',area,seaweed,ylat,xlon,my_cmap)
        # print('all done.')
        
        nharv = ncread(seaweed_file,'nharv_H')
        d2sink_ambient = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','d2sink')
        fseq_withtransport_ambient = ncread(r'/filepath/d2sink_fseq_maps_ambient_q50.nc','fseq_100years_withtransport')
        d2sink_flux = ncread(r'/filepath/d2sink_fseq_maps_flux_q50.nc','d2sink')
        fseq_withtransport_flux = ncread(r'/filepath/d2sink_fseq_maps_flux_q50.nc','fseq_100years_withtransport')
        
        
        seaweed_biomass = copy.deepcopy(seaweed_harvest_DW_g_m2)
        seaweed_biomass[seaweed_biomass<=50] = np.nan
        
        species = ncread(seaweed_file_flux,'index_Hq50')
        species = species * (seaweed_aboveseedweight * area) / (seaweed_aboveseedweight * area)
        
        speciesmask = copy.deepcopy(species)
        speciesmask[speciesmask == 0] = 0 # m line per km2, eucheuma line spacing 0.25 m
        speciesmask[speciesmask == 1] = 0 # sargassum line spacing 1.33 m
        speciesmask[speciesmask == 2] = 0 # porphyra line spacing net, 0.1m grid, total amount of line equal to 0.05m longline spacing minus line between individual 2mx100m nets, plus some extra for knots
        speciesmask[speciesmask == 3] = 1 # saccharina (kelp) line spacing 1.5 m
        speciesmask[speciesmask == 4] = 1 # macrocystis (kelp) line spacing 1.5 m
        
        seaweed = ncread(seaweed_file_flux,
                                         'Harvest_q50')
        nharv = ncread(seaweed_file_flux,
                                         'nharv_Hq50')
        speciesnharv = nharv * speciesmask
        speciesnharv[speciesnharv==0] = np.nan
        
        d2port = d2port * seaweed / seaweed
        fseq_original = fseq_original * seaweed / seaweed
        # my_cmap = sns.color_palette('viridis', as_cmap=True)
        
        seaweed_sinkorship = ncread(r'/filepath/flux_sinkorship_analysis.nc','tea_mean')
        d2port = ncread(d2pfile,'d2p')
        d2port = d2port * seaweed / seaweed
        fseq_original = ncread(seqfracfile,'fseq_bottom_100y')
        fseq_original = fseq_original * seaweed / seaweed
        d2sink = ncread(r'/filepath/d2sink_fseq_maps_flux_q50.nc','d2sink')
        d2sink = d2sink * seaweed / seaweed
        seq_frac = ncread(r'/filepath/d2sink_fseq_maps_flux_q50.nc','fseq_100years_withtransport')
        seq_frac = seq_frac * seaweed / seaweed
        figx = cwm_map_pcar(d2sink,[0,500],'distance to optimal sinking location (km),\nlimited nutrients, median yield',ylat,xlon,my_cmap)
        plt.savefig('seaweed_d2sink_limited_051622.png', dpi=900, format='png', bbox_inches='tight')
            
        fig2 = cwm_products_map_fig2(r'/filepath/ambient_data.nc',[0,3000],'avoided emissions via products cost ($/t-CO2e), ambient',area,seaweed,ylat,xlon,my_cmap)
        
    run=1;
    if run==0 or run==1:
        import copy
    
        netcostperton_area_mask = copy.deepcopy(seaweed_harvest_DW_g_m2)
        netcostperton_area_mask[~np.isnan(netcostperton_area_mask)] = 1
        prodcost_min = ncread(r'/filepath/ambient_prodcost_analysis.nc','tea_min')
        prodcost_median = ncread(r'/filepath/ambient_prodcost_analysis.nc','tea_median')
        prodcost_max = ncread(r'/filepath/ambient_prodcost_analysis.nc','tea_max')
        net_costperton_sink = ncread(r'/filepath/ambient_netcostpertonsink_analysis.nc','tea_median')
        net_costperton_product = ncread(r'/filepath/ambient_netcostpertonproduct_analysis.nc','tea_median')
        net_costperton_food = ncread(r'/filepath/ambient_netcostpertonfood_analysis.nc','tea_q5')
        net_costperton_feed = ncread(r'/filepath/ambient_netcostpertonfeed_analysis.nc','tea_q5')
        net_costperton_fuel = ncread(r'/filepath/ambient_netcostpertonfuel_analysis.nc','tea_q5')
        netcost_carbon_removed = copy.deepcopy(net_costperton_sink)
        # netcost_carbon_removed = copy.deepcopy(net_costperton_product)
        # netcost_carbon_removed = copy.deepcopy(prodcost_median)
        net_emsremoved_sink = ncread(r'/filepath/ambient_netemssink_analysis.nc','tea_median')
        net_emsavoided_product = ncread(r'/filepath/ambient_netemsproduct_analysis.nc','tea_median')
        net_carbon_removed = copy.deepcopy(net_emsremoved_sink)
        # net_carbon_removed = copy.deepcopy(net_emsavoided_product)
        net_carbon_removed = net_carbon_removed 
        seaweed_biomass = copy.deepcopy(seaweed_harvest_DW_g_m2)
         
        percent_total = 1.22
        percent_step = 0.01 # 0.01 # want 0.1% bar resolution
        percent_step_100p = percent_total * percent_step # number to multiply total number of cells by to get number of percent-step resolution bars in 100%
        # print(percent_step_100p)
        subchunks = round(1/percent_step); # number of bars per percent
        subchunks_pass = copy.deepcopy(subchunks)
        total_bars = (100*percent_total) * subchunks
        total_bars_pass = copy.deepcopy(total_bars)
        
        avg_percent_cost, low10p_cost_map, sum_carbonremoved, carbon_removed_map, area_farmed_km2, farmed_area_notblocked_km2, percent_area_blocked, percent_oceanarea_farmed, farmed_area_map, percent_seaweedarea_farmed = percent_CDF(percent_total,percent_step,percent_step_100p,subchunks_pass,total_bars_pass,netcost_carbon_removed,net_carbon_removed,netcostperton_area_mask,seaweed_biomass,ships,mpas)
        
        sum_carbonremoved_mt = copy.deepcopy(sum_carbonremoved)
        sum_carbonremoved_mt = [x / 1e6 for x in sum_carbonremoved_mt]
        sum_carbonremoved_gt = copy.deepcopy(sum_carbonremoved)
        sum_carbonremoved_gt = [x / 1e9 for x in sum_carbonremoved_gt]
        
        percents = np.arange(1,round((percent_total/percent_step)+1),1)
        print(percents)
        xticks = np.linspace(percent_step,percent_total,num=round(percent_total/percent_step))
        xticks = np.around(xticks,decimals=1)
        # print(len(xticks))
        # print(len(sum_carbonremoved_gt))
        
        cdf_stats = pd.DataFrame()
        cdf_stats['cumulative_CO2_removed'] = sum_carbonremoved_gt
        cdf_stats['cumulative_C_removed'] = (cdf_stats['cumulative_CO2_removed'] / carbon_to_co2)
        cdf_stats['area_farmed_km2'] = area_farmed_km2
        cdf_stats['percent_ocean_area'] = percent_oceanarea_farmed
        cdf_stats['percent_seaweed_area'] = percent_seaweedarea_farmed
        cdf_stats['percent_seaweed_pixels'] = percents * (percent_step)
        cdf_stats['avg_cost_per_tonne_CO2e'] = avg_percent_cost
        cdf_stats['percent_ocean_area_linear_increase'] = (percents * (percent_step)) / percent_total
        cdf_stats['cumulative_CO2_removed_linear_oceanarea_increase'] = sum_carbonremoved_gt * (((percents * (percent_step)) / percent_total) / percent_oceanarea_farmed)
        print(cdf_stats['percent_ocean_area_linear_increase'])
        # cdf_stats.to_excel('TEST_flux_cdf_stats_fuel_low1p_lowcost_seaweed.xlsx')
        
        ### Plotting CDF v1 
        fig,ax = plt.subplots()
        cmap = my_cmap
        norm = mpl.colors.Normalize(vmin=2000, vmax=2800)
        plt.title('Median Cost, Flux Median Carbon Avoided,\n Cumulative CO2 removed via sinking')
        plt.xlabel('cumulative seaweed growth pixels (%)')

        def numfmt(x, pos): # your custom formatter function: divide by 100.0
            s = '{}'.format(x / (percent_total/(percent_total * percent_step)))
            return s

        import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
        xfmt = tkr.FuncFormatter(numfmt)    # create your custom formatter function

        pylab.gca().xaxis.set_major_formatter(xfmt)
        plt.ylabel('cumulative CO2 removed (Gt)')
        ax.bar(percents, cdf_stats.cumulative_CO2_removed_linear_oceanarea_increase, color=cmap(norm(cdf_stats.avg_cost_per_tonne_CO2e.values)))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # only needed for matplotlib < 3.1
        fig.colorbar(sm)
        # plt.savefig('flux_carbon_removed_1%area_sinking_medcost.eps', format='eps') #, dpi=300) 
        plt.show()

        fig3 = cwm_map_pcar(low10p_cost_map,[2000,2800],'ambient lowest 1% cost areas, low cost, sinking',ylat,xlon,my_cmap)
        # plt.savefig('flux_map_1%area_sinking_medost.png', dpi=900, format='png', bbox_inches='tight')
        
    run = 1;
    if run==1:  
        #### For determining the average yield in the cheapest 1% areas
        low10p_cost_map[~np.isnan(low10p_cost_map)] = 1
        low10p_cost_map_mask = copy.deepcopy(low10p_cost_map)
        seaweed_lowcost_area_production = low10p_cost_map_mask * seaweed_q50
        # seaweed_lowcost_area_products[seaweed_lowcost_area_products >= 3000] = np.nan
        weights = area / np.min(area)
        weights[np.isnan(seaweed_lowcost_area_production)] = 0;
        seaweed_lowcost_area_production[np.isnan(seaweed_lowcost_area_production)] = 0;
        lowcost_yield = np.average(seaweed_lowcost_area_production, weights = weights)
        print(lowcost_yield)

