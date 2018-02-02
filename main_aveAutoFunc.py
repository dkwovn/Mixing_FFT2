from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
from scipy import ma
from plot_funcs import plot_local_diffu_NH_col1
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=True
#PROCESS_FLAG=False

npro=1
yr_st=1984
yr_end=1985
fold_num=1
#zlev=3
zz=3
#ylev=37
#xlev=0
mask_sig=120.
tLag=60
lonWidth=90.
tSigma=30/4. #in days

def get_spectr(yr):
    #yr_num=0
    #for yr in range(yr_st,yr_end+1):
    print( str(yr))
    df1_name = '/home/cjliu/data/MERRA2/6hourly/ivar2.'+str(yr)+'.nc'
    df2_name = '/home/cjliu/data/MERRA2/6hourly/ivar2.'+str(yr+1)+'.nc'
#        df_name = '/home/cjliu/data/MERRA2/6hourly/merra2.'+str(yr)+'.nc4'
    dfile = Dataset(df1_name,mode='r')
    dfile2 = Dataset(df2_name,mode='r')
    lon = dfile.variables["lon"][:]
    lat = dfile.variables["lat"][:]
    lev = dfile.variables["lev"][:]
    time1 = dfile.variables["time"][-94:]/(24.*60)
    uwnd1 = dfile.variables["U"][-94:,:,:,:]
    vwnd1 = dfile.variables["V"][-94:,:,:,:]
    time2 = dfile2.variables["time"][:236]/(24.*60)
    uwnd2 = dfile2.variables["U"][:236,:,:,:]
    vwnd2 = dfile2.variables["V"][:236,:,:,:]
    time = np.concatenate((time1,time2),axis=0)
    uwnd = np.concatenate((uwnd1,uwnd2),axis=0)
    vwnd = np.concatenate((vwnd1,vwnd2),axis=0)
    #time = time2
    #uwnd = uwnd2
    #vwnd = vwnd2
    dfile.close()
    dfile2.close()
    lon_r=np.deg2rad(lon)
    lon2,time2=np.meshgrid(lon_r,time)
    nx=lon.shape[0]
    ny=lat.shape[0]
    nz=lev.shape[0]
    print( lev[zz])

    #diffu_grid=np.zeros([ny,nx, 2*tLag+1, ])
    u_mn=np.zeros([nx, ny])
    for zlev in range(zz,zz+1):
        print('z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)

        diffu_grid = flow.get_autocorr(tLag, lonWidth)
        u_mn = flow.um.T

    return lon,lat,lev,diffu_grid,u_mn

#===== main program =====
data_fname='/home/cjliu/data/npz/DiffuAveAuto350K_DJF_'+str(int(lonWidth))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
if PROCESS_FLAG:
    yr_num=0
    for yr in range(yr_st,yr_end+1):
        if yr_num==0:
            lon,lat,lev,autocorr,u_mn=get_spectr(yr)
        else:
            lon,lat,lev,tmp1,tmp2=get_spectr(yr_st)
            autocorr+=tmp1
            u_mn+=tmp2
        yr_num+=1
    autocorr/=yr_num
    u_mn/=yr_num
    df_name = '/home/cjliu/data/MERRA2/6hourly/ivar2.1980.nc'
    dfile = Dataset(df_name,mode='r')

    lon = dfile.variables["lon"][:]
    lat = dfile.variables["lat"][:]
    uwnd = dfile.variables["U"][-94:,:,:,:]
    vwnd = dfile.variables["V"][-94:,:,:,:]
    flow = mw_diffu(uwnd[:,0,:,:],vwnd[:,0,:,:],lon,lat)

    nx = lon.shape[0]
    ny = lat.shape[0]
    diffu_spectr = np.zeros([ny, nx])
    for x in range(nx):
        for y in range(ny):
            corr_U = flow.extractU(autocorr[y, x, :, :], u_mn[x, y], y)
            diffu_spectr[y, x] = np.sum(corr_U) * flow.deltaT
    #=== Error: u_mn became a masked array which cannot be stored in npz file ===
    #um_filled = ma.filled(u_mn, fill_value = 0)
    #np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=diffu_spectr,um_filled =um_filled)
    np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=diffu_spectr,um_filled = u_mn)
    #np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=diffu_spectr)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    diffu_spectr=data['diffu_spectr']
    um_filled =data['um_filled']


nx=lon.shape[0]
ny=lat.shape[0]
nt=360
#fig,axes=plt.subplots(1,1,figsize=[8,6])


#=== plot lon lat plot ===
zz=3
plt_fld=diffu_spectr[:,:]

#*** transform longitude ***
dim=plt_fld.shape
plt_fld_tmp=np.zeros(dim)
u_mn_tmp=np.zeros(dim)
lon_tmp=np.zeros(nx)
xr=int((nx)/2)
xl=xr-nx
ii=0
for i in range(xl,xr):
    plt_fld_tmp[:,ii]=plt_fld[:,i]
    u_mn_tmp[:,ii]=um_filled[i,:]
    if lon[i]<0:
        lon_tmp[ii]=lon[i]+360.
    else:
        lon_tmp[ii]=lon[i]
    ii+=1
lon=lon_tmp
plt_fld=plt_fld_tmp

u_mn_plt=u_mn_tmp

#lon_fld,lat_fld=np.meshgrid(lon,lat)


#plt_map=Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=0,urcrnrlon=360,urcrnrlat=80,ax=axes,resolution='l')
#plt_map.drawcoastlines(linewidth=0.8,color='gray')
#grid_x,grid_y=plt_map(lon_fld,lat_fld)
#
##plt_map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],linewidth=0)
##plt_map.drawmeridians(np.arange(0,360,30),labels=[0,0,0,1],linewidth=0)
##plt_map.drawparallels(np.arange(-90,90,10),linewidth=0)
##cs=plt_map.contourf(grid_x,grid_y,plt_fld/1e5,cmap=cm.rainbow,levels=np.arange(-50,350.1,50))
#
#cs=plt_map.contourf(grid_x,grid_y,plt_fld/1e5,cmap=cm.rainbow)
#plt_map.contour(grid_x,grid_y,u_mn_plt,colors='k',levels=np.arange(10,60.1,10))
#ax=fig.add_axes([0.16,0.05,0.7,0.01])
#cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
#axes.set_title("350K "+str(yr_st)+'-'+str(yr_end)+' lonWidth='+str(lonWidth))
title="350K "+str(yr_st)+'-'+str(yr_end)+' lonWidth='+str(lonWidth)


plot_local_diffu_NH_col1(lon, lat, plt_fld/1e5, u_mn_plt, title, levs = np.linspace(0, 100, 31))



plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/01_2018/diffu_aveAuto_lonW"+str(lonWidth)+"_"+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

