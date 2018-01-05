from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc,xlev,ylev
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap


PROCESS_FLAG=True

npro=1
yr_st=1980
yr_end=1980
fold_num=16
#zlev=3
zz=3
#ylev=37
#xlev=0
mask_sig=10.
tLag=60

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
    dfile.close()
    dfile2.close()
    nx=lon.shape[0]
    ny=lat.shape[0]
    nz=lev.shape[0]
    print( lev[zz])

    diffu_spectr=np.zeros([nz,ny,nx,nx])
    #diffu_grid=np.zeros([nz,ny,nx])
    #==== generate lon_out,lat_out ====
    lon_out=lon[::20]
    lat_out=lat[::20]
    print(lon_out)
    print(lat_out)
    nx_o=lon_out.shape[0]
    ny_o=lat_out.shape[0]
    #==================================
    diffu_grid=np.zeros([nz,ny_o,nx_o])
    u_mn=np.zeros([nx,ny,nz])
    for zlev in range(zz,zz+1):
        print(  'z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat,lon_out,lat_out)

        #diffu_grid =flow.get_diffu_grid()
    #    result,corr_U=flow.get_diffu_grid3(tLag,100.)
        result=flow.test_diffu_traj(tLag)
        xWidth=int((result.shape[1]-1)/2)
#        print(result[:,xWidth])
        #print(result.shape)
        #print(flow.collectVal(100,110,130,'va'))
        #print(flow.interpVal(-180.5,110,130,'lon'))
        #print(flow.interpVal(180.,110,130,'lon'))
        u_mn[:,:,zlev]=flow.um.T
    lon_rel=np.arange(-xWidth,xWidth+1,1)*flow.dlon

    return lon_out,lat_out,lev,result,flow.m2deg_x(flow.um[ylev,xlev],ylev)

#===== main program =====
    
data_fname='npz/DiffuGrid350K_DJF_'+str(int(mask_sig))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'

lon,lat,lev,result,u_lon=get_spectr(1980)
u_lon=u_lon*86400.
fig,axes=plt.subplots(2,1,figsize=(8,8))
#traj_t=lon_rel/u_lon
t_plt=np.arange(-tLag,tLag+1,1)/4


integral=np.zeros(t_plt.shape)
#n_zero=corr_U[abs(corr_U)<1e-8].shape[0]
#bnd_idx=int((corr_U.shape[0]-n_zero)/2)
#t_cutoff=t_plt[-int(n_zero/2)]
t_cutoff=15.

integral=np.zeros(t_plt.shape)
integral[:]=np.mean(result[:,ylev,xlev],0)
axes[0].plot(t_plt,result[:,ylev,xlev])
axes[0].plot(t_plt,integral)
axes[0].plot(t_plt,integral-integral,'k',linestyle='--')
axes[0].set_xlim(-1.*t_cutoff,t_cutoff)
axes[0].set_title('lon='+str(lon[xlev])+' lat='+str(lat[ylev]))
#plt.savefig('/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/11_2017/traj_auto_test'+str(xlev)+'_'+str(ylev)+'.pdf',format='pdf')
plt.show()

