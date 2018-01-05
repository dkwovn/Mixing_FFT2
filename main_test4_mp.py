from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class_auto import mw_diffu,dc,xlev,ylev
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap


PROCESS_FLAG=True
#PROCESS_FLAG=False

npro=16
yr_st=1980
yr_end=1995
fold_num=16
#zlev=3
zz=3
#ylev=37
#xlev=0
mask_sig=10.
lonWidth=180.
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
    uwnd1 = dfile.variables["U"][-184:,:,:,:]
    vwnd1 = dfile.variables["V"][-184:,:,:,:]
    time2 = dfile2.variables["time"][:236]/(24.*60)
    uwnd2 = dfile2.variables["U"][:326,:,:,:]
    vwnd2 = dfile2.variables["V"][:326,:,:,:]
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
    u_mn=np.zeros([nx,ny,nz])
    for zlev in range(zz,zz+1):
        print(  'z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)

        #diffu_grid =flow.get_diffu_grid()
        result,corr_U=flow.get_diffu_grid3(tLag,lonWidth)
        result2=flow.get_diffu_traj(tLag)
        xWidth=int((result.shape[1]-1)/2)
#        print(result[:,xWidth])
        #print(result.shape)
        #print(flow.collectVal(100,110,130,'va'))
        #print(flow.interpVal(-180.5,110,130,'lon'))
        #print(flow.interpVal(180.,110,130,'lon'))
        u_mn[:,:,zlev]=flow.um.T
    lon_rel=np.arange(-xWidth,xWidth+1,1)*flow.dlon

    return lon,lat,lev,result,corr_U,lon_rel,flow.m2deg_x(flow.um[ylev,xlev],ylev),result2

#===== main program =====
    
#data_fname='npz/DiffuGrid350K_DJF_'+str(int(tLag))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
data_fname='npz/DiffuGrid350K_DJF_'+str(int(tLag))+'_'+str(xlev)+'_'+str(ylev)+'.npz'

if PROCESS_FLAG:
    pool=mp.Pool(processes=npro)
    output=[pool.apply_async(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
    results=[p.get() for p in output]
    lon=np.array(results[0][0])
    lat=np.array(results[0][1])
    lev=np.array(results[0][2])
    ps_num=len(results)
    ps=0
    for i in range(ps_num):
        if i==0:
            corr_U=results[i][4]
            result2=results[i][-1]
            u_lon=results[i][-2]
        else:
            corr_U+=results[i][4]
            result2+=results[i][-1]
            u_lon+=results[i][-2]
    
    corr_U/=ps_num
    result2/=ps_num
    u_lon/=ps_num
    np.savez(data_fname,lon=lon,lat=lat,lev=lev,corr_U=corr_U,result2=result2,u_lon=u_lon)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    corr_U=data['corr_U']
    result2=data['result2']
    u_lon=data['u_lon']


#lon,lat,lev,result,corr_U,lon_rel,u_lon,result2=get_spectr(1980)
u_lon=u_lon*86400.
fig,axes=plt.subplots(2,1,figsize=(8,8))
#traj_t=lon_rel/u_lon
t_plt=np.arange(-tLag,tLag+1,1)/4
#x_fld,y_fld=np.meshgrid(lon_rel,t_plt)
#vmax=result.max()
#cs=axes[0].contourf(x_fld,y_fld,result,cmap=cm.RdBu_r,levels=np.linspace(-vmax,vmax+0.1,20))
#cs=axes[0].contourf(x_fld,y_fld,result,cmap=cm.RdBu_r,levels=np.linspace(-150.,150.1,15))
#axes[0].plot(lon_rel,traj_t,color='k')
#axes[0].set_title('lon='+str(lon[xlev])+' lat='+str(lat[ylev]))
#axes[0].set_ylim(-15,15)
#ax=fig.add_axes([0.2,0.49,0.6,0.01])
#cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')


#n_zero=corr_U[abs(corr_U)<1e-8].shape[0]
#bnd_idx=int((corr_U.shape[0]-n_zero)/2)
#t_cutoff=t_plt[-int(n_zero/2)]
#integral[:]=np.sum(corr_U)/(bnd_idx*2)

ttLag=np.arange(tLag+1)/4.
integral=np.zeros(ttLag.shape[0])
for lag in range(tLag+1):
    integral[lag]=sum(corr_U[tLag-lag:tLag+lag])
axes[1].plot(ttLag,integral)
axes[1].plot(ttLag,integral-integral,'k',linestyle='--')

ttLag=np.arange(tLag+1)/4.
integral=np.zeros(ttLag.shape[0])
for lag in range(tLag+1):
    integral[lag]=sum(result2[tLag-lag:tLag+lag,ylev,xlev])
axes[0].plot(ttLag,integral)
axes[0].plot(ttLag,integral-integral,'k',linestyle='--')

#integral=np.zeros(t_plt.shape)
#integral[:]=np.sum(result2[:,ylev,xlev],0)/(bnd_idx*2)

#axes[0].plot(t_plt,result2[:,ylev,xlev])
#axes[0].plot(t_plt,integral)
#axes[0].plot(t_plt,integral-integral,'k',linestyle='--')
#axes[0].set_xlim(-1.*t_cutoff,t_cutoff)
axes[0].set_title('lon='+str(lon[xlev])+' lat='+str(lat[ylev]))
plt.savefig('/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/12_2017/diffu_tLag_test_'+str(tLag)+'_'+str(xlev)+'_'+str(ylev)+'.pdf',format='pdf')
plt.show()

