from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class_temp import mw_diffu,dc,xlev
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=True
PROCESS_FLAG=False

npro=1
yr_st=1980
yr_end=2015
fold_num=16
#zlev=3
zz=3
ylev=37
#xlev=0
mask_sig=120.
tLag=120
seasonStr="DJF"
day_st0=-94
day_end0=236
#day_st=237
#day_end=605
#day_st=605
#day_end=973
#day_st=973
#day_end=973+364


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
#    time1 = dfile.variables["time"][day_st:day_end]/(24.*60)
#    uwnd1 = dfile.variables["U"][day_st:day_end,:,:,:]
#    vwnd1 = dfile.variables["V"][day_st:day_end,:,:,:]
#    time = time1
#    uwnd = uwnd1
#    vwnd = vwnd1
    time1 = dfile.variables["time"][day_st0:]/(24.*60)
    uwnd1 = dfile.variables["U"][day_st0:,:,:,:]
    vwnd1 = dfile.variables["V"][day_st0:,:,:,:]
    time2 = dfile2.variables["time"][:day_end0]/(24.*60)
    uwnd2 = dfile2.variables["U"][:day_end0,:,:,:]
    vwnd2 = dfile2.variables["V"][:day_end0,:,:,:]
    time = np.concatenate((time1,time2),axis=0)
    uwnd = np.concatenate((uwnd1,uwnd2),axis=0)
    vwnd = np.concatenate((vwnd1,vwnd2),axis=0)
    #=== deal with undef data ===
    uwnd[abs(uwnd)>1e4] = np.nan
    vwnd[abs(vwnd)>1e4] = np.nan
    #============================
    dfile.close()
    dfile2.close()
    lon_r=np.deg2rad(lon)
    lon2,time2=np.meshgrid(lon_r,time)
    nx=lon.shape[0]
    ny=lat.shape[0]
    nz=lev.shape[0]
    print( lev[zz])

    diffu_spectr=np.zeros([nz,ny,nx,nx])
    diffu_grid=np.zeros([nz,ny,nx])
    u_mn=np.zeros([nx,ny,nz])
    rh_fld = []
#    for zlev in range(zz,zz+1):
    for zlev in range(nz):
        print(  'z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)

        maskFunc=flow.genMaskGauss(mask_sig)
        maskFunc[:]=1.
        #diffu_spectr[zlev,:,:,:],um,temp =flow.get_diffu_spectr(maskFunc,'zm')
        diffu_spectr[zlev,:,:,:],um,temp =flow.get_diffu_rh(maskFunc,'zm')
        rh_fld.append(temp)
        u_mn[:,:,zlev]=um

    #=== shape of rh_fld (nz,ny,nc)
    return lon,lat,lev,np.sum(diffu_spectr,3),u_mn,np.array(rh_fld),flow.cc
    #return lon,lat,lev,diffu_grid,u_mn

#===== main program =====
    
data_fname='/home/cjliu/data/npz/DiffuSpectr_zm_'+seasonStr+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
if PROCESS_FLAG:
    pool=mp.Pool(processes=npro)
    output=[pool.apply_async(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
    results=[p.get() for p in output]
    lon=np.array(results[0][0])
    lat=np.array(results[0][1])
    lev=np.array(results[0][2])
    cc=np.array(results[0][6])
    ps_num=len(results)
    #ps=0
    diffu_spectr = []
    u_mn = []
    rh_fld = []
    for i in range(ps_num):
        diffu_spectr.append(results[i][3])
        u_mn.append(results[i][4])
        rh_fld.append(results[i][5])
#        if i==0:
#            diffu_spectr=results[i][3]
#            u_mn=results[i][4]
#        else:
#            diffu_spectr+=results[i][3]
#            u_mn+=results[i][4]
    #diffu_spectr/=ps_num
    #u_mn/=ps_num
        

    np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=np.array(diffu_spectr),u_mn=np.array(u_mn),rh_fld=rh_fld,cc=cc)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    #cc=data['cc']
    diffu_spectr=data['diffu_spectr']
    u_mn=data['u_mn']

nx=lon.shape[0]
ny=lat.shape[0]
nt=360
fig,axes=plt.subplots(1,1,figsize=[8,8])

#=== plot line plot ===
#axes.plot(lat,np.mean(diffu_spectr[zz,:,:],1))

def linTrend(var):
    yrs = np.arange(yr_st,yr_end+1)
    d2 = var.shape[1]
    d3 = var.shape[2]
    result = np.zeros([d2,d3])
    for j in range(d2):
        for k in range(d3):
            #result[j,k] = np.mean(var[:,j,k])
            result[j,k] = np.polyfit(yrs,var[:,j,k],1)[0]
    return result
#=== shapes of variables ===
# diffu_sectr[nt,nz,ny,nx]
# u_mn[nt,nx,ny,nz]
# rh_fld[nt,ny,nc]
#=== plot lat lev plot  ===
grid_x,grid_y=np.meshgrid(lat,lev)
#diffu_spectr = np.mean(diffu_spectr,0)
diffu_spectr = linTrend(diffu_spectr[:,:,:,xlev])
u_mn = np.mean(u_mn,0)
#plt_fld=diffu_spectr[:,:,xlev]
plt_fld=diffu_spectr[:,:]
levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
#cs=axes.contourf(grid_x,grid_y,np.log(plt_fld)/np.log(10.),cmap=cm.Oranges,levels=np.arange(4,7,0.2))
#cs2=axes.contour(grid_x,grid_y,u_mn[xlev,:,:].T,colors='k',levels=[10.,20.,30.,40.])
cs=axes.contourf(grid_x,grid_y,plt_fld/1e4,cmap=cm.RdBu_r,levels=np.arange(-10.,10.1,1.))
cs2=axes.contour(grid_x,grid_y,u_mn[xlev,:,:].T,colors='k',levels=[10,20,30,40])
ax=fig.add_axes([0.16,0.05,0.7,0.01])
cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
axes.clabel(cs2)
axes.set_title("Zonal mean diffusivity")
axes.set_xticks(np.arange(-80,80.1,20.))

#=== plot lon lat plot ===
#zz=3
#diffu_spectr = np.mean(diffu_spectr,0)
#plt_fld=diffu_spectr[0,zz,:,:]
#
#lon_fld,lat_fld=np.meshgrid(lon,lat)
#
#zm=np.sum(plt_fld,1)/1e5
#axes[0].plot(lat,zm)
#axes[0].set_xlim(-85,85)

#plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/12_2017/diffu_spectr_zm_"+seasonStr+'_'+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

