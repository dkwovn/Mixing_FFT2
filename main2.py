from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
#print 'Multi-wavenumber theory for diffusivity\n'

PROCESS_FLAG=True
#PROCESS_FLAG=False

npro=16
yr_st=1980
yr_end=2015
fold_num=16
#zlev=3
zz=3
ylev=37
xlev=0

def get_spectr(yr):
    #yr_num=0
    #for yr in range(yr_st,yr_end+1):
    print str(yr)
#    df_name = '/home/cliu/data/MERRA2/6hourly/ivar2.'+str(yr)+'.nc'
#    dfile = Dataset(df_name,mode='r')
#    lon = dfile.variables["lon"][:]
#    lat = dfile.variables["lat"][:]
#    lev = dfile.variables["lev"][:]
#    time = dfile.variables["time"][:360]/(24.*60)
#    uwnd = dfile.variables["U"][:360,:,:,:]
#    vwnd = dfile.variables["V"][:360,:,:,:]
#    lon_r=np.deg2rad(lon)
#    lon2,time2=np.meshgrid(lon_r,time)
#    nx=lon.shape[0]
#    ny=lat.shape[0]
#    nz=lev.shape[0]
#    print lon[xlev]

    df1_name = '/home/cliu/data/MERRA2/6hourly/ivar2.'+str(yr)+'.nc'
    df2_name = '/home/cliu/data/MERRA2/6hourly/ivar2.'+str(yr+1)+'.nc'
#        df_name = '/home/cliu/data/MERRA2/6hourly/merra2.'+str(yr)+'.nc4'
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
    #==== simple waves ====
#        simple_v=np.zeros(uwnd[:,zlev,:,:].shape)
#        k0=6
#        period0=4
#        k1=6
#        period1=2
#    simple_v[:,ylev,:]=np.exp(1.0j*(k0*lon2-2*np.pi/period0*time2))
    #for y in range(ny):
    #    simple_v[:,y,:]=np.exp(1.0j*(k0*lon2-2*np.pi/period0*time2))+np.exp(1.0j*(k1*lon2-2*np.pi/period1*time2))
#    simple_v[:,:,:]=simple_v[:,ylev,:]
    #simple_flow=mw_diffu(uwnd[:,zlev,:,:],simple_v,lon,lat)
    #if yr_num==0: 
    lon_r=np.deg2rad(lon)
    lon2,time2=np.meshgrid(lon_r,time)
    nx=lon.shape[0]
    ny=lat.shape[0]
    nz=lev.shape[0]
    diffu_spectr=np.zeros([nz,ny,nx,nx])
    u_mn=np.zeros([nx,ny,nz])
    #for zlev in range(zz,zz+1):
    for zlev in range(nz):
        print  'z=',zlev
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)
        maskFunc=flow.genMaskGauss(22.)
        maskFunc[:]=1.

        #if yr_num==0:
        diffu_spectr[zlev,:,:,:],u_mn[:,:,zlev]=flow.get_diffu(maskFunc,'zm')
        #else:
            #tmp1,tmp2=flow.get_diffu(maskFunc)
            #diffu_spectr[zlev,:,:,:]+=tmp1
            #u_mn[:,:,zlev]+=tmp2
    #yr_num+=1

    #diffu_spectr/=yr_num
    #u_mn/=yr_num
    return lon,lat,lev,np.sum(diffu_spectr,3),u_mn

#===== main program =====
    
data_fname='/home/cjliu/data/npz/zmDiffu_DJF_'+str(yr_st)+'_'+str(yr_end)+'.npz'
if PROCESS_FLAG:
#    yr_num=0
#    for yr in range(yr_st,yr_end+1):
#        if yr_num==0:
#            lon,lat,lev,diffu_spectr,u_mn=get_spectr(yr)
#        else:
#            lon,lat,lev,tmp1,tmp2=get_spectr(yr_st)
#            diffu_spectr+=tmp1
#            u_mn+=tmp2
#        yr_num+=1
#    diffu_spectr/=yr_num
#    u_mn/=yr_num
    pool=mp.Pool(processes=npro)
    #results=[pool.apply(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
    output=[pool.apply_async(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
    results=[p.get() for p in output]
    lon=np.array(results[0][0])
    lat=np.array(results[0][1])
    lev=np.array(results[0][2])
    ps_num=len(results)
    ps=0
    for i in range(ps_num):
        if i==0:
            diffu_spectr=results[i][3]
            u_mn=results[i][4]
        else:
            diffu_spectr+=results[i][3]
            u_mn+=results[i][4]
    diffu_spectr/=ps_num
    u_mn/=ps_num
        

    np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=diffu_spectr,u_mn=u_mn)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    diffu_spectr=data['diffu_spectr']
    u_mn=data['u_mn']

nx=lon.shape[0]
nt=360
fig,axes=plt.subplots(1,1,figsize=[8,5])

#=== plot yz diffu ===
grid_x,grid_y=np.meshgrid(lat,lev)
plt_fld=diffu_spectr[:,:,xlev]
levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
cs=axes.contourf(grid_x,grid_y,np.log(plt_fld)/np.log(10.),cmap=cm.Oranges,levels=np.arange(4,7,0.2))
axes.contour(grid_x,grid_y,u_mn[xlev,:,:].T,colors='k')
ax=fig.add_axes([0.16,0.05,0.7,0.01])
cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
axes.set_title("Zonal mean diffusivity (1980-2016)")
axes.set_xticks(np.arange(-80,80.1,20.))

#=== plot lon lat plot ===
#grid_x,grid_y=np.meshgrid(lon,lat)
#plt_fld=diffu_spectr[zz,:,:]
#cs=axes.contourf(grid_x,grid_y,np.log(plt_fld)/np.log(10.),cmap=cm.Oranges)
#axes.contour(grid_x,grid_y,u_mn[:,:,zz].T,colors='k')
#ax=fig.add_axes([0.16,0.05,0.7,0.01])
#cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')

#plt.savefig("/home/cliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/11_2017/diffu_zm_yz.pdf",format='pdf')
plt.show()


    

