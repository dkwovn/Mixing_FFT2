from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=False

npro=16
yr_st=1980
yr_end=1995
fold_num=16
#zlev=3
zz=3
ylev=37
xlev=0
mask_sig=120.

def get_spectr(yr):
    #yr_num=0
    #for yr in range(yr_st,yr_end+1):
    print str(yr)
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
    lon_r=np.deg2rad(lon)
    lon2,time2=np.meshgrid(lon_r,time)
    nx=lon.shape[0]
    ny=lat.shape[0]
    nz=lev.shape[0]
    print lon[xlev]
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
    diffu_spectr=np.zeros([nz,ny,nx,nx])
    u_mn=np.zeros([nx,ny,nz])
    for zlev in range(zz,zz+1):
    #for zlev in range(nz):
        print  'z=',zlev
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)

        maskFunc=flow.genMaskGauss(mask_sig)
    #    maskFunc[:]=1.

        #if yr_num==0:
        diffu_spectr[zlev,:,:,:],u_mn[:,:,zlev]=flow.get_diffu(maskFunc)
        #else:
            #tmp1,tmp2=flow.get_diffu(maskFunc)
            #diffu_spectr[zlev,:,:,:]+=tmp1
            #u_mn[:,:,zlev]+=tmp2
    #yr_num+=1

    #diffu_spectr/=yr_num
    #u_mn/=yr_num
    return lon,lat,lev,np.sum(diffu_spectr,3),u_mn

#===== main program =====
    
#data_fname='diffu350K_test_xy_mp.npz'
#data_fname='diffu350K_test.npz'
#data_fname='npz/Diffu350K__'+str(yr_st)+'_'+str(yr_end)+'.npz'
data_fname='npz/Diffu350K_DJF_'+str(int(mask_sig))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
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
ny=lat.shape[0]
nt=360
fig,axes=plt.subplots(2,1,figsize=[8,10])


#=== plot lon lat plot ===
#plt_fld=np.zeros([ny,nx])
#lon_plt=np.zeros(lon.shape)
zz=3
#plt_fld=diffu_spectr[zz,:,:]*nx/(np.sqrt(2*np.pi)*mask_sig)
plt_fld=diffu_spectr[zz,:,:]
#for x in range(nx):
    #plt_fld[:,x]=diffu_spectr[zz,:,x-nx/2]*nx/(np.sqrt(2*np.pi)*mask_sig)
    #lon_plt[x]=lon[x-nx/2]
    #if lon[x]<0:
        #lon[x]+=360.
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
    u_mn_tmp[:,ii]=u_mn[i,:,zz]
    if lon[i]<0:
        lon_tmp[ii]=lon[i]+360.
    else:
        lon_tmp[ii]=lon[i]
    ii+=1
lon=lon_tmp
plt_fld=plt_fld_tmp*nx/(np.sqrt(2*np.pi)*mask_sig)

u_mn_plt=u_mn_tmp

#u_mn_plt=u_mn[:,:,zz].T
lon_fld,lat_fld=np.meshgrid(lon,lat)


plt_zm=np.sum(plt_fld,1)/nx
#=== plot lines for isentropes ===
#grid_x,grid_y=np.meshgrid(lat,lev)
#plt_fld=diffu_spectr[:,:,xlev]
cs=axes[1].plot(lat,(plt_zm)/1e5,color='k')
axes[1].set_xlim(-85,85)
axes[1].set_ylim(0,15)
axes[1].set_title("Zonal mean diffusivity (1980-2016)")
axes[1].set_xticks(np.arange(-80,80.1,20.))


#cs=axes.contourf(grid_x,grid_y,np.log(plt_fld)/np.log(10.),cmap=cm.Oranges)
plt_map=Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-80,urcrnrlon=360,urcrnrlat=80,ax=axes[0],resolution='l')
plt_map.drawcoastlines(linewidth=0.8,color='gray')
grid_x,grid_y=plt_map(lon_fld,lat_fld)
plt_map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],linewidth=0)
plt_map.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1],linewidth=0)
cs=plt_map.contourf(grid_x,grid_y,plt_fld/1e5,cmap=cm.rainbow,levels=np.arange(0,25.1,2.))
plt_map.contour(grid_x,grid_y,u_mn_plt,colors='k',levels=np.arange(10,60.1,10))
ax=fig.add_axes([0.16,0.54,0.7,0.01])
cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
cb0.set_label("x10$^5 m^2s^{-1}$",labelpad=-1.0)
axes[0].set_title("350K ("+str(yr_st)+'-'+str(yr_end)+") sigma="+str(mask_sig))

plt.savefig("/home/cliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/11_2017/diffu_350K_2pnls"+str(mask_sig)+".pdf",format='pdf')
plt.show()


    
