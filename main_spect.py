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


PROCESS_FLAG=True
#PROCESS_FLAG=False

npro=8
yr_st=1980
yr_end=1995
fold_num=16
#zlev=3
zz=3
ylev=37
xlev=0
mask_sig=120.
tLag=120
day_st=237
day_end=605


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
    time1 = dfile.variables["time"][day_st:day_end]/(24.*60)
    uwnd1 = dfile.variables["U"][day_st:day_end,:,:,:]
    vwnd1 = dfile.variables["V"][day_st:day_end,:,:,:]
#    time1 = dfile.variables["time"][-94:]/(24.*60)
#    uwnd1 = dfile.variables["U"][-94:,:,:,:]
#    vwnd1 = dfile.variables["V"][-94:,:,:,:]
    #time2 = dfile2.variables["time"][:236]/(24.*60)
    #uwnd2 = dfile2.variables["U"][:236,:,:,:]
    #vwnd2 = dfile2.variables["V"][:236,:,:,:]
    #time = np.concatenate((time1,time2),axis=0)
    #uwnd = np.concatenate((uwnd1,uwnd2),axis=0)
    #vwnd = np.concatenate((vwnd1,vwnd2),axis=0)
    time = time1
    uwnd = uwnd1
    vwnd = vwnd1
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
    for zlev in range(zz,zz+1):
    #for zlev in range(nz):
        print(  'z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)

        maskFunc=flow.genMaskGauss(mask_sig)
    #    diffu_grid[zlev,:,:] =np.mean(flow.get_diffu_grid(tLag),2)
#        diffu_spectr[zlev,:,:,:],um =flow.get_diffu_spectr(maskFunc,'lonRes')
        diffu_spectr[zlev,:,:,:],um =flow.get_diffu_spectr(maskFunc,'zm')
        u_mn[:,:,zlev]=um

    return lon,lat,lev,np.sum(diffu_spectr,3),u_mn
    #return lon,lat,lev,diffu_grid,u_mn

#===== main program =====
    
data_fname='/home/cjliu/data/npz/Diffu350K_U0_MAM_'+str(int(mask_sig))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
#data_fname='/home/cjliu/data/npz/DiffuGrid350K_DJF_'+str(int(tLag))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
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
fig,axes=plt.subplots(2,1,figsize=[8,8])

#=== plot line plot ===
#axes.plot(lat,np.mean(diffu_spectr[zz,:,:],1))

#=== plot lat lev plot  ===
#grid_x,grid_y=np.meshgrid(lat,lev)
#plt_fld=diffu_spectr[:,:,xlev]
#levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
#cs=axes.contourf(grid_x,grid_y,np.log(plt_fld)/np.log(10.),cmap=cm.Oranges,levels=np.arange(4,7,0.2))
#axes.contour(grid_x,grid_y,u_mn[xlev,:,:].T,colors='k')
#ax=fig.add_axes([0.16,0.05,0.7,0.01])
#cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
#axes.set_title("Zonal mean diffusivity")
#axes.set_xticks(np.arange(-80,80.1,20.))

#=== plot lon lat plot ===
zz=3
plt_fld=diffu_spectr[zz,:,:]
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
#plt_fld=plt_fld_tmp*nx/(np.sqrt(2*np.pi)*mask_sig)
plt_fld=plt_fld_tmp

u_mn_plt=u_mn_tmp

lon_fld,lat_fld=np.meshgrid(lon,lat)


#axes.contourf(lon_fld,lat_fld,plt_fld)
plt_map=Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-80,urcrnrlon=360,urcrnrlat=80,ax=axes[0],resolution='l')
plt_map.drawcoastlines(linewidth=0.8,color='gray')
grid_x,grid_y=plt_map(lon_fld,lat_fld)
#plt_map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],linewidth=0)
#plt_map.drawmeridians(np.arange(0,360,30),labels=[0,0,0,1],linewidth=0)
#plt_map.drawparallels(np.arange(-90,90,10),linewidth=0)
cs=plt_map.contourf(grid_x,grid_y,plt_fld/1e5,cmap=cm.rainbow)
plt_map.contour(grid_x,grid_y,u_mn_plt,colors='k',levels=np.arange(10,60.1,10))
#plt_map.contour(grid_x,grid_y,grid_x,colors='k')
ax=fig.add_axes([0.16,0.05,0.7,0.01])
cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
axes[0].set_title("350K "+str(yr_st)+'-'+str(yr_end)+" sig="+str(mask_sig))

zm=np.mean(plt_fld,1)/1e5
axes[1].plot(lat,zm)
axes[1].set_xlim(-85,85)

plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/11_2017/diffu_spectr_350K_lonSig"+str(mask_sig)+'_'+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

