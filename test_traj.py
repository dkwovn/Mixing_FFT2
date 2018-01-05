from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc
import matplotlib as mpl
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
yr_end=1980
fold_num=1
#zlev=3
zz=3
#ylev=37
#xlev=0
mask_sig=120.
tLag=30
lonWidth=15.

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
    time2 = dfile2.variables["time"][:84]/(24.*60)
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

    diffu_spectr=np.zeros([nz,ny,nx,nx])

    #==== generate lon_out,lat_out ====
    lon_out=lon[::2]
    lat_out=lat[::2]
    print(lon_out)
    print(lat_out)
    nx_o=lon_out.shape[0]
    ny_o=lat_out.shape[0]
    #==================================
    #diffu_grid=np.zeros([nz,ny_o,nx_o])
    u_mn=np.zeros([nx,ny,nz])
    for zlev in range(zz,zz+1):
    #for zlev in range(nz):
        print(  'z=',zlev)
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat,lon_out,lat_out)

    #    diffu_grid[zlev,:,:] =np.mean(flow.get_diffu_grid(tLag),2)
        diffu_grid,lon_traj,lat_traj =flow.test_diffu_traj(tLag)
        u_mn[:,:,zlev]=flow.um.T

    #return lon,lat,lev,np.sum(diffu_spectr,3),u_mn
    return lon_out,lat_out,lev,diffu_grid,lon_traj,lat_traj

#===== main program =====
    
data_fname='/home/cjliu/data/npz/testDiffuTraj350K_DJF_'+str(int(tLag))+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
if PROCESS_FLAG:
    lon,lat,lev,diffu_spectr,lon_traj,lat_traj=get_spectr(yr_st)
    np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=diffu_spectr,lon_traj=lon_traj,lat_traj=lat_traj)
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
#    pool=mp.Pool(processes=npro)
#    #results=[pool.apply(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
#    output=[pool.apply_async(get_spectr,args=(yr,)) for yr in range(yr_st,yr_end+1)]
#    results=[p.get() for p in output]
#    lon=np.array(results[0][0])
#    lat=np.array(results[0][1])
#    lev=np.array(results[0][2])
#    diffu_spectr=[]
#    u_mn=[]
#    ps_num=len(results)
#    ps=0
#    for i in range(ps_num):
#        diffu_spectr.append(results[i][3])
#        u_mn.append(results[i][4])
##        if i==0:
##            diffu_spectr=results[i][3]
##            u_mn=results[i][4]
##        else:
##            diffu_spectr+=results[i][3]
##            u_mn+=results[i][4]
##    diffu_spectr/=ps_num
##    u_mn/=ps_num
#    diffu_spectr=np.array(diffu_spectr)
#    u_mn=np.array(u_mn)
        

else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    diffu_spectr=data['diffu_spectr']
    lon_traj=data['lon_traj']
    lat_traj=data['lat_traj']

nx=lon.shape[0]
ny=lat.shape[0]
nt=300
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
#plt_fld=np.mean(diffu_spectr[10:15,:,:,:],0)[zz,:,:]
#*** transform longitude ***
#dim=plt_fld.shape
#plt_fld_tmp=np.zeros(dim)
#u_mn_tmp=np.zeros(dim)
#lon_tmp=np.zeros(nx)
#xr=int((nx)/2)
#xl=xr-nx
#ii=0
#for i in range(xl,xr):
#    plt_fld_tmp[:,ii]=plt_fld[:,i]
#    #u_mn_tmp[:,ii]=u_mn[i,:,zz]
#    if lon[i]<0:
#        lon_tmp[ii]=lon[i]+360.
#    else:
#        lon_tmp[ii]=lon[i]
#    ii+=1
#lon=lon_tmp
##plt_fld=plt_fld_tmp*nx/(np.sqrt(2*np.pi)*mask_sig)
#plt_fld=plt_fld_tmp
#
##u_mn_plt=u_mn_tmp
#
lon_fld,lat_fld=np.meshgrid(lon,lat)


#axes.contourf(lon_fld,lat_fld,plt_fld)
plt_map=Basemap(projection='cyl',llcrnrlon=-180,llcrnrlat=-80,urcrnrlon=180,urcrnrlat=80,ax=axes[0],resolution='l')
plt_map.drawcoastlines(linewidth=0.8,color='gray')
#grid_x,grid_y=plt_map(lon_fld,lat_fld)
cmap=cm.rainbow
bounds=np.arange(-10,100.1,10.)
norm=mpl.colors.BoundaryNorm(bounds,cmap.N)
#cs=plt_map.pcolor(grid_x,grid_y,plt_fld/1e5,cmap=cmap,norm=norm)
#ax=fig.add_axes([0.16,0.05,0.7,0.01])
#cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
#for t in range(tLag,nt-tLag):
for t in range(100,105):
    grid_x,grid_y=plt_map(lon_traj[t,10:tLag+1,18,115],lat_traj[t,10:tLag+1,18,115])
    plt_map.plot(grid_x,grid_y)
    for tt in range(2*tLag+1):
        print(lon_traj[t,tt,18,105],lat_traj[t,tt,18,105],diffu_spectr[t,tt,18,105])

#for t in range(100,105):
    #grid_x,grid_y=plt_map(lon_traj[t,10:tLag+1,18,95],lat_traj[t,10:tLag+1,18,95])
    #plt_map.plot(grid_x,grid_y)
axes[0].set_title("350K "+str(yr_st)+'-'+str(yr_end)+' tLag='+str(tLag/4.))

plt_map=Basemap(projection='cyl',llcrnrlon=-180,llcrnrlat=-80,urcrnrlon=180,urcrnrlat=80,ax=axes[1],resolution='l')
plt_map.drawcoastlines(linewidth=0.8,color='gray')
for t in range(100,105):
    grid_x,grid_y=plt_map(lon_traj[t,10:tLag+1,18,95],lat_traj[t,10:tLag+1,18,95])
    plt_map.plot(grid_x,grid_y)
plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/12_2017/test_traj_tLag"+str(tLag)+"_"+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

