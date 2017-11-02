from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc
import matplotlib.pyplot as plt
from matplotlib import cm
#print 'Multi-wavenumber theory for diffusivity\n'

PROCESS_FLAG=True
#PROCESS_FLAG=False

yr_st=1983
yr_end=2016
fold_num=16
zlev=3
ylev=37

def get_spectr():
    yr_num=0
    for yr in range(yr_st,yr_end+1):
        print str(yr)
        df_name = '/home/cliu/data/MERRA2/6hourly/ivar.'+str(yr)+'.nc'
#        df_name = '/home/cliu/data/MERRA2/6hourly/merra2.'+str(yr)+'.nc4'
        dfile = Dataset(df_name,mode='r')
        lon = dfile.variables["lon"][:]
        lat = dfile.variables["lat"][:]
        lev = dfile.variables["lev"][:]
        time = dfile.variables["time"][:360]/(24.*60)
        uwnd = dfile.variables["U"][:360,:,:,:]
        vwnd = dfile.variables["V"][:360,:,:,:]
        lon_r=np.deg2rad(lon)
        lon2,time2=np.meshgrid(lon_r,time)
        simple_v=np.zeros(uwnd[:,zlev,:,:].shape)
        k0=6
        period0=4
        k1=6
        period1=2
    #    simple_v[:,ylev,:]=np.exp(1.0j*(k0*lon2-2*np.pi/period0*time2))
        ny=lat.shape[0]
        #for y in range(ny):
        #    simple_v[:,y,:]=np.exp(1.0j*(k0*lon2-2*np.pi/period0*time2))+np.exp(1.0j*(k1*lon2-2*np.pi/period1*time2))
    #    simple_v[:,:,:]=simple_v[:,ylev,:]
        flow = mw_diffu(uwnd[:,zlev,:,:],vwnd[:,zlev,:,:],lon,lat)
        #simple_flow=mw_diffu(uwnd[:,zlev,:,:],simple_v,lon,lat)
    
        maskFunc=flow.genMaskRect(1)
        maskFunc[:]=1.
        #Spectr2,Spectr_k = simple_flow.test_diffu(maskFunc)
        if yr_num==0:
            Spectr_yz,Spectr_yz2,u_mn=flow.get_randel_held(maskFunc)
            variance_y=flow.get_vv_xt()
            variance_y2=flow.get_uv_xt()
        else:
            tmp1,tmp2,tmp3=flow.get_randel_held(maskFunc)
            Spectr_yz+=tmp1
            Spectr_yz2+=tmp2
            variance_y+=flow.get_vv_xt()
            variance_y2+=flow.get_uv_xt()
            u_mn+=tmp3
        yr_num+=1
    
        dfile.close()
    Spectr_yz/=yr_num
    Spectr_yz2/=yr_num
    variance_y/=yr_num
    variance_y2/=yr_num
    u_mn/=yr_num
    return lon,lat,Spectr_yz,Spectr_yz2,u_mn,variance_y,variance_y2

#===== main program =====
    
data_fname='spectr350K_yc.npz'
if PROCESS_FLAG:
    lon,lat,Spectr_yz,Spectr_yz2,u_mn,variance_y,variance_y2=get_spectr()
    np.savez(data_fname,lon=lon,lat=lat,Spectr_yz=Spectr_yz,Spectr_yz2=Spectr_yz2,u_mn=u_mn,variance_y=variance_y,variance_y2=variance_y2)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    Spectr_yz=data['Spectr_yz']
    Spectr_yz2=data['Spectr_yz2']
    variance_y=data['variance_y']
    variance_y2=data['variance_y2']
    u_mn=data['u_mn']
    #==== read the nc file ====

nx=lon.shape[0]
nt=360
fig,axes=plt.subplots(2,2,figsize=[8,9])

#=== plot randel_held type uv diagram ===
cc=np.arange(-50,50,dc)*1.0
grid_x,grid_y=np.meshgrid(cc,lat)
plt_fld=Spectr_yz[100,:,:]/(nx*nt)
#cs=axes[0,0].contourf(grid_x,grid_y,plt_fld,cmap=cm.RdBu_r,levels=np.arange(-1,1.1,0.2))
cs=axes[0,0].contourf(grid_x,grid_y,plt_fld,cmap=cm.RdBu_r)
axes[0,0].plot(u_mn[100,:],lat)
axes[0,0].set_ylim(-80,80)
axes[0,0].set_xlim(-10,50)
ax1=fig.add_axes([0.13,0.5,0.35,0.01])
cb0=plt.colorbar(cs,cax=ax1,extend=0.8,orientation='horizontal')
axes[0,0].set_title("UV on 350K")
#plt.colorbar(cs)


#=== plot randel_held type vv diagram ===
cc=np.arange(-50,50,dc)*1.0
grid_x,grid_y=np.meshgrid(cc,lat)
plt_fld=Spectr_yz2[100,:,:]/(nx*nt)
#cs2=axes[0,1].contourf(grid_x,grid_y,plt_fld,cmap=cm.Oranges,levels=np.arange(0,5.1,0.5))
cs2=axes[0,1].contourf(grid_x,grid_y,plt_fld,cmap=cm.Oranges)
axes[0,1].plot(u_mn[100,:],lat)
axes[0,1].set_ylim(-80,80)
axes[0,1].set_xlim(-10,50)
ax2=fig.add_axes([0.56,0.5,0.35,0.01])
cb0=plt.colorbar(cs2,cax=ax2,extend=0.8,orientation='horizontal')
axes[0,1].set_title("VV on 350K")
#plt.savefig('Fig1_isen.pdf',format='pdf')

#=== plot variance verification ===
plt_rh_uv=2*np.sum(Spectr_yz[100,:,:],1)/(nx*nt)
plt_rh_vv=2*np.sum(Spectr_yz2[100,:,:],1)/(nx*nt)
plt_uv=variance_y2
plt_vv=variance_y
axes[1,0].plot(plt_rh_uv/(nx*nt),lat)
axes[1,0].plot(plt_uv/(nx*nt),lat)

axes[1,1].plot(plt_rh_vv/(nx*nt),lat)
axes[1,1].plot(plt_vv/(nx*nt),lat)


#plt.savefig("/home/cliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/11_2017/rh91_350K.pdf",format='pdf')
plt.show()


    

