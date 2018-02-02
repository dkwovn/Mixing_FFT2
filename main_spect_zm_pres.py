from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc,xlev
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
from scipy.stats import pearsonr
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=True
PROCESS_FLAG=False

sigLvl=0.05
npro=16
yr_st=1980
yr_end=2015
fold_num=16
#zlev=3
zz=7
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
    #df1_name = '/home/cjliu/data/MERRA2/6hourly/ivar2.'+str(yr)+'.nc'
    #df2_name = '/home/cjliu/data/MERRA2/6hourly/ivar2.'+str(yr+1)+'.nc'
    df1_name = '/home/cjliu/data/MERRA2/6hourly/merra2.'+str(yr)+'.nc4'
    df2_name = '/home/cjliu/data/MERRA2/6hourly/merra2.'+str(yr+1)+'.nc4'
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
    
data_fname='/home/cjliu/data/npz/DiffuSpectrRH_zmP_'+seasonStr+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
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
        

    np.savez(data_fname,lon=lon,lat=lat,lev=lev,diffu_spectr=np.array(diffu_spectr),u_mn=np.array(u_mn),rh_fld=np.array(rh_fld),cc=cc)
else:
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    cc=data['cc']
    diffu_spectr=data['diffu_spectr']
    rh_fld=data['rh_fld']
    u_mn=data['u_mn']


def linTrend(var):
    yrs = np.arange(yr_st,yr_end+1)
    d2 = var.shape[1]
    d3 = var.shape[2]
    result = np.zeros([d2,d3])
    pVal = np.zeros([d2,d3])
    for j in range(d2):
        for k in range(d3):
            ts = var[:,j,k]
            if (np.nan not in ts):
            #result[j,k] = np.mean(var[:,j,k])
                result[j,k] = np.polyfit(yrs,ts,1)[0]
                sth, pVal[j,k] = pearsonr(yrs,ts)
    #            print(np.polyfit(yrs,ts,1))
            else:
                result[j,k] = np.nan
                pVal[j,k] = np.nan
    return result, pVal


def linTrend_diff(var):
    yrs = np.arange(yr_st,yr_end+1)
    d2 = var.shape[1]
    d3 = var.shape[2]
    result1 = np.zeros([d2,d3])
    result2 = np.zeros([d2,d3])
    for j in range(d2):
        for k in range(d3):
            ts = var[:,j,k]
            if (np.nan not in ts):
                a = np.polyfit(yrs,ts,1)[0]
                b = np.polyfit(yrs,ts,1)[1]
                result1[j,k] = a*yrs[0] + b
                result2[j,k] = a*yrs[-1] + b
            else:
                result1[j,k] = np.nan
                result2[j,k] = np.nan
    return result1,result2
#=== shapes of variables ===
# diffu_sectr[nt,nz,ny,nx]
# u_mn[nt,nx,ny,nz]
# rh_fld[nt,nz,ny,nc]

fig = plt.figure(figsize=[6,9])
fig_ax1 = fig.add_axes([0.1,0.55,0.8,0.4])
fig_ax2 = fig.add_axes([0.1,0.1,0.8,0.35])
nx=lon.shape[0]
ny=lat.shape[0]
nt=360
#=== plot lat lev plot  ===
grid_x,grid_y=np.meshgrid(lat,lev)
diffu_trend, diffu_pVal = linTrend(diffu_spectr[:,:,:,xlev])
diffu_spectr = np.mean(diffu_spectr[:,:,:,xlev],0)
u_st,u_end = linTrend_diff(u_mn[:,xlev,:,:])
u_mn = np.mean(u_mn,0)
rh_fld_trend, rh_pVal = linTrend(rh_fld[:,zz,:,:])
rh_fld = np.mean(rh_fld[:,zz,:,:],0)
#plt_fld=diffu_spectr[:,:]
levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
cs2=fig_ax1.contour(grid_x,grid_y,u_mn[xlev,:,:].T,colors='k',levels=[10.,20.,30.,40.])
fig_ax1.contour(grid_x,grid_y,grid_y,colors='k',linestyles='--',levels=(lev[zz],))
cs=fig_ax1.contourf(grid_x,grid_y,diffu_trend/1e4,cmap=cm.RdBu_r,levels=np.arange(-5,5.01,0.5))
ax=fig.add_axes([0.16,0.49,0.7,0.01])
cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
fig_ax1.clabel(cs2,fmt="%.0f")
fig_ax1.set_title("Zonal mean diffusivity")
fig_ax1.set_xticks(np.arange(-80,80.1,20.))
fig_ax1.contourf(grid_x,grid_y,sigLvl-diffu_pVal,levels=[-1e9,0,1e9],hatches=[None,'////'],colors='none')
fig_ax1.invert_yaxis()

#=== plot rh91 plot ===
grid_x,grid_y = np.meshgrid(cc,lat)
cs3=fig_ax2.contourf(grid_x,grid_y,rh_fld_trend,cmap=cm.RdBu_r,levels=np.arange(-0.18,0.181,0.02))
cs4=fig_ax2.contour(grid_x,grid_y,rh_fld,colors='grey',levels=np.arange(5,25,5))
fig_ax2.clabel(cs4,fmt="%.0f")
fig_ax2.plot(u_st[:,zz],lat,color='k',linestyle='--')
fig_ax2.plot(u_end[:,zz],lat,color='k',linestyle='-')
ax=fig.add_axes([0.16,0.05,0.7,0.01])
cb0=plt.colorbar(cs3,cax=ax,orientation='horizontal')
fig_ax2.contourf(grid_x,grid_y,sigLvl-rh_pVal,levels=[-1e9,0,1e9],hatches=[None,'///'],colors='none')

plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/12_2017/diffu_rh_zmP_"+seasonStr+str(lev[zz])+'_'+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

