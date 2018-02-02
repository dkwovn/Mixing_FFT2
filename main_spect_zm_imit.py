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
from plot_funcs import plot_zm_diffu
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=True
PROCESS_FLAG=False

npro=18
yr_st=1980
yr_end=2015
fold_num=16
zlev=3
zz=3
ylev=37
#xlev=0
mask_sig=90.
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
sigLvl = 0.05


def get_spectr(yr):
    #yr_num=0
    #for yr in range(yr_st,yr_end+1):
    print( str(yr))
    df1_name = '/home/cjliu/data/MERRA2/6hourly/ivar.middle.'+str(yr)+'.nc'
    df2_name = '/home/cjliu/data/MERRA2/6hourly/ivar.middle.'+str(yr+1)+'.nc'
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
    
data_fname='/home/cjliu/data/npz/DiffuSpectrRH_zm_'+seasonStr+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
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

nx=lon.shape[0]
ny=lat.shape[0]
nt=360

diffu_spectr = np.mean(diffu_spectr[:, :, :, xlev], 0)
contour_fld = np.mean(u_mn[:, xlev, :, :], 0).T
shading_fld = np.log(diffu_spectr)/np.log(10)
title = "Zonal mean diffusivity"
plot_zm_diffu(lat, lev, shading_fld, contour_fld, title)
##=== plot rh91 plot ===
#grid_x,grid_y = np.meshgrid(cc,lat)
#cs3=fig_ax2.contourf(grid_x,grid_y,rh_fld_trend,cmap=cm.RdBu_r,levels=np.arange(-0.18,0.181,0.02))
#cs4=fig_ax2.contour(grid_x,grid_y,rh_fld,colors='grey',levels=np.arange(5,25,5))
#fig_ax2.clabel(cs4,fmt="%.0f")
#fig_ax2.plot(u_st[:,zz],lat,color='k',linestyle='--')
#fig_ax2.plot(u_end[:,zz],lat,color='k',linestyle='-')
#ax=fig.add_axes([0.16,0.05,0.7,0.01])
#cb0=plt.colorbar(cs3,cax=ax,orientation='horizontal')
#fig_ax2.contourf(grid_x,grid_y,sigLvl-rh_pVal,levels=[-1e9,0,1e9],hatches=[None,'///'],colors='none')

plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/01_2018/diffu_zm_imit_"+seasonStr+'_'+str(yr_st)+"_"+str(yr_end)+".pdf",format='pdf')
plt.show()


    

