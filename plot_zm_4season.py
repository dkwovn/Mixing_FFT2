from netCDF4 import Dataset
import numpy as np
from scipy.fftpack import fft, ifft
from diffu_class import mw_diffu,dc,xlev
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
#print 'Multi-wavenumber theory for diffusivity\n'


PROCESS_FLAG=False

npro=8
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
#day_st=237
#day_end=605
day_st=605
day_end=973
#day_st=973
#day_end=973+364



#===== main program =====
seasonList = ["DJF","MAM","JJA","SON"]
fig,axes=plt.subplots(2,2,figsize=[8,8])
for k,seasonStr in enumerate(seasonList):
    data_fname='/home/cjliu/data/npz/DiffuSpectr_zm_'+seasonStr+'_'+str(yr_st)+'_'+str(yr_end)+'.npz'
    data=np.load(data_fname)
    lon=data['lon']
    lat=data['lat']
    lev=data['lev']
    diffu_spectr=data['diffu_spectr']
    u_mn=data['u_mn']
    
    nx=lon.shape[0]
    ny=lat.shape[0]
    nt=360
    
    
    #=== plot lon lat plot ===
    zz=3
    plt_fld=diffu_spectr[zz,:,:]
    
    #lon_fld,lat_fld=np.meshgrid(lon,lat)
    
    
    zm=np.sum(plt_fld,1)/1e5
    i=k%2
    j=int(k/2)
    axes[i,j].plot(lat,zm)
    axes[i,j].set_xlim(-85,85)
    if k==2:
        axes[i,j].set_ylim(0,30)
    else:
        axes[i,j].set_ylim(0,25)
    axes[i,j].set_title(seasonStr)

plt.savefig("/home/cjliu/Documents/RESEARCH/2017/DIFFUSIVITY/FIGURES/12_2017/diffu_spectr_zm_4seasons.pdf",format='pdf')
plt.show()


    

