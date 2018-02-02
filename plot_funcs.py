
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
## the number of levels = number of colors + 1
col0 = [(0, 0, 0), (100, 100, 100), (180, 180, 180)]
col1 = [(0, 0, 0),
        (26, 0, 17),
        (47, 0, 26),
        (63, 18, 87),
        (65, 52, 142),
        # 5++
        (54, 52, 143),
        (47, 53, 144),
        (30, 78, 157),
        (0, 142, 206),
        (0, 174, 157),
        # 10++
        (0, 166, 95),
        (54, 176, 79),
        (74, 180, 79),
        (116, 188, 76),
        (170, 205, 69),
        # 15++
        (238, 212, 47),
        (249, 184, 50),
        (246, 161, 51),
        (243, 148, 49),
        (242, 135, 48),
        # 20++
        (240, 116, 48),
        (238, 98, 47),
        (235, 81, 47),
        (233, 74, 48),
        (234, 72, 47),
        # 25++
        (233, 68, 48),
        (234, 48, 62),
        (234, 57, 102),
        (229, 70, 150),
        (170, 82, 155)
        ]

def transRGB(source):
    target = []
    for item in source:
        target.append(tuple(np.divide(item,255.0)))
    return target
        

default_levs1 = np.linspace(0, 30, 31)

def plot_local_diffu_NH_col1(lon, lat, shading_fld, contour_fld, title, levs = default_levs1):
    fig,axes=plt.subplots(1,1,figsize=[8,4])
    lon_fld,lat_fld=np.meshgrid(lon,lat)

    plt_map=Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=0,urcrnrlon=360,urcrnrlat=80,ax=axes,resolution='l')
    plt_map.drawcoastlines(linewidth=0.8,color='gray')
    grid_x,grid_y=plt_map(lon_fld,lat_fld)

    #plt_map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],linewidth=0)
    #plt_map.drawmeridians(np.arange(0,360,30),labels=[0,0,0,1],linewidth=0)
    #plt_map.drawparallels(np.arange(-90,90,10),linewidth=0)

    cmap = colors.ListedColormap(transRGB(col1))
    cmap.set_over(transRGB([col1[-1]])[0])
    cmap.set_under(transRGB([col1[0]])[0])
    #bounds = np.linspace(0, 30, 31)
    bounds = levs
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #cs=plt_map.contourf(grid_x,grid_y,shading_fld,cmap=cm.rainbow)
    cs=plt_map.contourf(grid_x,grid_y,shading_fld,cmap=cmap, norm = norm, levels = [-1e8] + list(bounds) + [1e8])
    plt_map.contour(grid_x,grid_y,contour_fld,colors='k',levels=np.arange(10,60.1,10))
    ax=fig.add_axes([0.16,0.05,0.7,0.01])
    cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
    axes.set_title(title)

    #zm=np.mean(shading_fld,1)
    #axes[1].plot(lat,zm)

#=== plot lat lev plot  ===
default_levs2 = np.linspace(4.5, 7., 31)
def plot_zm_diffu(lat, lev, shading_fld, contour_fld, title, levs = default_levs2):
    fig = plt.figure(figsize=[6, 5])
    fig_ax1 = fig.add_axes([0.1,0.16,0.8,0.75])
    grid_x,grid_y=np.meshgrid(lat,lev)
    #levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
    cmap = colors.ListedColormap(transRGB(col1))
    cmap.set_over(transRGB([col1[-1]])[0])
    cmap.set_under(transRGB([col1[0]])[0])
    bounds = levs
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cs2=fig_ax1.contour(grid_x,grid_y,contour_fld,colors='k',levels=[10.,20.,30.,40.])
    cs3=fig_ax1.contour(grid_x,grid_y,contour_fld,colors='k',linestyle = '--', levels=[-10])
    #cs4=fig_ax1.contour(grid_x,grid_y,contour_fld,colors='k',linestyle = '--', levels=[0])
    cs=fig_ax1.contourf(grid_x,grid_y,shading_fld,cmap=cmap, norm = norm,levels=[0] + list(bounds) + [10])
    fig_ax1.set_ylim(300, 450)
    fig_ax1.set_ylabel("potential temperature (K)")
    fig_ax1.set_xlabel("latitude")
    fig_ax1.set_xticks(np.arange(-90, 90.1, 30))
    ax=fig.add_axes([0.16,0.06,0.7,0.01])
    cb0=plt.colorbar(cs,cax=ax,orientation='horizontal')
    fig_ax1.clabel(cs2,fmt="%.0f")
    fig_ax1.clabel(cs3,fmt="%.0f")
    fig_ax1.set_title(title)
    #fig_ax1.contourf(grid_x,grid_y,sigLvl-diffu_pVal,levels=[-1e9,0,1e9],hatches=[None,'////'],colors='none')
    #fig_ax1.invert_yaxis()

def plot_rh(plt_axe, cbar_axe, cc, lat, shading_fld, u_mn, title):
    grid_x,grid_y=np.meshgrid(cc,lat)
    cs=plt_axe.contourf(grid_x,grid_y,shading_fld,cmap=cm.RdBu_r,levels=np.arange(-4,4.1,0.5))
    #cs=axes[0,0].contourf(grid_x,grid_y,plt_fld,cmap=cm.RdBu_r)
    plt_axe.plot(u_mn,lat)
    plt_axe.set_ylim(-80,80)
    plt_axe.set_xlim(-50,50)
    #ax1=fig.add_axes([0.13,0.5,0.35,0.01])
    cb0=plt.colorbar(cs,cax=cbar_axe,extend=0.8,orientation='horizontal')
    plt_axe.set_title(title)

