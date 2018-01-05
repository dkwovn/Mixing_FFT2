import numpy as np
from caltra import integrate
from caltra import interp as intmod

class traj:
    @staticmethod
    def intgr(lon0,lat0,pres0,lon,lat,pres,ps,u,v,w,t,deltaT,fbFlag,rpos0=0.,rpos1=1.,jump=0,undef=9.99e8,wfactor=1.):
        #lon1=10.
        #lat1=10.
        #pres1=300.
        left=0
        lon_0=lon[0]
        lat_0=lat[0]
        dlon=lon[1]-lon[0]
        dlat=lat[1]-lat[0]
        nx=u.shape[3]
        ny=u.shape[2]
        nz=u.shape[1]
        ps0=ps[t,:,:].flatten()
        ps1=ps[t+fbFlag,:,:].flatten()
        u0=u[t,:,:,:].flatten()
        u1=u[t+fbFlag,:,:,:].flatten()
        v0=v[t,:,:,:].flatten()
        v1=v[t+fbFlag,:,:,:].flatten()
        w0=w[t,:,:,:].flatten()
        w1=w[t+fbFlag,:,:,:].flatten()
        p3d0=pres[t,:,:,:].flatten()
        p3d1=pres[t+fbFlag,:,:,:].flatten()
        #integrate.runge(lon1,lat1,pres1,left,lon0,lat0,pres0,rpos0,rpos1,deltaT,1,jump,undef,wfactor,fbFlag,ps0,ps1,p3d0,p3d1,u0,u1,v0,v1,w0,w1,lon_0,lat_0,dlon,dlat,360.,1,nx,ny,nz)
        lon1,lat1,pres1=integrate.runge(left,lon0,lat0,pres0,rpos0,rpos1,deltaT,1,jump,undef,wfactor,fbFlag,ps0,ps1,p3d0,p3d1,u0,u1,v0,v1,w0,w1,lon_0,lat_0,dlon,dlat,360.,1,nx,ny,nz)
        #print('left=',left)
        return lon1,lat1,pres1

    def intgr2d(lon0,lat0,lon,lat,u,v,t,deltaT,fbFlag,rpos0=0.,rpos1=1.,jump=0,undef=9.99e8):
        left=0
        lon_0=lon[0]
        lat_0=lat[0]
        dlon=lon[1]-lon[0]
        dlat=lat[1]-lat[0]
        nx=u.shape[-1]
        ny=u.shape[-2]
        nz=1
        u0=u[t,:,:].flatten()
        u1=u[t+fbFlag,:,:].flatten()
        v0=v[t,:,:].flatten()
        v1=v[t+fbFlag,:,:].flatten()
        w0=np.zeros(v0.shape)
        p3d0=w0+100.
        ps0=w0+1000.
        pres0=100
        wfactor=0.
        lon1,lat1,pres1=integrate.runge(left,lon0,lat0,pres0,rpos0,rpos1,deltaT,1,jump,undef,wfactor,fbFlag,ps0,ps0,p3d0,p3d0,u0,u1,v0,v1,w0,w0,lon_0,lat_0,dlon,dlat,360.,1,nx,ny,nz)
        return lon1,lat1

    def interp(lon0,lat0,pres0,t,lon,lat,pres_fld,ps_fld,var_fld,mode=3,undef=-9.99e8):
        nx=pres_fld.shape[-1]
        ny=pres_fld.shape[-2]
        nz=pres_fld.shape[-3]
        lon_0=lon[0]
        lat_0=lat[0]
        dlon=lon[1]-lon[0]
        dlat=lat[1]-lat[0]
        tt=int(t)
        reltDt=t-tt
        pres1=pres_fld[tt,:,:,:].flatten()
        pres2=pres_fld[tt+1,:,:,:].flatten()
        ps1=ps_fld[tt,:,:].flatten()
        ps2=ps_fld[tt+1,:,:].flatten()
        var1=var_fld[tt,:,:,:].flatten()
        var2=var_fld[tt+1,:,:,:].flatten()

        xind,yind,zind=intmod.get_index4(lon0,lat0,pres0,reltDt,pres1,pres2,ps1,ps2,3,nx,ny,nz,lon_0,lat_0,dlon,dlat,undef)
        return intmod.int_index4(var1,var2,nx,ny,nz,xind,yind,zind,reltDt,undef)

    def interp2d(lon0,lat0,t,lon,lat,var_fld,mode=3,undef=-9.99e8):
        nx=var_fld.shape[-1]
        ny=var_fld.shape[-2]
        nz=1
        lon_0=lon[0]
        lat_0=lat[0]
        dlon=lon[1]-lon[0]
        dlat=lat[1]-lat[0]
        tt=int(t)
        reltDt=t-tt
        pres1=np.zeros([nz,ny,nx]).flatten()
        ps1=np.zeros([ny,nx]).flatten()+100.
        pres0=0.
        var1=var_fld[tt,:,:].flatten()
        var2=var_fld[tt,:,:].flatten()
        xind,yind,zind=intmod.get_index4(lon0,lat0,pres0,reltDt,pres1,pres1,ps1,ps1,3,nx,ny,nz,lon_0,lat_0,dlon,dlat,undef)
        return intmod.int_index4(var1,var2,nx,ny,nz,xind,yind,zind,reltDt,undef)
        




