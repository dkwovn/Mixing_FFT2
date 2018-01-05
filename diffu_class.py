#use gaussian weighting to get the spectra along a phase speed line 
#keep the spectra matrix the same shape as data matrix
#use cross correlation to get local diffusivity

#Attention: stopped using flipping dimension for diffu_spectr, use ff=-ff, haven't tested yet
import numpy as np
from numpy import exp,log,deg2rad
from scipy.fftpack import fft, ifft, fft2
from traj import traj

a=6.378e6
seconds_in_a_day = 86400
ylev=18
xlev=8
xlev_st=100
xlev_end=130
dw=2.0
dc=2

class mw_diffu:
    u=None
    v=None
    ua=None
    va=None
    um=None
    vm=None
    vvSpectr=None
    lon=None
    lat=None
    dlon=None
    lon_r=None
    lat_r=None
    lon_m=None #lon for mask
    ff=None
    kk=None
    cc=None
    nx=None
    ny=None
    nt=None
    nx_o=None
    ny_o=None
    lon_o=None
    lat_o=None
    T_len=None
    L_len=None
    deltaT=None
    def __init__(self,u_in,v_in,lon_in,lat_in,lon_out=None,lat_out=None):
        #self.u = np.flipud(u_in)
        #self.v = np.flipud(v_in)
        self.u = u_in
        self.v = v_in
        #=== nan out ===
        idx=np.where(abs(self.u)>1e9)
        self.u[idx]=np.nan
        idx=np.where(abs(self.v)>1e9)
        self.v[idx]=np.nan
        #===============
        self.um = np.mean(u_in,0)
        self.vm = np.mean(v_in,0)
        self.lon=lon_in
        self.lat=lat_in
        self.dlon=abs(self.lon[2]-self.lon[1])
        self.lon_r=np.deg2rad(self.lon)
        self.lat_r=np.deg2rad(self.lat)
        self.nx=lon_in.shape[0]
        self.ny=lat_in.shape[0]
        self.nt=u_in.shape[0]
        self.va=np.zeros(self.v.shape)
        self.ua=np.zeros(self.u.shape)
        self.lon_m=np.linspace(0,360.,self.nx)
        self.vvSpectr=np.zeros([self.nx,self.ny,self.nx])
        try:
            if lon_out==None:
                lon_out=lon_in
        except:
            pass
        try:
            if lat_out==None:
                lat_out=lat_in
        except:
            pass
        self.lon_o=lon_out
        self.lat_o=lat_out
        self.nx_o=lon_out.shape[0]
        self.ny_o=lat_out.shape[0]

        self.L_len=a*np.cos(np.deg2rad(lat_in))*2*np.pi #unit: m
        self.T_len=0.25*self.nt*seconds_in_a_day #unit:seconds
        self.deltaT=self.T_len/self.nt
        for i in range(self.nx):
            if self.lon_m[i]>180.:
                self.lon_m[i]-=360.
        for t in range(self.nt):
            self.va[t,:,:] = self.v[t,:,:] - self.vm
            self.ua[t,:,:] = self.u[t,:,:] - self.um
        if u_in.shape!=v_in.shape:
            print( 'ERROR: shape dismatch!')
        if u_in.ndim!=3 or v_in.ndim!=3:
            print( 'ERROR: input dimension must be 3!')
            print( 'The shape should be (time,lat,lon)')
        #=== get ff,kk, cc ===
        self.kk=np.zeros(self.nx)
        self.ff=np.zeros(self.nt)

        for i in range(self.nx):
            if i<=abs(i-self.nx):
                self.kk[i]=i
            else:
                self.kk[i]=i-self.nx
        
        for j in range(self.nt):
            if j<=abs(j-self.nt):
                self.ff[j]=j
            else:
                self.ff[j]=j-self.nt

        #self.cc=np.arange(-50,50,dc)*1.0
        self.cc=np.arange(0,50,dc)*1.0
        #=== have not tested it yet ===
        self.ff=-1*self.ff

    def get_uv_xt(self):
        result=np.zeros(self.ny)
        for y in range(self.ny):
            result[y]=np.sum(self.ua[:,y,:]*self.va[:,y,:])
        return result

    def get_vv_xt(self):
        result=np.zeros(self.ny)
        for y in range(self.ny):
            result[y]=np.sum(self.va[:,y,:]*self.va[:,y,:])
        return result

    def mask_ave(self,var,mask_func):
        pass
#       return a real

    def test_mask(self,mask_func):
        vMask=np.zeros(self.nx)
        for t in range(1):
            for y in range(50,51):
                for x in range(xlev,xlev+1):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
        return vMask

    def test_diffu(self,mask_func):
        vMask=np.zeros([self.nx,self.nt])
        for y in range(ylev,ylev+1):
            for x in range(xlev,xlev+1):
                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                vSpectr2=fft2(vMask)
                vSpectr_k=fft(vMask.T)
                #vSpectr_kw=fft(vSpectr_k)
        return vSpectr2,vSpectr_k

    def localU(self,x,y,mask_func):
        uMask=np.zeros([self.nx])
        #==== obtain local U ====
        for xx in range(self.nx):
            uMask[x+xx-self.nx]=self.um[y,x+xx-self.nx]*mask_func[xx]
        return np.sum(uMask,0)/np.sum(mask_func)
        #========================

    def extractC(self,spectr,c,j):
        nk=spectr.shape[0]
        nf=spectr.shape[1]
        ff=np.arange(nf)
        extr=np.zeros(int(nk/2))
        for k in range(nk/2):
            f=self.uk2f(c,k,j)
            extr[k]=np.interp(f,ff,spectr[k,:],left=0,right=0)
        return extr
            

    def extractC_wt(self,spectr,c,j):
        extr=np.zeros(self.nx)
        for k in range(self.nx):
            try:
                f=self.uk2f(c,self.kk[k],j)
            except:
                print( self.kk)
            #extr[k]=np.interp(f,ff,spectr[k,:],left=0,right=0)*(k/(a*np.cos(np.deg2rad(self.lat[j]))))
            #extr[k]=sum(self.Gauss(self.ff,f,dw)*spectr[k,:]*(self.T_len/(2*np.pi)*k/(a*np.cos(np.deg2rad(self.lat[j])))),0)/(np.pi*2*dw**2)**0.5
            extr[k]=sum(self.Gauss(self.ff,f,dw)*spectr[k,:]*(self.T_len/self.L_len[j]*abs(self.kk[k])),0)/max(sum(self.Gauss(self.ff,f,dw)),1.)

        return extr

    def extractC_wt2(self,spectr,c,j):
        extr=np.zeros(self.nx)
        for k in range(self.nx):
            try:
                f=self.uk2f(c,self.kk[k],j)
            except:
                print( self.kk)
            extr[k]=np.sum(self.Gauss(self.ff,f,dw)*spectr[k,:]*(self.T_len/(2*np.pi)),0)/max(sum(self.Gauss(self.ff,f,dw)),1.)
        return extr


    #def get_diffu(self,mask_func):
    def get_diffu_spectr(self,mask_func_org,option='lonRes'):
        vMask=np.zeros([self.nx,self.nt])
        u_mn=np.zeros([self.nx,self.ny])
        diffu_spectr=np.zeros([self.ny,self.nx,self.nx])
        u_zm=np.mean(u_mn,0)


        rh_fld = []

        if option=='zm':
            mask_func=mask_func_org*0.+1.
        else:
            mask_func=mask_func_org*1.
        for y in range(self.ny):

            if option=='zm':
                x_list=range(xlev,xlev+1)
            elif option=='lonRes':
                x_list=range(self.nx)

            for x in x_list:
                #==== calculate local average U ====
                if option=='lonRes':
                    u_mn[x,y]=self.um[y,x]
                else:
                    u_mn[x,y]=self.localU(x,y,mask_func)

                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                #vSpectr2=fft2(vMask)/(self.nx*self.nt)
                #diffu_spectr[y,x,:]=self.extractC_wt2(abs(vSpectr2)**2,u_mn[x,y],y)
                vSpectr2=abs(fft2(vMask)/(self.nx*self.nt))**2
                diffu_spectr[y,x,:]=self.extractC_wt2(vSpectr2,u_mn[x,y],y)

        return diffu_spectr,u_mn

    def get_diffu_rh(self,mask_func_org,option='lonRes'):
        vMask=np.zeros([self.nx,self.nt])
        u_mn=np.zeros([self.nx,self.ny])
        diffu_spectr=np.zeros([self.ny,self.nx,self.nx])
        u_zm=np.mean(u_mn,0)

        rh_fld = []

        if option=='zm':
            mask_func=mask_func_org*0.+1.
        else:
            mask_func=mask_func_org*1.
        for y in range(self.ny):

            if option=='zm':
                x_list=range(xlev,xlev+1)
            elif option=='lonRes':
                x_list=range(self.nx)

            for x in x_list:
                #==== calculate local average U ====
                if option=='lonRes':
                    u_mn[x,y]=self.um[y,x]
                else:
                    u_mn[x,y]=self.localU(x,y,mask_func)

                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                vSpectr2=abs(fft2(vMask)/(self.nx*self.nt))**2
                diffu_spectr[y,x,:]=self.extractC_wt2(vSpectr2,u_mn[x,y],y)
                rh_fld.append(self.randel_held(vSpectr2,y))

        return diffu_spectr,u_mn,np.array(rh_fld)
    #=== use autocorrelation to get diffu ===
    def get_diffu_grid3(self,tLag,dlon):
        #dlon=half length of lon window
        xWidth=int(dlon/self.dlon)
        corr=np.zeros([2*tLag+1,2*xWidth+1])
        diffu_grid=np.zeros([self.ny,self.nx])
        nn=self.nt-2*tLag
        for x in range(self.nx):
        #for x in range(xlev,xlev+1):
            print('x=',x)
            for y in range(self.ny):
        #    for y in range(ylev,ylev+1):
                ts0=self.va[tLag:self.nt-tLag,y,x]
                for xx in range(-xWidth,xWidth+1):
                    corr[:,xx+xWidth]=np.correlate(self.va[:,y,(x+xx)%self.nx],ts0)/nn
                corr_U=self.extractU(corr,self.um[y,x],y)
                diffu_grid[y,x]=np.sum(corr_U)*self.deltaT

        return diffu_grid

    def get_diffu_auto_wt(self,tLag,dlon,sigmaT):
        #** sigmaT in days 
        #dlon=half length of lon window
        xWidth=int(dlon/self.dlon)
        corr=np.zeros([2*tLag+1,2*xWidth+1])
        diffu_grid=np.zeros([self.ny,self.nx])
        nn=self.nt-2*tLag
        #==== define weight function ====
        ttLag=np.arange(-tLag,tLag+0.1,1)/4.
        wt_func=self.Gauss(ttLag,0.,sigmaT)
        #wt_tt=sum(wt_func)
        #================================
        for x in range(self.nx):
        #for x in range(xlev,xlev+1):
            #print('x=',x)
            for y in range(self.ny):
        #    for y in range(ylev,ylev+1):
                ts0=self.va[tLag:self.nt-tLag,y,x]
                for xx in range(-xWidth,xWidth+1):
                    corr[:,xx+xWidth]=np.correlate(self.va[:,y,(x+xx)%self.nx],ts0)/nn
                corr_U=self.extractU(corr,self.um[y,x],y)
                diffu_grid[y,x]=np.sum(corr_U*wt_func)*self.deltaT

        return diffu_grid

    def get_diffu_grid3_test(self,tLag,dlon):
        #dlon=half length of lon window
        xWidth=int(dlon/self.dlon)
        corr=np.zeros([2*tLag+1,2*xWidth+1])
        diffu_grid=np.zeros([self.ny,self.nx])
        #for x in range(self.nx):
        for x in range(xlev,xlev+1):
            #for y in range(self.ny):
            for y in range(ylev,ylev+1):
                ts0=self.va[tLag:self.nt-tLag,y,x]
                for xx in range(-xWidth,xWidth+1):
                    corr[:,xx+xWidth]=np.correlate(self.va[:,y,x+xx],ts0)
                    corr_U=self.extractU(corr,self.um[y,x],y)
        return corr,corr_U

    def extractU(self,corr,U,j):
        nt=corr.shape[0]
        nx=corr.shape[1]
        xWidth=int((nx-1)/2)
        tLag=int((nt-1)/2)
        xVal=np.arange(-xWidth,xWidth+1,1)
        result=np.zeros(nt)
        result[tLag]=corr[tLag,xWidth]
        x_prev=0
        for tt in range(tLag):
            x_next=self.nextLon_u0(x_prev,U,j)
            result[tLag-(tt+1)]=np.interp(x_next,xVal,corr[tLag-(tt+1),:],left=0,right=0)
            x_prev=x_next

        x_prev=0
        for tt in range(tLag):
            x_next=self.nextLon_u0(x_prev,U,j,'f')
            result[tLag+(tt+1)]=np.interp(x_next,xVal,corr[tLag+(tt+1),:],left=0,right=0)
            x_prev=x_next
#
        return result #return a time series based on U_mean
    #========================================

    def get_diffu_grid(self,tLag):
        diffu_grid=np.zeros([self.ny,self.nx,self.nt-2*tLag])
        for x in range(self.nx):
        #for x in range(xlev,xlev+1):
            print(x)
            for y in range(self.ny):
                for t in range(tLag,self.nt-tLag):
                    diffu_grid[y,x,t-tLag]=np.sum(self.va[t,y,x]*self.collectVal(x,y,t,tLag))*self.deltaT
        print(diffu_grid.shape)
        return diffu_grid
        
    def get_diffu_grid2(self,tLag):
        diffu_grid=np.zeros([self.ny,self.nx,self.nt-2*tLag,2*tLag+1])
        #for x in range(self.nx):
        for x in range(xlev,xlev+1):
            print(x)
            #for y in range(self.ny):
            for y in range(ylev,ylev+1):
                for t in range(tLag,self.nt-tLag):
                    diffu_grid[y,x,t-tLag,:]=self.va[t,y,x]*self.collectVal(x,y,t,tLag)*self.deltaT
        return np.mean(diffu_grid,2)

#    def get_diffu_grid3(self,tLag):

    def nextLon(self,lon,y,t,option='b'):
        #x can be a real number#
        if option=='b':
            fb=-1
        elif option=='f':
            fb=1
        else:
            print('Wrong option for nextLon')
            quit()
        return lon+fb*self.m2deg_x(self.interpVal(lon,y,t,'um')*self.deltaT,y)

    def nextLon_u0(self,lon,u0,y,option='b'):
        #x can be a real number#
        if option=='b':
            fb=-1
        elif option=='f':
            fb=1
        else:
            print('Wrong option for nextLon')
            quit()
        return lon+fb*self.m2deg_x(u0*self.deltaT,y)

    def m2deg_x(self,m,y):
        return m/self.L_len[y]*360.

    def nextX_u0(self,x,u0,y,option='b'):
        if option=='b':
            fb=-1
        elif option=='f':
            fb=1
        else:
            print('Wrong option for nextLon')
            quit()
        return x+(fb*self.m2deg_x(u0*self.deltaT,y))/self.dlon
        

    def interpVal(self,lon,y,t,var='va'):
        if var=='va':
            var_arr=self.va[t,y,:]
            #result= np.interp(lon,self.lon,self.va[t,y,:],period=360.)
        elif var=='ua':
            var_arr=self.ua[t,y,:]
            #result= np.interp(lon,self.lon,self.ua[t,y,:],period=360.)
        elif var=='um':
            var_arr=self.um[y,:]
            #result= np.interp(lon,self.lon,self.um[y,:],period=360.)
        elif var=='lon':
            var_arr=self.lon
            #result= np.interp(lon,self.lon,self.lon[:],period=360.)
        else:
            print("Wrong parameter for var!")
            quit()

        result= np.interp(lon,self.lon,var_arr,period=360.)

        return result

    def collectVal(self,x,y,t,tLag,var='va'):
        result=np.zeros(1+2*tLag)
        result[tLag]=self.interpVal(self.lon[x],y,t,var)
        lon_prev=self.lon[x]
        for tt in range(tLag):
         #   lon_next=self.nextLon(lon_prev,y,t-tt)
            lon_next=self.nextLon_u0(lon_prev,self.um[y,x],y)
            result[tLag-(tt+1)]=self.interpVal(lon_next,y,t-tt-1,var)
            lon_prev=lon_next

        lon_prev=self.lon[x]
        for tt in range(tLag):
            #lon_next=self.nextLon(lon_prev,y,t+tt,'f')
            lon_next=self.nextLon_u0(lon_prev,self.um[y,x],y,'f')
            result[tLag+(tt+1)]=self.interpVal(lon_next,y,t+tt+1,var)
            lon_prev=lon_next

        return result #return a time series based on U_mean

    #==== cal diffu using 2D trajectories ====
    def get_diffu_traj(self,tLag):
        diffu_traj=np.zeros([self.nt-2*tLag,self.ny_o,self.nx_o])
        #print(self.nx_o,self.ny_o)
        for x in range(self.nx_o):
            print('x=',x)
            for y in range(self.ny_o):
                print('y=',y)
                for t in range(tLag,self.nt-tLag):
        #            diffu_traj[t-tLag,y,x]=self.va[t,y,x]*np.sum(self.trajVal(self.lon_o[x],self.lat_o[y],t,tLag))
                    diffu_traj[t-tLag,y,x]=traj.interp2d(self.lon_o[x],self.lat_o[y],t,self.lon,self.lat,self.va)*np.sum(self.trajVal(self.lon_o[x],self.lat_o[y],t,tLag))
        return np.mean(diffu_traj,0)*self.deltaT

    def trajVal(self,lon0,lat0,t,tLag,var='va'):
        # do forward and backward trajectories with length of tLag 
        result=np.zeros(2*tLag+1)
        if var=='va':
            varFld=self.va
        result[tLag]=traj.interp2d(lon0,lat0,t,self.lon,self.lat,varFld)
        #==== debug code ====
        #um_temp=np.zeros(self.u.shape)
        #for t in 
        #um_temp[:]
        #====================

        for fbFlag in [-1,1]:    
            lon1=lon0
            lat1=lat0
            for tt in range(tLag):
                lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u,self.v,t+fbFlag*tt,self.deltaT,fbFlag)
            #    lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u,self.v*0.,t+fbFlag*tt,self.deltaT,fbFlag)
        #        lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u*0.+self.um,self.v*0.+self.vm,t+fbFlag*tt,self.deltaT,fbFlag)
                lon1=lon2
                lat1=lat2
                result[tLag+fbFlag*(tt+1)]=traj.interp2d(lon2,lat2,t+fbFlag*(tt+1),self.lon,self.lat,varFld)
        return result

    def trajLonLat(self,lon0,lat0,t,tLag):
        # do forward and backward trajectories with length of tLag 
        result=np.zeros([2*tLag+1,2])
        result[tLag,0]=lon0
        result[tLag,1]=lat0


        for fbFlag in [-1,1]:    
            lon1=lon0
            lat1=lat0
            for tt in range(tLag):
                lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u,self.v,t+fbFlag*tt,self.deltaT,fbFlag)
                lon1=lon2
                lat1=lat2
                result[tLag+fbFlag*(tt+1),0]=lon2
                result[tLag+fbFlag*(tt+1),1]=lat2
        return result[:,0],result[:,1]
#        lon1=lon0
#        lat1=lat0
#        for tt in range(tLag):
#            lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u,self.v,t-tt,self.deltaT,-1)
#            lon1=lon2
#            lat1=lat2
#            result[tLag-(tt+1)]=traj.interp2d(lon2,lat2,t-(tt+1),self.lon,self.lat,varFld)
#
#        lon1=lon0
#        lat1=lat0
#        for tt in range(tLag):
#            lon2,lat2=traj.intgr2d(lon1,lat1,self.lon,self.lat,self.u,self.v,t+tt,self.deltaT,1)
#            lon1=lon2
#            lat1=lat2
#            result[tLag+(tt+1)]=traj.interp2d(lon2,lat2,t+(tt+1),self.lon,self.lat,varFld)


        
    #=========================================

    def get_randel_held(self,mask_func):
        cc=np.arange(-50,50,dc)*1.0
        c_num=cc.shape[0]
        vMask=np.zeros([self.nx,self.nt])
        uMask=np.zeros([self.nx,self.nt])
        Spectr_yc=np.zeros([self.nx,self.ny,c_num])
        Spectr_yc2=np.zeros([self.nx,self.ny,c_num])
        ### Spectr_yc: uv, Spectr_yc2:vv
        u_mn=np.zeros([self.nx,self.ny])

        for y in range(self.ny):
        #for y in range(ylev,ylev+1):
            #for x in range(self.nx):
            print( 'y=',y)
            for x in range(xlev,xlev+1):
                #==== calculate local average U ====
                u_mn[x,y]=self.localU(x,y,mask_func)
                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                        uMask[x+xx-self.nx,t]=self.ua[t,y,x+xx-self.nx]*mask_func[xx]
                vSpectr2=fft2(vMask)/(self.nx*self.nt)
                uSpectr2=fft2(uMask)/(self.nx*self.nt)
                #vSpectr2=fft2(vMask)
                #uSpectr2=fft2(uMask)
                #uvSpectr2=fft2(uMask*vMask)
                for n,c in enumerate(cc):
                    extr=self.extractC_wt(np.real(uSpectr2*np.conj(vSpectr2)),c,y)
                    #extr[12:]=0
                    Spectr_yc[x,y,n]=np.sum(extr,0)
                    Spectr_yc2[x,y,n]=np.sum(self.extractC_wt(abs(vSpectr2)**2,c,y),0)
        
                #self.vvSpectr[x,y,:]=abs(self.extractC(vSpectr2,U,y))**2
        return Spectr_yc,Spectr_yc2,u_mn

    def randel_held(self,spectr2,y):
        #cc=np.arange(-50,50,dc)*1.0
        c_num=self.cc.shape[0]

        Spectr_c=np.zeros([c_num])
        for n,c in enumerate(self.cc):
        #    Spectr_c[x,n]=np.sum(self.extractC_wt(abs(vSpectr2)**2,c,y),0)
            Spectr_c[n]=np.sum(self.extractC_wt(spectr2,c,y),0)
        return Spectr_c

    def frequency(self):
        freq=np.zeros(self.nt)
        for i in range(self.nt):
            freq[i] = i/self.T_len
        return freq

    def wave_number(self,j):
        wn=np.zeros(self.nx)
        for i in range(self.nx):
            wn[i]= i*2*np.pi/self.L_len[j]
        return wn

    def uk2f(self,u,k,j):
        #u_tmp=abs(u)*seconds_in_a_day
        #f=k*self.T_len/self.L_len[j]*u_tmp
        f=k*self.T_len/self.L_len[j]*u
        return f
                

    def xt2kw(self,j):
        pass

    @staticmethod
    def testFFT(series):
        return fft(series)

    @staticmethod
    def Gauss(xx,x0,dx):
        return np.exp(-(xx-x0)**2/(2*dx**2))

     
    def genMaskRect(self,fold):
        # lon_len is half length
        xl=int((self.nx/fold-1)/2)
        maskFunc = np.zeros(self.nx)
        maskFunc[0] = 1.
        for i in range(xl+1):
            maskFunc[i] = 1.
            maskFunc[-i] = 1.
        return maskFunc

    def genMaskGauss(self,sig_lon):
        maskFunc = np.zeros(self.nx)
        maskFunc = np.exp(-(self.lon_m**2)/(2*sig_lon**2))
        return maskFunc
        

#    def test_diffu_traj(self,tLag):
#        diffu_traj=np.zeros([self.nt-2*tLag,self.ny,self.nx])
#        corr=np.zeros([self.nt-2*tLag,2*tLag+1,self.ny,self.nx])
#        #for x in range(self.nx):
#        for x in range(xlev,xlev+1):
#            print('x=',x)
#            #for y in range(self.ny):
#            for y in range(ylev,ylev+1):
#                print('y=',y)
#                for t in range(tLag,self.nt-tLag):
#        #            diffu_traj[t-tLag,y,x]=self.va[t,y,x]*np.sum(self.trajVal(self.lon[x],self.lat[y],t,tLag))
#                    corr[t-tLag,:,y,x]=self.va[t,y,x]*self.trajVal(self.lon[x],self.lat[y],t,tLag)
#        #return np.mean(diffu_traj,0)
#        return np.mean(corr,0)


    def test_diffu_traj(self,tLag):
        diffu_traj=np.zeros([self.nt-2*tLag,self.ny_o,self.nx_o])
        corr=np.zeros([self.nt-2*tLag,2*tLag+1,self.ny_o,self.nx_o])
        lon_traj=np.zeros([self.nt-2*tLag,2*tLag+1,self.ny_o,self.nx_o])
        lat_traj=np.zeros([self.nt-2*tLag,2*tLag+1,self.ny_o,self.nx_o])
        print(self.nx_o,self.ny_o)
        #for x in range(self.nx_o):
        for x in range(xlev_st,xlev_end+1):
        #for x in range(xlev,xlev+1):
            print('x=',x)
            #for y in range(self.ny_o):
            for y in range(ylev,ylev+1):
                print('y=',y)
                for t in range(tLag,self.nt-tLag):
        #            diffu_traj[t-tLag,y,x]=self.va[t,y,x]*np.sum(self.trajVal(self.lon_o[x],self.lat_o[y],t,tLag))
                    corr[t-tLag,:,y,x]=traj.interp2d(self.lon_o[x],self.lat_o[y],t,self.lon,self.lat,self.va)*self.trajVal(self.lon_o[x],self.lat_o[y],t,tLag)
                    lon_traj[t-tLag,:,y,x],lat_traj[t-tLag,:,y,x]=self.trajLonLat(self.lon_o[x],self.lat_o[y],t,tLag)
#        return np.mean(corr,0)
        return corr,lon_traj,lat_traj


