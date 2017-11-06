#use gaussian weighting to get the spectra along a phase speed line 
#keep the spectra matrix the same shape as data matrix
import numpy as np
from scipy.fftpack import fft, ifft, fft2

a=6.378e6
seconds_in_a_day = 86400
ylev=37
xlev=100
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
    lon_r=None
    lat_r=None
    lon_m=None #lon for mask
    ff=None
    kk=None
    nx=None
    ny=None
    nt=None
    T_len=None
    L_len=None
    def __init__(self,u_in,v_in,lon_in,lat_in):
        self.u = np.flipud(u_in)
        self.v = np.flipud(v_in)
        self.um = np.mean(u_in,0)
        self.vm = np.mean(v_in,0)
        self.lon=lon_in
        self.lat=lat_in
        self.lon_r=np.deg2rad(self.lon)
        self.lat_r=np.deg2rad(self.lat)
        self.nx=lon_in.shape[0]
        self.ny=lat_in.shape[0]
        self.nt=u_in.shape[0]
        self.va=np.zeros(self.v.shape)
        self.ua=np.zeros(self.u.shape)
        self.lon_m=np.linspace(0,360.,self.nx)
        self.vvSpectr=np.zeros([self.nx,self.ny,self.nx])

        self.L_len=a*np.cos(np.deg2rad(lat_in))*2*np.pi #unit: m
        self.T_len=0.25*self.nt*seconds_in_a_day #unit:seconds
        for i in range(self.nx):
            if self.lon_m[i]>180.:
                self.lon_m[i]-=360.
        for t in range(self.nt):
            self.va[t,:,:] = self.v[t,:,:] - self.vm
            self.ua[t,:,:] = self.u[t,:,:] - self.um
        if u_in.shape!=v_in.shape:
            print 'ERROR: shape dismatch!'
        if u_in.ndim!=3 or v_in.ndim!=3:
            print 'ERROR: input dimension must be 3!'
            print 'The shape should be (time,lat,lon)'
        #=== get ff,kk ===
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
                print self.kk
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
                print self.kk
            extr[k]=np.sum(self.Gauss(self.ff,f,dw)*spectr[k,:]*(self.T_len/(2*np.pi)),0)/max(sum(self.Gauss(self.ff,f,dw)),1.)
        return extr

    def get_diffu(self,mask_func):
        vMask=np.zeros([self.nx,self.nt])
        u_mn=np.zeros([self.nx,self.ny])
        diffu_spectr=np.zeros([self.ny,self.nx,self.nx])

        for y in range(self.ny):
            #print 'y=',y
        #for y in range(85,86):
            for x in range(self.nx):
            #for x in range(xlev,xlev+1):
                #==== calculate local average U ====
                #==== replace this U with eddy phase speed sampling, one gets Randel and Held (1991) type calculation ====
                #U=self.localU(x,y,mask_func)
                u_mn[x,y]=self.localU(x,y,mask_func)
    #            U=15.
                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                vSpectr2=fft2(vMask)/(self.nx*self.nt)
                diffu_spectr[y,x,:]=self.extractC_wt2(abs(vSpectr2)**2,u_mn[x,y],y)
                #print np.sum(diffu_spectr[y,x,:],0)
                #self.vvSpectr[x,y,:]=abs(self.extractC(vSpectr2,U,y))**2

        return diffu_spectr,u_mn

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
            print 'y=',y
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
        





