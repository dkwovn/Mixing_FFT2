
import numpy as np
from scipy.fftpack import fft, ifft, fft2

a=6.378e6
seconds_in_a_day = 86400
ylev=37

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
    lon_m=None #lon for mask
    nx=None
    ny=None
    nt=None
    nx_half=None
    T_len=None
    L_len=None
    def __init__(self,u_in,v_in,lon_in,lat_in):
        self.u = np.flipud(u_in)
        self.v = np.flipud(v_in)
        self.um = np.mean(u_in,0)
        self.vm = np.mean(v_in,0)
        self.lon=lon_in
        self.lat=lat_in
        self.nx=lon_in.shape[0]
        self.nx_half=int(self.nx/2)
        self.ny=lat_in.shape[0]
        self.nt=u_in.shape[0]
        self.va=np.zeros(self.v.shape)
        self.ua=np.zeros(self.u.shape)
        self.lon_m=np.linspace(0,360.,self.nx)
        self.vvSpectr=np.zeros([self.nx,self.ny,self.nx_half])

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

    def mask_ave(self,var,mask_func):
        pass
#       return a real

    def test_mask(self,mask_func):
        vMask=np.zeros(self.nx)
        for t in range(1):
            for y in range(50,51):
                for x in range(100,101):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
        return vMask

    def test_diffu(self,mask_func):
        vMask=np.zeros([self.nx,self.nt])
        for y in range(ylev,ylev+1):
            for x in range(100,101):
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
        nk=spectr.shape[0]
        nf=spectr.shape[1]
        ff=np.arange(nf)
        #for i in range(nf):


        extr=np.zeros(int(nk/2))
        for k in range(-nk/2,nk/2):
            f=self.uk2f(c,k,j)
            if f<0:
                f+=self.nt
            extr[k]=np.interp(f,ff,spectr[k,:],left=0,right=0)*(k/(a*np.cos(np.deg2rad(self.lat[j]))))
        return extr

    def get_diffu(self,mask_func):
        vMask=np.zeros([self.nx,self.nt])
        #for y in range(self.ny):
        for y in range(85,86):
            #for x in range(self.nx):
            for x in range(100,101):
                #==== calculate local average U ====
                #==== replace this U with eddy phase speed sampling, one gets Randel and Held (1991) type calculation ====
        #        U=self.localU(x,y,mask_func)
                U=15.
                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                vSpectr2=fft2(vMask)
                self.vvSpectr[x,y,:]=self.extractC(abs(vSpectr2),U,y)**2
                #self.vvSpectr[x,y,:]=abs(self.extractC(vSpectr2,U,y))**2

        return

    def get_randel_held(self,mask_func):
        cc=np.arange(1,50)*1.0
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
            #print 'y=',y
            for x in range(100,101):
                #==== calculate local average U ====
                u_mn[x,y]=self.localU(x,y,mask_func)
                for t in range(self.nt):
                    #=== apply mask ===
                    for xx in range(self.nx):
                        vMask[x+xx-self.nx,t]=self.va[t,y,x+xx-self.nx]*mask_func[xx]
                        uMask[x+xx-self.nx,t]=self.ua[t,y,x+xx-self.nx]*mask_func[xx]
                #vSpectr2=fft2(vMask)/(self.nx*self.nt)
                #uSpectr2=fft2(uMask)/(self.nx*self.nt)
                vSpectr2=fft2(vMask)
                uSpectr2=fft2(uMask)
                #uvSpectr2=fft2(uMask*vMask)
                for n,c in enumerate(cc):
                    extr=self.extractC_wt(np.real(uSpectr2*np.conj(vSpectr2)),c,y)
                    #extr[12:]=0
                    Spectr_yc[x,y,n]=np.sum(extr,0)/self.nx
                    Spectr_yc2[x,y,n]=np.sum(self.extractC_wt(abs(vSpectr2)**2,c,y),0)/self.nx
                    #for k in range(self.nx_half):
                        #Spectr_yc[x,y,n]+=self.
        
                #self.vvSpectr[x,y,:]=abs(self.extractC(vSpectr2,U,y))**2
        return Spectr_yc,Spectr_yc2,u_mn

    def frequency(self):
        freq=np.zeros(self.nt/2)
        for i in range(self.nt/2):
            freq[i] = i/self.T_len
        return freq

    def wave_number(self,j):
        wn=np.zeros(self.nx/2)
        for i in range(self.nx/2):
            wn[i]= i*2*np.pi/self.L_len[j]
        return wn

    def uk2f(self,u,k,j):
        #u_tmp=abs(u)*seconds_in_a_day
        #f=k*self.T_len/self.L_len[j]*u_tmp
        f=k*self.T_len/self.L_len[j]*abs(u)
        return f
                

    def xt2kw(self,j):
        pass

    @staticmethod
    def testFFT(series):
        return fft(series)
     
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
        





