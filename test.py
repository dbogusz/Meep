import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import math as math
from matplotlib import ticker, cm
from scipy.integrate import quad
from meep import mpb

from meep.materials import Au


Ac = mp.Medium(index=1.8)
PMMA = mp.Medium(index=1.48)
Air = mp.Medium(index=1)
#Au = mp.Medium(index = 0.397)
SiO2 = mp.Medium(index = 1.45)

def system_command(command):
    if mp.am_master():
        os.system(command)

class Bullseye:

    def __init__(self, period, diameter, filling_factor, num, dau, ddev, dsio2):
        self.period = period                        # period of gratings
        self.diameter = diameter                    # diameter of cavity disk
        self.filling_factor = filling_factor        # filling factor of gratings
        self.num = num                              # number of gratings
        self.dau = dau                              # thickness of gold back-reflector
        self.ddev = ddev                            # thickness of device layer (TiO2)
        self.dsio2 = dsio2                          # thickness of SiO2 spacer
        
        self._cellsize = None
        self._geometry = None
        self._sources = None
        self._dimensions = mp.CYLINDRICAL
        self._sim = None

    def define_cellsize(self, pad, dpml):
        self.pad = pad
        self.dpml = dpml

        self.sr = (self.num+1-self.filling_factor)*self.period+self.diameter/2+self.pad+self.dpml             # cell size in r direction     
            
        self.sz = 2*(self.dpml+self.pad)+self.dau+self.ddev+self.dsio2                              # cell size in z direction
                  
        self._cellsize = mp.Vector3(self.sr, 0, self.sz)  # define the cell size

    def define_geometry(self):
        self._geometry = []
        current_layer_position = - 0.5 * self.sz + self.pad + self.dpml

        current_layer_thickness = self.dau
        
        au_film = mp.Block(size=mp.Vector3(mp.inf,0,current_layer_thickness),             # Add Au back reflector
                                  center=mp.Vector3(0,0, current_layer_position + 0.5 * current_layer_thickness),
                                  material=Au)
  
        current_layer_position += current_layer_thickness
        current_layer_thickness = self.dsio2
        
        basis = mp.Block(size=mp.Vector3(mp.inf,0,current_layer_thickness),              # Add SiO2
                     center=mp.Vector3(0,0,current_layer_position + 0.5 * current_layer_thickness),
                     material=SiO2)
       
        current_layer_position += current_layer_thickness

        current_layer_thickness = self.ddev
        device = mp.Block(size=mp.Vector3(mp.inf,0,current_layer_thickness),            # Add TiO2 device layer
                     center=mp.Vector3(0,0,current_layer_position + 0.5 * current_layer_thickness),
                     material=PMMA)
        
       
        self._geometry.append(au_film)
        self._geometry.append(basis)
        self._geometry.append(device)
                
        for n in range (0,self.num+1,+1):                                        # Add N+1 etches (N gratings)
            self._geometry.append(mp.Block(size=mp.Vector3((1-self.filling_factor)*self.period, 0, current_layer_thickness),
                                               center=mp.Vector3(self.diameter/2+(1-self.filling_factor)*self.period/2+n*self.period,0,                                                    current_layer_position + 0.5 * current_layer_thickness), 
                                               material=Air))
            
        self.source_position = current_layer_position 
        
    def define_sources(self, wvl_min=0.6, wvl_max=1, shift=0):
        self.wvl_min = wvl_min
        self.wvl_max = wvl_max
        self.shift = shift

        self.wvl_cen = 0.5 * (self.wvl_min + self.wvl_max)
        self.fmin = 1 / self.wvl_max
        self.fmax = 1 / self.wvl_min
        self.fcen = 0.5 * (self.fmin + self.fmax)
        self.df = self.fmax - self.fmin

        self._sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df),
                                   component=mp.Er,
                                   center=mp.Vector3(0, 0, self.source_position + self.shift),
                                   amplitude=1),
                         mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df),
                                   component=mp.Ep,
                                   center=mp.Vector3(0, 0, self.source_position + self.shift),
                                   amplitude=+1j)
                        ]

    def define_simulation(self, resolution=64, m=+1):
        self.resolution = resolution
        self.m = m

        self._sim = mp.Simulation(cell_size=self._cellsize,
                                  geometry=self._geometry,
                                  boundary_layers=[mp.PML(self.dpml)],
                                  resolution=self.resolution,
                                  sources=self._sources,
                                  dimensions=self._dimensions,
                                  m=self.m,
                                  progress_interval=60,
                                  force_complex_fields=True)

    def define_box_monitors(self, wvl_flux_min=0.6, wvl_flux_max=1, box_dis=1, nfreq_box=200):
        self.wvl_flux_min = wvl_flux_min
        self.wvl_flux_max = wvl_flux_max
        self.box_dis = box_dis
        self.nfreq_box = nfreq_box

        self.fmin_flux = 1 / self.wvl_flux_max
        self.fmax_flux = 1 / self.wvl_flux_min
        self.fcen_flux = 0.5 * (self.fmin_flux + self.fmax_flux)
        self.df_flux = self.fmax_flux - self.fmin_flux

        # bottom surface
        self.box_z1 = self._sim.add_flux(self.fcen_flux, self.df_flux, self.nfreq_box,
                                         mp.FluxRegion(center=mp.Vector3(0, 0, self.source_position + self.shift - self.box_dis),
                                                       size=mp.Vector3(2 * self.box_dis), direction=mp.Z,
                                                       weight=-1))
        # upper surface
        self.box_z2 = self._sim.add_flux(self.fcen_flux, self.df_flux, self.nfreq_box,
                                         mp.FluxRegion(center=mp.Vector3(0, 0, self.source_position +self.shift+ self.box_dis),
                                                       size=mp.Vector3(2 * self.box_dis), direction=mp.Z,
                                                       weight=+1))
        # side surface
        self.box_r = self._sim.add_flux(self.fcen_flux, self.df_flux, self.nfreq_box,
                                        mp.FluxRegion(center=mp.Vector3(self.box_dis, 0, self.source_position +self.shift),
                                                      size=mp.Vector3(z=2 * self.box_dis), direction=mp.R,
                                                      weight=+1))

    def end_simulation(self):
        self._sim.reset_meep()
        self._cellsize = None
        self._geometry = None
        self._sources = None
        self._sim = None

    def plot_epsilon(self):
        self._sim.init_fields()
        # z*r source plane
        self.eps_data = self._sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(2 * self.sr, 0, self.sz),
            component=mp.Dielectric)

        plt.figure(dpi=150)
        plt.imshow(self.eps_data, interpolation='none', origin='lower', cmap="Greys")
        plt.ylabel('z')
        plt.xlabel(r'r $\phi$=0')
        plt.title('Z-R plane')
        plt.tight_layout()
        plt.show()
        
        self.eps_data_rp =self._sim.get_array(center=mp.Vector3(0,0,0), 
                          size=mp.Vector3(2*self.sr,0,0),
                          component=mp.Dielectric)
            
            
        thetas = np.radians(np.arange(0, 360, 1))
        zeniths = np.linspace(0, self.sr, len(self.eps_data_rp))

        values=[]
        for i in range(len(thetas)):
            values.append(self.eps_data_rp)
    
        values=np.array(values)

        r, theta = np.meshgrid(zeniths, thetas)
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),dpi=200)
        ax.contourf(theta, r, values,cmap=cm.Greys_r)
        plt.title('R plane')
        plt.show()

    def get_q_values(self, time_after_sources=150):
        self.time_after_sources = time_after_sources

        # get q values at sources
        harminv_instance = mp.Harminv(mp.Er, mp.Vector3(0, 0, self.source_position+self.shift),
                                      self.fcen, self.df)
        self._sim.run(mp.after_sources(harminv_instance), until_after_sources=self.time_after_sources)

        self.q_results = []
        for mode in harminv_instance.modes:
            self.q_results.append([1000 / mode.freq, mode.decay, mode.Q, abs(mode.amp)])
        self.q_results = np.array(self.q_results)

        self.print_qs()
        
    def get_ldos(self, time_after_sources=150, nfreq_ldos=200):
        self.nfreq_ldos = nfreq_ldos

        self.time_after_sources = time_after_sources
        self.ldos_instance = mp.Ldos(self.fcen, self.df, self.nfreq_ldos)
        
        self._sim.run(mp.dft_ldos(ldos=self.ldos_instance),
                      until_after_sources=self.time_after_sources)

        self.ldos_results = np.transpose(np.array([self.ldos_instance.freqs(), self._sim.ldos_data]))
        
        maximum = max(self.ldos_results[:, 1])
        index = np.where(self.ldos_results[:, 1] == maximum)
        mode_wvl = 1000 / self.ldos_results[index, 0]
        
        self.mode_wvl=mode_wvl
        

    def run_all(self, time_after_sources=150, nfreq_ldos=200):
        self.time_after_sources = time_after_sources
        self.nfreq_ldos = nfreq_ldos

        self.harminv_instance = mp.Harminv(mp.Er, mp.Vector3(0, 0, self.source_position+self.shift), self.fcen, self.df)

        self.ldos_instance = mp.Ldos(self.fcen, self.df, self.nfreq_ldos)

        self._sim.run(mp.after_sources(self.harminv_instance), mp.dft_ldos(ldos=self.ldos_instance),
                      until_after_sources=self.time_after_sources)

        self.ldos_results = np.transpose(np.array([self.ldos_instance.freqs(), self._sim.ldos_data]))

        self.q_results = []
        for mode in self.harminv_instance.modes:
            self.q_results.append([1000 / mode.freq, mode.decay, mode.Q, abs(mode.amp)])
        self.q_results = np.array(self.q_results)

        self.print_qs()
        self.plot_power()
        self.plot_ldos()
        self.plot_er_field()

    def print_qs(self):
        print("-")
        for i in range(len(self.q_results)):
            print("Wavelength in nm:", self.q_results[i, 0])
            print("Decay:", self.q_results[i, 1])
            print("Q factor:", self.q_results[i, 2])
            print("Amplitude:", self.q_results[i, 3])
            print("-")

    def plot_ldos(self):
        maximum = max(self.ldos_results[:, 1])
        index = np.where(self.ldos_results[:, 1] == maximum)
        mode_wvl = 1000 / self.ldos_results[index, 0]

        print('Peak at', mode_wvl, 'nm')

        plt.figure(dpi=150)
        plt.plot(1 / self.ldos_results[:, 0], self.ldos_results[:, 1], 'b-')
        plt.plot(1 / self.ldos_results[index, 0], self.ldos_results[index, 1], 'r.')
        plt.xlabel("Wavelength $\lambda$ ($\mu m$)")
        plt.title("LDOS")
        plt.show()

        return mode_wvl  # structure's fundamental mode wvl

    def plot_power(self):
        flux_freqs = np.array(mp.get_flux_freqs(self.box_z2))
        flux_up = np.array(mp.get_fluxes(self.box_z2))
        flux_bot = np.array(mp.get_fluxes(self.box_z1))
        flux_side = np.array(mp.get_fluxes(self.box_r))
        flux_total = flux_bot + flux_up + flux_side

        flux_wvl = 1 / flux_freqs

        plt.figure(dpi=150)
        plt.plot(flux_wvl, flux_total, 'r-', label='Total emission')
        plt.plot(flux_wvl, flux_bot, 'g-', label='Bottom emission')
        plt.plot(flux_wvl, flux_side, 'y-', label='Side emission')
        plt.plot(flux_wvl, flux_up, 'b-', label='Upward emission')
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Arbitrary intensity')

        return flux_freqs, flux_up, flux_bot, flux_side 
    
    def run_power(self, time_after_sources=100):
        
        self.time_after_sources = time_after_sources

        self._sim.run(until_after_sources=self.time_after_sources)

   
        flux_freqs = np.array(mp.get_flux_freqs(self.box_z2))
        flux_up = np.array(mp.get_fluxes(self.box_z2))
        flux_bot = np.array(mp.get_fluxes(self.box_z1))
        flux_side = np.array(mp.get_fluxes(self.box_r))
        flux_total = flux_bot + flux_up + flux_side

        max_uppower = max(flux_up)
        max_totalpower = max(flux_total)

        max_upindex = np.where(flux_up == max_uppower)
        maxwvl = 1 / flux_freqs[max_upindex]  # find the wavelength of maximum

        sum_upratio = sum(flux_up) / sum(flux_total)  # the ratio of total upward power
        max_upratio = max(flux_up) / max(flux_total)  # at the most productive wavelength, thr ratio of upward power

        flux_wvl = 1 / flux_freqs

        plt.figure(dpi=150)
        plt.axvline(x=maxwvl, color='b', linestyle='--')  # mark where moset productive wavelength
        plt.plot(flux_wvl, flux_total, 'r-', label='Total emission')
        plt.plot(flux_wvl, flux_bot, 'g-', label='Bottom emission')
        plt.plot(flux_wvl, flux_side, 'y-', label='Side emission')
        plt.plot(flux_wvl, flux_up, 'b-', label='Upward emission')
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Arbitrary intensity')

        return flux_freqs, flux_up, flux_bot, flux_side 

    def plot_er_field(self):
        self.er_data = self._sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(2 * self.sr, 0, self.sz),
            component=mp.Er)

        plt.figure(dpi=150)
        plt.imshow(self.er_data.real, interpolation='none', origin='lower')
        plt.ylabel('z')
        plt.xlabel(r'r $\phi$=0')
        plt.title('Z-R plane')
        plt.tight_layout()
        plt.show()


            
    def get_mode_field(self,mode_wvl=[],until_after_sources=100):
        
        self.dev_side_position=self.sr-self.pad-self.dpml
        self.dev_top_position=0.5*self.sz-self.pad-self.dpml


        if self._sim is not None:
            
            self.mode_fcen=[]
            for i in range(len(mode_wvl)):
                self.mode_fcen.append(1/mode_wvl[i])
            self.dft_obj=self._sim.add_dft_fields([mp.Er],self.mode_fcen, self.mode_fcen, 1, 
                                                    where=mp.Volume(center=mp.Vector3((self.dev_side_position)/2,0,self.dev_top_position),
                                                    size=mp.Vector3(self.dev_side_position,0,0)))     
            
            self.until_after_sources=until_after_sources
        
            self._sim.run(until_after_sources=self.until_after_sources)
            
            self.mode_field_data=[]
            
            for i in range(len(self.mode_fcen)):
                self.mode_field_data.append(
                    self._sim.get_dft_array(self.dft_obj,mp.Er,i))

            self.mode_field_data=np.array(self.mode_field_data)

            
    def plot_mode_field(self, mode_wvl=[], until_after_sources=100):
        
        self.get_mode_field(mode_wvl=mode_wvl, until_after_sources=until_after_sources)
        
        for n in range(len(self.mode_fcen)):
                
            thetas = np.radians(np.arange(0, 360, 0.1))
            zeniths = np.linspace(0, self.sr ,len(self.mode_field_data[n]))
            mode_I=np.real(
               np.multiply(np.conj(self.mode_field_data[n]),
                           self.mode_field_data[n])
                          )
                
            values=[]
            for i in range(len(thetas)):
                values.append(mode_I)
    
            values=np.array(values)

            
            r, theta = np.meshgrid(zeniths, thetas)

            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),dpi=150)
            cs=ax.pcolormesh(theta, r, values,cmap=cm.RdYlBu_r,shading='auto')
            fig.colorbar(cs)
            plt.title('|Er|^2 distribution of central disk for mode %1.3f um at device surface' %(1/self.mode_fcen[n]))
            plt.show()

        
    def define_near_field_box(self,wvl_near_min=0.6,wvl_near_max=1,nfreq_near=1,near_box_dis=1):
        
        if self._sim is not None:
            self.wvl_near_min=wvl_near_min
            self.wvl_near_max=wvl_near_max
            self.nfreq_near=nfreq_near
            self.near_box_dis=near_box_dis #distance from the whole structure
        
            self.f_near_min=1/self.wvl_near_max
            self.f_near_max=1/self.wvl_near_min
            self.fcen_near=0.5*(self.f_near_min+self.f_near_max)
            self.df_near=self.f_near_max-self.f_near_min
    
            #upper surface
            self.near_field_z1=self._sim.add_near2far(self.fcen_near,self.df_near,self.nfreq_near,
                            mp.Near2FarRegion(center=mp.Vector3(0, 0, self.source_position + self.shift + self.near_box_dis),
                                              size=mp.Vector3(2 * self.near_box_dis), direction=mp.Z, weight=+1))


    def get_farfield(self, time_after_sources=50, angle=89, npts=1000, ff_res=10):
                                       
        self.time_after_sources = time_after_sources
        self._sim.run(until_after_sources=self.time_after_sources)
            
        self.dis = npts*self.fcen_near  
        self.angle=math.pi*(89/180)            
        size = math.sin(self.angle)*self.dis
        self.ff_res=ff_res
        
            
        self.farfield = self._sim.get_farfields(self.near_field_z1, self.ff_res, center=mp.Vector3(size/2, 0, self.source_position+self.shift+self.dis), size=mp.Vector3(size))
                   
        thetas = np.radians(np.arange(0, 360, 0.1))
        zeniths = np.linspace(0, 180*self.angle/math.pi, len(self.farfield['Ex']))
        values=[]

        self.E2=np.absolute(self.farfield['Ex'])**2+np.absolute(self.farfield['Ey'])**2 +np.absolute(self.farfield['Ez'])**2
       
        for j in range(len(thetas)):
            values.append(self.E2)
        
        self.values = np.array(values)
        self.r,self.theta=np.meshgrid(zeniths, thetas)
                                                      
            
        total = np.sum(self.values[0])
        self.col_eff=[]
        x=[]
        
        self.col=100*np.sum(self.E2)/total
        
        for i in range(0, len(zeniths), 20):
            self.col_eff.append(100*np.sum(self.E2[0:i])/total)
            x.append(zeniths[i])
            
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),dpi=150)
        cs=ax.pcolormesh(self.theta, self.r, self.values, cmap=cm.RdYlBu_r,shading='auto')
        ax.grid(True)
        fig.colorbar(cs)
        plt.title('Far-field emission')
        plt.show()
      
        
        plt.plot(x,self.col_eff)
        plt.ylabel('Collection efficiency %')
        plt.xlabel('Degrees')
        plt.xlim(0,90)
        plt.ylim(0,100)
        
        
        if angle < 89:
            
            self.angle_small=math.pi*(angle/180)          
            size_small = math.sin(self.angle_small)*self.dis
            
            self.farfield_small = self._sim.get_farfields(self.near_field_z1, self.ff_res, center=mp.Vector3(size/2, 0, self.source_position+self.shift+self.dis), size=mp.Vector3(size_small))
                   
            zeniths_small = np.linspace(0, 180*self.angle_small/math.pi, len(self.farfield_small['Ex']))
            values_small=[]

            self.E2_small=np.absolute(self.farfield_small['Ex'])**2+np.absolute(self.farfield_small['Ey'])**2 +np.absolute(self.farfield_small['Ez'])**2
       
            for j in range(len(thetas)):
                values_small.append(self.E2_small)
        
            self.values_small = np.array(values_small)
            self.r_small,self.theta_small=np.meshgrid(zeniths_small, thetas)
            
            
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),dpi=150)
            cs=ax.pcolormesh(self.theta_small, self.r_small, self.values_small, cmap=cm.RdYlBu_r,shading='auto')
            ax.grid(True)
            fig.colorbar(cs)
            plt.title('Far-field emission')
            plt.show()
            
            
        
        
        
        
