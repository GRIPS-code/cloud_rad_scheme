import math

import netCDF4 as nc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import gamma

from .spec_util import convert_lblOD_to_band


m_to_micron = 10**6
cm_to_um = 10**4

class optics_var(object):
    def __init__(self, r, s, v, ext, sca, ssa, asy, rau, wavenum=None, band_limit=None):
        self.wavenum = wavenum  # spectral wavenumber, cm-1
        self.band_limit = band_limit  # spectral wavenumber, cm-1
        self.r = r  # effective particle radius, micron
        self.s = s  # particle area, 
        self.v = v  # particle volumn, 
        self.rau = rau # density, g/m**3

        # remove out-of-range values
        ext[ext<0] = 0
        sca[sca<0] = 0
        sca[sca>ext] = ext[sca>ext]
        ssa[ssa<0] = 0
        ssa[ssa>1] = 1
        asy[asy<0] = 0
        asy[asy>1] = 1

        self.ext = ext # extinction coefficient m**2/g
        self.sca = sca # scattering coefficient m**2/g
        self.ssa = ssa # single-scattering albedo unitless
        self.asy = asy # asymmetry factor unitless

    def load_from_nc(file_in):
        ncfile = nc.Dataset(file_in,'r')
        wavenum = ncfile.variables['wavenum'][:]
        r = ncfile.variables['re'][:]
        v = ncfile.variables['v'][:]
        s = ncfile.variables['s'][:]
        rau = ncfile.variables['rau'][:]
        ext = ncfile.variables['ext'][:,:]
        ssa = ncfile.variables['ssa'][:,:]
        asy = ncfile.variables['asy'][:,:]
        sca = ssa * ext
        result = optics_var(r, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum)
        return result

    def write_lut_spectralpoints(self,file_out):
        """generate piecewise-linear fit coefficients and write to a given path
           write look-up-table parameterization
        """
        try:
            nband = len(self.wavenum)
        except:
            nband = np.shape(self.band_limit)[0]
        nr = len(self.r)
        with nc.Dataset(file_out, mode='w', format='NETCDF4_CLASSIC') as ncfile:
            # create Dimension
            ncfile.createDimension('Band', nband)
            ncfile.createDimension('Re', nr)
            ncfile.createDimension('Constant', 1)
            # create Variable
            try:
                wavenum = ncfile.createVariable('wavenum', np.float32, ('Band'))
                wavenum.units = 'cm-1'
                freq = ncfile.createVariable('freq', np.float32, ('Band'))
                freq.units = 'micrometer'
                wavenum[:] = self.wavenum
                freq[:] = cm_to_um/wavenum[:]
            except:
                Band_limits_lwr = ncfile.createVariable('Band_limits_lwr', np.float32, ('Band',))
                Band_limits_lwr.units = 'cm-1'
                Band_limits_upr = ncfile.createVariable('Band_limits_upr', np.float32, ('Band',))
                Band_limits_upr.units = 'cm-1'
                Band_limits_lwr[:] = self.band_limit[:,0]
                Band_limits_upr[:] = self.band_limit[:,1]
            re = ncfile.createVariable('re', np.float32, ('Re'))
            re.units = 'micrometer'
            v = ncfile.createVariable('v', np.float32, ('Re'))
            v.units = 'micrometer**3'
            s = ncfile.createVariable('s', np.float32, ('Re'))
            s.units = 'micrometer**2'
            ext = ncfile.createVariable('ext', np.float32, ('Re','Band'))
            ext.units = 'm**2/g'
            ssa = ncfile.createVariable('ssa', np.float32, ('Re','Band'))
            ssa.units = 'unitless'
            asy = ncfile.createVariable('asy', np.float32, ('Re','Band'))
            asy.units = 'unitless'
            rau = ncfile.createVariable('rau', np.float32, ('Constant'))
            rau.units = 'g/m**3'
            re[:] = self.r
            v[:] = self.v
            s[:] = self.s
            ext[:] = self.ext
            ssa[:] = self.ssa
            asy[:] = self.asy
            rau = self.rau


    def gamma_int(wavenum, a, d_in, v_in, s_in, d_out, dr, ext_cross_section_in, scat_cross_section_in, asy_in, rau):
        """intergrate over given gamma particle size distribution"""
        b = d_out/a # gamma shape parameter beta; pdf = b**a x**(a-1) * exp(-b*x)/Gamma(a)
        nsize = len(d_out)
        nwav = len(wavenum)
        asy = np.zeros((nsize,nwav))
        ext = np.zeros((nsize,nwav)) 
        sca = np.zeros((nsize,nwav))
        v = np.zeros((nsize,))
        s = np.zeros((nsize,))
        for i in range(nsize):
            f = np.zeros(np.shape(scat_cross_section_in))
            f[:,:] = gamma.pdf(d_in,a[i],0,b[i])
            int_f_over_r = np.sum(f[0,:] * dr)
            int_f_over_v = np.sum(f[0,:] * v_in[:] * dr)
            asy[i,:] = np.sum(f * asy_in * scat_cross_section_in * dr, axis=1)/np.sum(f * scat_cross_section_in * dr, axis = 1)
            ext[i,:] = np.sum(f * ext_cross_section_in * dr, axis=1)/int_f_over_v/rau * m_to_micron # m**2/m**3/g*m**3 = m**2/g
            sca[i,:] = np.sum(f * scat_cross_section_in * dr, axis=1)/int_f_over_v/rau * m_to_micron # m**2/m**3/g*m**3 = m**2/g
            v[i] = np.sum(f[0,:] * v_in[:] * dr)/int_f_over_r # m**3
            s[i] = np.sum(f[0,:] * s_in[:] * dr)/int_f_over_r # m**2
        r = v / s * 0.75 
        ssa = sca/ext
        result = optics_var(r, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum)
        return result

    def lognormal_int(wavenum, mu, sigma, dr,d_in,v_in,s_in,ext_cross_section_in, scat_cross_section_in, asy_in,rau):
        """intergrate over given lognormal particle size distribution"""
        nwav = len(wavenum)
        asy = np.zeros((nwav,))
        ext = np.zeros((nwav,)) 
        sca = np.zeros((nwav,))

        r_in = d_in/2 
        f = np.zeros(np.shape(s_in))
        f[:] = lognormpdf(r_in,mu,sigma)
        int_f_over_r = np.sum(f[:] * dr)
        int_f_over_v = np.sum(f[:] * v_in[:] * dr)
        asy[:] = np.sum(f * asy_in * scat_cross_section_in * dr, axis=1)/np.sum(f * scat_cross_section_in * dr, axis = 1)
        ext[:] = np.sum(f * ext_cross_section_in * dr, axis=1)/int_f_over_v/rau * m_to_micron # m**2/m**3/g*m**3 = m**2/g
        sca[:] = np.sum(f * scat_cross_section_in * dr, axis=1)/int_f_over_v/rau * m_to_micron # m**2/m**3/g*m**3 = m**2/g
        v = np.sum(f[:] * v_in[:] * dr)/int_f_over_r # m**3
        s = np.sum(f[:] * s_in[:] * dr)/int_f_over_r # m**2
        r = v / s * 0.75 
        ssa = sca/ext
        return r, s, v, ext, sca, ssa, asy
    
    def interp_cloud_optics(self, wavenum_out):
        """interpolate cloud optics to higher spectral frequency"""
        nsize = len(self.r)
        nwav = len(wavenum_out)
        ext_out = np.zeros((nsize, nwav))
        ssa_out = np.zeros((nsize, nwav))
        asy_out = np.zeros((nsize, nwav))
        for i in range(nsize):
            ext_out[i,:] = interp1d(self.wavenum, self.ext[i,:],fill_value="extrapolate")(wavenum_out)
            ssa_out[i,:] = interp1d(self.wavenum, self.ext[i,:]*self.ssa[i,:],fill_value="extrapolate")(wavenum_out)/ext_out[i,:]
            asy_out[i,:] = interp1d(self.wavenum, self.ext[i,:]*self.ssa[i,:]*self.asy[i,:],fill_value="extrapolate")(wavenum_out)/ext_out[i,:]/ssa_out[i,:]
        sca_out = ssa_out * ext_out
        result = optics_var(self.r, self.s, self.v,
                                  ext_out, sca_out, ssa_out, asy_out,self.rau,
                                  wavenum=wavenum_out)
        return result
    
    def band2wav_cloud_optics(self, wavenum_out):
        """interpolate cloud optics to higher spectral frequency"""
        nsize = len(self.r)
        nwav = len(wavenum_out)
        ext_out = np.zeros((nsize, nwav))
        ssa_out = np.zeros((nsize, nwav))
        asy_out = np.zeros((nsize, nwav))
        sca_out = np.zeros((nsize, nwav))
        for i in range(len(self.band_limit)):
            if i == 0:   # Assume first band covers 0 to band_limit[0]
                id = np.squeeze(np.where(wavenum_out<=self.band_limit[i]))
            else:
                id = np.squeeze(np.where((wavenum_out <= self.band_limit[i]) & (wavenum_out > self.band_limit[i-1])))
            ext_out[:,id] = np.repeat(self.ext[:,i].reshape(nsize,1),len(id),axis=1)
            ssa_out[:,id] = np.repeat(self.ssa[:,i].reshape(nsize,1),len(id),axis=1)
            asy_out[:,id] = np.repeat(self.asy[:,i].reshape(nsize,1),len(id),axis=1)
            sca_out[:,id] = np.repeat(self.sca[:,i].reshape(nsize,1),len(id),axis=1)
        # In case last band limits < last wavenumber
        id = np.arange(id[-1],nwav,1)
        ext_out[:,id] = np.repeat(self.ext[:,-1].reshape(nsize,1),len(id),axis=1)
        ssa_out[:,id] = np.repeat(self.ssa[:,-1].reshape(nsize,1),len(id),axis=1)
        asy_out[:,id] = np.repeat(self.asy[:,-1].reshape(nsize,1),len(id),axis=1)
        sca_out[:,id] = np.repeat(self.sca[:,-1].reshape(nsize,1),len(id),axis=1)

        result = optics_var(self.r, self.s, self.v,
                                  ext_out, sca_out, ssa_out, asy_out,self.rau,
                                  wavenum=wavenum_out)
        return result


    def thin_average(self,source,band_limit):
        """perform thin average"""
        nsize = len(self.r)
        nband = np.shape(band_limit)[0]
        ext_out = np.zeros((nsize, nband))
        sca_out = np.zeros((nsize, nband))
        asy_out = np.zeros((nsize, nband))
        for i in range(nband):
            id_wave = np.where((self.wavenum[:]>=band_limit[i,0]) & (self.wavenum[:]<=band_limit[i,1]))
            ext_out[:,i] = convert_lblOD_to_band(source, self.ext,id_wave)
            sca_out[:,i] = convert_lblOD_to_band(source, self.sca,id_wave)
            asy_out[:,i] = convert_lblOD_to_band(source, self.asy,id_wave)
        ssa_out = sca_out/ext_out
        result = optics_var(self.r, self.s, self.v,
                                  ext_out, sca_out, ssa_out, asy_out, self.rau,
                                  band_limit=band_limit)
        return result


    def thick_average(self,source,band_limit):
        """perform thick average 
           Edwards and Slingo, 1996, https://doi.org/10.1002/qj.49712253107
        """
        nsize = len(self.r)
        nband = np.shape(band_limit)[0]
        ext_out = np.zeros((nsize, nband))
        ssa_out = np.zeros((nsize, nband))
        asy_out = np.zeros((nsize, nband))
        R_ave = np.zeros((nsize, nband))
        s = np.sqrt((1 - self.ssa)/(1 - self.asy*self.ssa))
        s[self.asy*self.ssa==1] = 0
        R = (1 - s)/(1 + s)
        for i in range(nband):
            id_wave = np.where((self.wavenum[:]>=band_limit[i,0]) & (self.wavenum[:]<=band_limit[i,1]))
            ext_out[:,i] = convert_lblOD_to_band(source, self.ext, id_wave)
            R_ave[:,i] = convert_lblOD_to_band(source, R, id_wave)
            asy_out[:,i] = convert_lblOD_to_band(source, self.asy, id_wave)
            ssa_out[:,i] = 4 * R_ave[:,i] /((1 + R_ave[:,i])**2 - asy_out[:,i]*(1 - R_ave[:,i])**2)
        ssa_out[R_ave[:,:]==0] = 0
        sca_out = ssa_out * ext_out
        result = optics_var(self.r, self.s, self.v,
                                  ext_out, sca_out, ssa_out, asy_out, self.rau,
                                  band_limit=band_limit)
        return result


    def create_lut_coeff(self,re_range,v_range,file_out):
        """generate piecewise-linear fit coefficients and write to a given path
           write look-up-table parameterization
        """
        nband = np.shape(self.band_limit)[0]
        nr = np.shape(re_range)[1]
        with nc.Dataset(file_out, mode='w', format='NETCDF4_CLASSIC') as ncfile:
            # create Dimension
            ncfile.createDimension('Band', nband)
            ncfile.createDimension('Re_range', nr)
            # create Variable
            Band_limits_lwr = ncfile.createVariable('Band_limits_lwr', np.float32, ('Band',))
            Band_limits_lwr.units = 'cm-1'
            Band_limits_upr = ncfile.createVariable('Band_limits_upr', np.float32, ('Band',))
            Band_limits_upr.units = 'cm-1'
            Effective_Radius_limits_lwr = ncfile.createVariable('Effective_Radius_limits_lwr', np.float32, ('Re_range',))
            Effective_Radius_limits_lwr.units = 'microns'
            Effective_Radius_limits_upr = ncfile.createVariable('Effective_Radius_limits_upr', np.float32, ('Re_range',))
            Effective_Radius_limits_upr.units = 'microns'
            Particle_Volume_limits_lwr = ncfile.createVariable('Particle_Volume_limits_lwr', np.float32, ('Re_range',))
            Particle_Volume_limits_lwr.units = 'microns**3'
            Particle_Volume_limits_upr = ncfile.createVariable('Particle_Volume_limits_upr', np.float32, ('Re_range',))
            Particle_Volume_limits_upr.units = 'microns**3'

            lut_ext_a = ncfile.createVariable('ext_coef_a', np.float32, ('Re_range','Band'))
            lut_ext_a.units = 'm**2/g'
            lut_ext_b = ncfile.createVariable('ext_coef_b', np.float32, ('Re_range','Band'))
            lut_ext_b.units = 'm**2/g'
            lut_ssa_a = ncfile.createVariable('ssa_coef_a', np.float32, ('Re_range','Band'))
            lut_ssa_a.units = 'unitless'
            lut_ssa_b = ncfile.createVariable('ssa_coef_b', np.float32, ('Re_range','Band'))
            lut_ssa_b.units = 'unitless'
            lut_asy_a = ncfile.createVariable('asy_coef_a', np.float32, ('Re_range','Band'))
            lut_asy_a.units = 'unitless'
            lut_asy_b = ncfile.createVariable('asy_coef_b', np.float32, ('Re_range','Band'))
            lut_asy_b.units = 'unitless'

            # compute & write
            Band_limits_lwr[:] = self.band_limit[:,0]
            Band_limits_upr[:] = self.band_limit[:,1]
            Effective_Radius_limits_lwr[:] = re_range[0,:]
            Effective_Radius_limits_upr[:] = re_range[1,:]
            Particle_Volume_limits_lwr[:] = v_range[0,:]
            Particle_Volume_limits_upr[:] = v_range[1,:]

            for k in range(nr):
                for i in range(nband):
                    id = np.where((self.r>=re_range[0,k]) & (self.r<re_range[1,k]))
                    r_sample = self.r[id]
                    id = np.unique(id)
                    if len(id)==1:
                        lut_ext_a[k,i] = np.squeeze(self.ext[id,i])
                        lut_ext_b[k,i] = 0.
                        lut_ssa_a[k,i] = np.squeeze(self.ssa[id,i])
                        lut_ssa_b[k,i] = 0.
                        lut_asy_a[k,i] = np.squeeze(self.asy[id,i])             
                        lut_asy_b[k,i] = 0.
                    else:
                        f = np.polyfit(r_sample, np.squeeze(self.ext[id,i]), 1)
                        lut_ext_a[k,i] = f[1]  # a + b*r
                        lut_ext_b[k,i] = f[0]

                        f = np.polyfit(r_sample, np.squeeze(self.ssa[id,i]), 1)
                        lut_ssa_a[k,i] = f[1]  # a + b*r
                        lut_ssa_b[k,i] = f[0]

                        f = np.polyfit(r_sample, np.squeeze(self.asy[id,i]), 1)
                        lut_asy_a[k,i] = f[1]  # a + b*r
                        lut_asy_b[k,i] = f[0]

    def create_pade_coeff(self,re_range,re_ref,v_range,file_out):
        """generate Padé approximantsize coefficients and write to a given path
           write look-up-table parameterization
        """
        nband = np.shape(self.band_limit)[0]
        nr = np.shape(re_range)[1]
        with nc.Dataset(file_out, mode='w', format='NETCDF4_CLASSIC') as ncfile:
            # create Dimension
            ncfile.createDimension('Band', nband)
            ncfile.createDimension('Re_range', nr)
            ncfile.createDimension('n', 3)
            ncfile.createDimension('m', 3)
            # create Variable
            Band_limits_lwr = ncfile.createVariable('Band_limits_lwr', np.float32, ('Band',))
            Band_limits_lwr.units = 'cm-1'
            Band_limits_upr = ncfile.createVariable('Band_limits_upr', np.float32, ('Band',))
            Band_limits_upr.units = 'cm-1'
            Effective_Radius_Ref = ncfile.createVariable('Effective_Radius_Ref', np.float32, ('Re_range',))
            Effective_Radius_Ref.units = 'microns'
            Effective_Radius_limits_lwr = ncfile.createVariable('Effective_Radius_limits_lwr', np.float32, ('Re_range',))
            Effective_Radius_limits_lwr.units = 'microns'
            Effective_Radius_limits_upr = ncfile.createVariable('Effective_Radius_limits_upr', np.float32, ('Re_range',))
            Effective_Radius_limits_upr.units = 'microns'
            Particle_Volume_limits_lwr = ncfile.createVariable('Particle_Volume_limits_lwr', np.float32, ('Re_range',))
            Particle_Volume_limits_lwr.units = 'microns**3'
            Particle_Volume_limits_upr = ncfile.createVariable('Particle_Volume_limits_upr', np.float32, ('Re_range',))
            Particle_Volume_limits_upr.units = 'microns**3'

            pade_ext_p = ncfile.createVariable('Pade_ext_p', np.float32, ('n', 'Re_range', 'Band'))
            pade_ext_p.units = 'm**2/g'
            pade_ext_q = ncfile.createVariable('Pade_ext_q', np.float32, ('m', 'Re_range', 'Band'))
            pade_ext_q.units = 'm**2/g'
            pade_ssa_p = ncfile.createVariable('Pade_ssa_p', np.float32, ('n', 'Re_range', 'Band'))
            pade_ssa_p.units = 'unitless'
            pade_ssa_q = ncfile.createVariable('Pade_ssa_q', np.float32, ('m', 'Re_range', 'Band'))
            pade_ssa_q.units = 'unitless'
            pade_asy_p = ncfile.createVariable('Pade_asy_p', np.float32, ('n', 'Re_range', 'Band'))
            pade_asy_p.units = 'unitless'
            pade_asy_q = ncfile.createVariable('Pade_asy_q', np.float32, ('m', 'Re_range', 'Band'))
            pade_asy_q.units = 'unitless'

            # compute & write
            Effective_Radius_Ref[:] = re_ref
            Band_limits_lwr[:] = self.band_limit[:,0]
            Band_limits_upr[:] = self.band_limit[:,1]
            Effective_Radius_limits_lwr[:] = re_range[0,:]
            Effective_Radius_limits_upr[:] = re_range[1,:]
            Particle_Volume_limits_lwr[:] = v_range[0,:]
            Particle_Volume_limits_upr[:] = v_range[1,:]

            re_range[0,0] = re_range[0,0]

            def pade(x, p1, p2, p3, q1, q2, q3):
                return ((p3) + (p2)*x + (p1)*x**2)/((q3) + (q2)*x + (q1)*x**2)
            def pade_abs(x, p1, p2, p3, q1, q2, q3):
                return ((p3) + (p2)*x + (p1)*x**2)/(abs(q3)+ abs(q2)*x + abs(q1)*x**2)
            def pade_2p2q_abs(x, p2, p3, q2,q3):
                return (p3 + p2*x)/(abs(q3) + abs(q2)*x)
            def pade_2p2q(x, p2, p3, q2,q3):
                return (p3 + p2*x)/(q3+ q2*x)
            def piecewise(x,y):
                f1 = (y[-1]-y[0])/(x[-1]-x[0])
                f2 = y[0] - f1*x[0]
                return [f1,f2]
            for k in range(nr):
                for i in range(nband):
                    id = np.where((self.r>=re_range[0,k]) & (self.r<re_range[1,k]))
                    r_sample = self.r[id]
                    try:
                        tmp, cov = curve_fit(pade_abs, r_sample-re_ref[k], np.squeeze(self.ext[id,i]))
                        pade_ext_p[:,k,i] = (tmp[:3])
                        pade_ext_q[:,k,i] = abs(tmp[3:])
                    except:
                        try:
                            tmp, cov = curve_fit(pade, r_sample-re_ref[k], np.squeeze(self.ext[id,i]))
                            pade_ext_p[:,k,i] = (tmp[:3])
                            pade_ext_q[:,k,i] = (tmp[3:])
                        except:
                            tmp, cov = curve_fit(pade_2p2q_abs, r_sample-re_ref[k], np.squeeze(self.ext[id,i]))
                            pade_ext_p[:,k,i] = [0, (tmp[0]), (tmp[1])]
                            pade_ext_q[:,k,i] = [0, abs(tmp[2]), abs(tmp[3])]
                            # too much degree-of-freedom. using linear fit instead
                            #f = piecewise(r_sample, np.squeeze(self.ext[id,i]))
                            #pade_ext_p[:,k,i] = [0, f[0], f[1]]  # a + b*r
                            #pade_ext_q[:,k,i] = [0, 0, 1]

                    try:
                        tmp, cov = curve_fit(pade_abs, r_sample-re_ref[k], 1-np.squeeze(self.ssa[id,i]))
                        pade_ssa_p[:,k,i] = abs(tmp[3:])-(tmp[:3])
                        pade_ssa_q[:,k,i] = abs(tmp[3:])
                    except:
                        # too much degree-of-freedom. using linear fit instead
                        f = piecewise(r_sample, np.squeeze(self.ssa[id,i]))
                        pade_ssa_p[:,k,i] = [0, f[0], f[1]]  # a + b*r
                        pade_ssa_q[:,k,i] = [0, 0, 1]

                    try:
                        tmp, cov = curve_fit(pade_abs, r_sample-re_ref[k], 1-np.squeeze(self.asy[id,i]))
                        pade_asy_p[:,k,i] = abs(tmp[3:])-(tmp[:3])
                        pade_asy_q[:,k,i] = abs(tmp[3:])
                    except:
                        # too much degree-of-freedom. using linear fit instead
                        tmp, cov = curve_fit(pade_2p2q_abs, r_sample-re_ref[k], 1-np.squeeze(self.asy[id,i]))
                        pade_asy_p[:,k,i] = [0, abs(tmp[2])-tmp[0], abs(tmp[3])-tmp[1]]
                        pade_asy_q[:,k,i] = [0, abs(tmp[2]), abs(tmp[3])]
                        #f = np.polyfit(r_sample,np.squeeze(self.asy[id,i]),1)
                        #pade_asy_p[:,k,i] = [0, f[0], f[1]]  # a + b*r
                        #pade_asy_q[:,k,i] = [0, 0, 1]

def lognormpdf(x,mu,sigma):
    y = 1.0/x/sigma/math.sqrt(2*math.pi) * np.exp(-1.0/2.0*(np.log(x/mu)/sigma)**2)
    return y
