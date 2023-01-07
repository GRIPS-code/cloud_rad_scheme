from os.path import exists
import numpy as np
import tarfile
cm_to_um = 10**4
m_to_micron = 10**6
def read_yang_ice_library(path_ori, habit='solid_column',roughness=50, path_table='./data/'):
# The seven columns are: wavelength (um), maximum dimension of particle size (um), volume of
# particle  (um^3), projected area (um^2), extinction efficiency, single-scattering albedo, and asymmetry 
# factor.
    file_table = [path_table+habit+'_r'+'{:03d}'.format(roughness)+'.nc']
    #if FALSE:#exists(file_table.is_file):
        #print('skip')
        #wavenum_table, r_table, d_table, s_table, v_table, ext_table, scat_table, asy_table = read(file_table)
    if True:
        # extract MIR coefficients
        if not exists(path_table + 'MIR/'+ habit + '/Rough' + '{:03d}'.format(roughness)+'/isca.dat'):
            tar_MIR = tarfile.open(path_ori + 'Rough' + '{:03d}'.format(roughness) + '_' + habit + '.tar.gz')
            tar_MIR.extract('Rough' + '{:03d}'.format(roughness) +'/isca.dat',path_table + 'MIR/'+ habit + '/')
            print('Yang [2013] library is extracted under ./data/MIR')
        freq_raw, d_raw, s_raw, v_raw, ext_table, ssa_table, asy_table = read_isca(path_table + 'MIR/'+ habit + '/Rough' + '{:03d}'.format(roughness)+'/isca.dat')
        # extract FIR coefficients
        if not exists(path_table + 'FIR/'+ habit + '/Rough' + '{:03d}'.format(roughness)+'/isca.dat'):
            tar_FIR = tarfile.open(path_ori + habit + '.tar.gz')
            tar_FIR.extract('Rough' + '{:03d}'.format(roughness) +'/isca.dat', path_table + 'FIR/'+ habit + '/')
            print('Yang [2013] library is extracted under ./data/FIR')
        freq_FIR, d_FIR, s_FIR, v_FIR, ext_FIR, ssa_FIR, asy_FIR = read_isca(path_table + 'FIR/'+ habit + '/Rough' + '{:03d}'.format(roughness)+'/isca.dat')

        freq = np.append(freq_raw,freq_FIR)
        d    = np.append(d_raw,d_FIR) 
        s    = np.append(s_raw,s_FIR) 
        v    = np.append(v_raw,v_FIR) 

        ext_cross_section_table = np.append(ext_table,ext_FIR) * s
        sca_cross_section_table = np.append(ssa_table,ssa_FIR) * ext_cross_section_table

        freq_table, ifreq = np.unique(freq, return_index=True)
        d_table, isize    = np.unique(d, return_index=True) 
        s_table = s[isize] 
        v_table = v[isize] 

        iwav = np.argsort(cm_to_um/freq_table)
        wavenum_table = cm_to_um/freq_table[iwav]

        tmp                     = ext_cross_section_table.reshape((len(ifreq),len(isize)))
        ext_cross_section_table = tmp[iwav,:]
        tmp                     = sca_cross_section_table.reshape((len(ifreq),len(isize)))
        sca_cross_section_table = tmp[iwav,:]
        tmp = np.append(asy_table,asy_FIR)
        tmp0  = tmp.reshape((len(ifreq),len(isize)))
        asy_table  = tmp0[iwav,:]
        r_table = v_table / d_table * 0.75
        #wavenum_table, r_table, d_table, s_table, v_table, ext_table, ssa_table, asy_table = read(file_table)
    return wavenum_table, r_table, d_table, s_table, v_table, ext_cross_section_table, sca_cross_section_table, asy_table

def read_isca(path):
    f       = open(path,"r")
    lines   = f.readlines()
    freq = []
    d = []
    v = []
    s = []
    ext = []
    ssa = []
    asy = []
    for x in lines:
        freq.append(float(x.split('   ')[1])) # wavelength (micron)
        d.append   (float(x.split('   ')[2])) # maximum dimension (micron)
        v.append   (float(x.split('   ')[3])) # volume particle  (micron**3)
        s.append   (float(x.split('   ')[4])) # projected area   (micron**2)
        ext.append (float(x.split('   ')[5])) # extinction efficiency
        ssa.append (float(x.split('   ')[6])) # single-scattering albedo
        asy.append (float(x.split('   ')[7])) # asymmetry factor
    f.close()
    return np.array(freq), np.array(d), np.array(s), np.array(v), np.array(ext), np.array(ssa), np.array(asy)