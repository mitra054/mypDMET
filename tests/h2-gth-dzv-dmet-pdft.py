import os, sys, re, mcu
import numpy as np
from pyscf import lib
from pyscf.tools import molden
from pyscf import mcscf, mrpt, fci, dft
from pyscf.pbc import gto, scf, df, tools
import pywannier90
from pdmet import dmet
from pdmet.tools import tchkfile, tplot
from mrh.my_pyscf.mcpdft import mcpdft, otfnal


lib.logger.TIMER_LEVEL = lib.logger.INFO

cell = gto.Cell()
cell.atom = '''H 5 5 4; H 5 5 5'''
cell.basis = 'gth-dzv'
cell.spin = 0
#
# Note the extra attribute ".a" in the "cell" initialization.
# .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
#
cell.verbose = 2
cell.max_memory=10000
cell.a = np.eye(3)*10
cell.build()

'''================================'''
''' Build GDF '''
'''================================'''
kmesh = [1, 1, 1]
kpts = cell.make_kpts(kmesh)
if not os.path.exists('gdf.h5'):
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = 'gdf.h5'
    gdf.build()
    
'''================================'''
''' Read the HF wave function'''
'''================================'''
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
khf = scf.KROHF(cell, kpts).density_fit()
khf.with_df._cderi = 'gdf.h5'
khf.exxdiv = None
khf.run()
print("khf mo coeff",khf.mo_coeff)
tchkfile.save_kmf(khf, 'chk_HF')



'''================================'''
''' Contruct MLWFs '''
'''================================'''
kmf = tchkfile.load_kmf(cell, khf, kmesh, 'chk_HF')
num_wann = cell.nao
keywords = \
'''
num_iter = 5000
begin projections
random
H: s
end projections
guiding_centres = .true.
'''
w90 = pywannier90.W90(kmf, cell, kmesh, num_wann, other_keywords=keywords)
w90.kernel()
w90.plot_wf(outfile='./WFs/MLWF')
tchkfile.save_w90(w90, 'chk_w90')
# print("kmf mo coeff",kmf.mo_coeff)

'''================================'''
''' Run MC-PDFT '''
'''================================'''
from pyscf import gto, scf, mcscf
from pyscf.pbc import gto, scf, cc, df
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math


'''================================'''
''' Gamma-point MC-PDFT  '''
'''================================'''
hf = scf.ROHF(cell).density_fit()
hf.with_df._cderi = 'gdf.h5'
hf.exxdiv = None
hf.verbose=1
hf_2=hf
hf.run()
hf.mol
print("hf mo coeff",hf.mo_coeff)

######################## print("randommc.mo_coeff",randommc.mo_coeff)
mc = mcpdft.CASSCF (hf, 'tPBE', 2, 2, grids_level=6)
mc = mc.fix_spin_(shift=0.5, ss=0)
print("mcpdft mo coeff is --------------------------------------",mc.mo_coeff)
############################## mc.fcisolver = csf_solver (cell, smult = 1)
mc.verbose = 3
Vnn = mc._scf.energy_nuc()
mc_dup=mc
mc.kernel ()
dm1s = np.asarray ( mc.make_rdm1s() )
print("dm1s",dm1s)
# tplot.plot_mo_gamma(mc, 'nat')
# molden.from_scf(mc, 'nat.molden')
# mc.get_energy_decomposition()
# mc._scf=hf
# print("mc.mo_occ",mc.mo_occ)
# print("mc.mo_energy",mc.mo_energy)
################################# print ("MC-PDFT module over------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


'''================================'''
''' Run DMET '''
'''================================'''
pdmet = dmet.pDMET(cell, kmf, w90, solver = 'CASPDFT') #pass an hf object (scf.ROHF(cell).density_fit()), not a khf object i.e. scf.KROHF(cell, kpts).density_fit(). scf.KROHF(cell, kpts).density_fit() prints an output type not compatible with slicing.
pdmet.impCluster = [1]
pdmet._impOrbs_threshold = 1.5
# pdmet.bath_truncation = False
pdmet.kmf_chkfile = 'chk_HF'
pdmet.w90_chkfile = 'chk_w90'
pdmet.twoS = 0
pdmet.cas = (2,2)
# pdmet.molist = [0,1]
###pdmet.state_average_ = [0.25]*2
pdmet.e_shift = 0.5
###pdmet.nevpt2_roots = np.arange(4)
###pdmet.nevpt2_nroots = 4
pdmet.initialize()
pdmet.one_shot()
print ("Vnn; In DMET-PDFT VNN is 0 since we use a fake molecule")
print("Add this Vnn to the dmet C-pDFT energy ",Vnn)
print("Occupancy:", pdmet.qcsolver.mc.mo_occ)
pdmet.plot('nat', path='./nat')
#hello hello`
"""

'''================================'''
''' Molecular-point MC-PDFT  '''
'''================================'''


mol2 = cell.to_mol()
hf = scf.ROHF(mol2).density_fit()
# hf.with_df._cderi = 'gdf.h5'
hf.verbose=5
hf_2=hf
hf.run()
print("hf mo coeff",hf.mo_coeff)
######################## print("mo coefficient of hfyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",hf.mo_coeff)
######################## randommc= mcscf.CASSCF(hf, 4, 2)

######################## print("randommc.mo_coeff",randommc.mo_coeff)
mc = mcpdft.CASSCF (hf, 'tPBE', 4, 2, grids_level=6)
mc = mc.fix_spin_(shift=0.5, ss=2)
print("mcpdft mo coeff is --------------------------------------",mc.mo_coeff)
############################## mc.fcisolver = csf_solver (cell, smult = 1)
mc.verbose = 3
Vnn = mc._scf.energy_nuc()
print("Vnn ----------------------------------------------------------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",Vnn)
mc.kernel ()
print("mc.mo_occ",mc.mo_occ)

"""