#!/usr/bin/env python -u 
'''
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: Hung Q. Pham <pqh3.14@gmail.com>
'''

import numpy as np
import sys, os, ctypes
from functools import reduce
from pyscf import lib, gto, ao2mo, scf, cc, fci, mcscf, mrpt
from pyscf.mcscf import addons
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

import numpy as np
from scipy import linalg
from mrh.my_dmet import localintegrals
import os, time
import sys, copy
from pyscf import gto, scf, ao2mo, mcscf, fci, lib, dft
import time
from pyscf import dft, ao2mo, fci, mcscf
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft.otfnal import transfnal
from pyscf import gto, scf, mcscf
from pyscf.pbc import gto, scf, cc, df
from pyscf.pbc import gto as cellgto
from pyscf.lib import logger
from pyscf.mcscf.addons import StateAverageFCISolver
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math
from pdmet import localbasis, qcsolvers, diis, helper, df_hamiltonian

class QCsolvers:
    def __init__(self, solver, twoS=0, is_KROHF=False, e_shift=None, nroots=1, state_percent=None, verbose=0, memory=4000):
        self.solver = solver       
        self.state_percent = state_percent
        self.SS =  0.5*twoS*(0.5*twoS + 1)      
        self.twoS = twoS 
        self.e_shift = e_shift

        self._is_KROHF = is_KROHF
        self.mol = gto.Mole()
        self.mol.build(verbose = 0)
        self.mol.atom.append(('S', (0, 0, 0)))
        self.mol.nelectron = 2 + self.twoS
        self.mol.incore_anyway = True
        self.mol.max_memory = memory 
        self.mol.spin = self.twoS  

        # self.cell = cellgto.Cell()
        # self.cell.atom = '''H 5 5 4; H 5 5 5'''
        # self.cell.basis = 'gth-dzv'
        # self.cell.spin = 0
        #
        # Note the extra attribute ".a" in the "cell" initialization.
        # .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
        #
        # self.cell.verbose = 2
        # self.cell.max_memory=10000
        # self.cell.a = np.eye(3)*10
        # self.cell.build()
        
        
        self.xc_omega           = None
        if self.mol.spin == 0 and not self._is_KROHF:        
            self.mf = scf.RHF(self.mol)    
        else:     
            self.mf = scf.ROHF(self.mol)      

        # Replace FCI solver by DMRG solver in CheMPS2 or BLOCK
        if self.solver is 'CASCI':
            self.cas    = None
            self.molist = None   
            self.mo     = None  
            self.mo_nat = None   
            self.mc = mcscf.CASCI(self.mf, 2, 2)
            self.nroots = nroots   
            self.mc.verbose = verbose 
            self.mc.max_memory = memory
            self.mc.natorb = True
        elif self.solver is 'DMRG-CI':
            from pyscf import dmrgscf  
            self.cas    = None
            self.molist = None   
            self.mo     = None  
            self.mo_nat = None       
            self.mc = mcscf.CASCI(self.mf, 2, 2)
            self.nroots = nroots   
            self.mc.verbose = verbose 
            self.mc.max_memory = memory 
            self.mc.natorb = True            
        elif self.solver in ['CASSCF','SS-CASSCF','SA-CASSCF','CASPDFT','SS-CASPDFT','SA-CASPDFT']:
            self.cas    = None
            self.molist = None   
            self.mo     = None 
            self.mo_nat = None   
            self.pdftmc = mcpdft.CASSCF (self.mf, 'tPBE', 2, 2, grids_level=3)           
            self.mc = mcscf.CASSCF(self.mf, 2, 2)
            self.nroots = nroots  
            self.mc.verbose = verbose
            self.mc.max_memory = memory 
            self.mc.natorb = True   
            self.chkfile = None  
            self.pdftmc.verbose = verbose
            self.pdftmc.max_memory = memory 
            self.pdftmc.natorb = True   
        elif self.solver in ['DMRG-SCF', 'SS-DMRG-SCF','SA-DMRG-SCF']:
            from pyscf import dmrgscf 
            self.cas    = None
            self.molist = None   
            self.mo     = None 
            self.mo_nat = None              
            self.mc = mcscf.CASSCF(self.mf, 2, 2)
            self.nroots = nroots  
            self.mc.verbose = verbose 
            self.mc.max_memory = memory 
            self.mc.natorb = True 
        elif self.solver == 'FCI':          
            self.fs = None
            self.fs_conv_tol            = 1e-10   
            self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.verbose = verbose
        elif self.solver == 'SHCI':   
            from pyscf.shciscf import shci 
            self.mch = None
            # self.fs_conv_tol            = 1e-10   
            # self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.mo_coeff = None
            self.verbose = verbose
        elif self.solver == 'DMRG':
            from pyscf import PyCheMPS2
            self.CheMPS2print   = False
            self._D             = [200,500,1000,1000]
            self._convergence   = [1.e-4,1.e-5,1.e-6,1.e-8]
            self.noise          = 0.03
            self.max_sweep      = 100
            self._davidson_tol  = [1.e-3,1.e-4,1.e-5,1.e-6]
            if self.mol.verbose > 0: 
                self.CheMPS2print = True
            else:
                self.CheMPS2print = False                                
        elif self.solver == 'MP2': 
            self.mp2 = None
        elif self.solver == 'RCCSD': 
            self.cc = cc.CCSD(self.mf)
            self.t1 = None
            self.t2 = None
        elif self.solver == 'RCCSD_T': 
            self.cc = cc.CCSD(self.mf)
            self.t1 = None
            self.t2 = None
            self.verbose = verbose
         
    def initialize(self, kmf_ecore, OEI, TEI, JK, DMguess, Norb, Nel, Nimp, chempot=0.0):
        self.kmf_ecore      = kmf_ecore       
        self.OEI            = OEI
        self.TEI            = TEI
        self.FOCK           = OEI + JK
        self.DMguess        = DMguess
        self.Norb           = Norb
        self.Nel            = Nel
        self.Nimp           = Nimp
        chempot_array = np.zeros(Norb)
        chempot_array[:Nimp] = chempot
        self.chempot         = np.diag(chempot_array)
        


#####################################        
########## RHF/ROHF solver ##########
#####################################        
    def HF(self):
        '''
        Restricted open/close-shell Hartree-Fock (RHF/ROHF)
        '''        
        
        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot 
     
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        
        ERHF = self.mf.e_tot
        RDM1 = self.mf.make_rdm1()
        JK   = self.mf.get_veff(None, dm=RDM1) 
        # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
        if self.mol.spin == 0 and not self._is_KROHF:        
            ImpurityEnergy = 0.5*lib.einsum('ij,ij->', RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[:Nimp,:], JK[:Nimp,:])                                                  
        else:         
            ImpurityEnergy_a = 0.5*lib.einsum('ij,ij->', RDM1[0][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[0][:Nimp,:], JK[0][:Nimp,:])        
            ImpurityEnergy_b = 0.5*lib.einsum('ij,ij->', RDM1[1][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[1][:Nimp,:], JK[1][:Nimp,:])
            ImpurityEnergy =  ImpurityEnergy_a + ImpurityEnergy_b     
            RDM1 = RDM1.sum(axis=0)
                
        # Compute total energy        
        e_cell = self.kmf_ecore + ImpurityEnergy  

        return (e_cell, ERHF, RDM1) 
        
        
##################################
########## RCCSD solver ########## 
##################################           
    def MP2(self):
        '''
        MP2 solver
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # MP2 calculation
        self.mp2 = self.mf.MP2()
        ecorr, t2 = self.mp2.kernel()
        EMP2 = self.mf.e_tot + ecorr
        RDM1_mo = self.mp2.make_rdm1(t2=t2)
        RDM2_mo = self.mp2.make_rdm2(t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, EMP2, RDM1)  
        
        
##################################
########## RCCSD solver ########## 
##################################           
    def RCCSD(self):
        '''
        Couple-cluster Single-Double 
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ECCSD = Ecorr + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')        
        RDM1_mo = self.cc.make_rdm1(t1=t1, t2=t2)
        RDM2_mo = self.cc.make_rdm2(t1=t1, t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD, RDM1)            


##################################
########## RCCSD(T) solver ########## 
##################################           
    def RCCSD_T(self):
        '''
        Couple-cluster Single-Double (T) with CCSD RDM 
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None 
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ET = self.cc.ccsd_t()
        ECCSD_T = Ecorr + ET + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')   
        
        # Get CCSD rdm
        RDM1_mo = self.cc.make_rdm1(t1=t1, t2=t2)
        RDM2_mo = self.cc.make_rdm2(t1=t1, t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD_T, RDM1)            
        
##################################
########## RCCSD(T) solver ########## 
##################################           
    def RCCSD_T_slow(self):
        '''
        Couple-cluster Single-Double (T) with full CCSD(T) rdm, very very expensive
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None 
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ET = self.cc.ccsd_t()
        ECCSD_T = Ecorr + ET + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')   
        
        # Get CCSD(T) rdm
        eris = self.cc.ao2mo()      # Consume too much memory, need to be fixed!
        l1, l2 = ccsd_t_lambda.kernel(self.cc, eris, t1, t2, verbose=self.verbose)[1:]
        RDM1_mo = ccsd_t_rdm.make_rdm1(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)
        RDM2_mo = ccsd_t_rdm.make_rdm2(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD_T, RDM1) 
        
#################################           
########## DMRG solver ########## 
#################################           
    @property
    def D(self):
        return self._D
    @D.setter
    def D(self, value):    
        assert isinstance(value, int)
        Dvec = [value//4, value//2, value, value] 
        print("from soler", Dvec)                
        self._D = Dvec
    @property
    def convergence(self):    
        return self._convergence
    @convergence.setter
    def convergence(self, value):    
        converg = [value*1.e4, value*1.e3, value*1.e2, value] 
        self._convergence = converg     
    @property
    def davidson_tol(self):    
        return self._davidson_tol
    @davidson_tol.setter
    def davidson_tol(self, value):    
        dav_tol = [value*1.e3, value*1.e2, value*1.e1, value] 
        self._davidson_tol = dav_tol    
        
    def DMRG(self):
        '''
        Density Matrix Renormalization Group using CheMPS2 library     
        '''    
        
        Norb = self.Norb
        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot                                
                
        # CheMPS2 calculation                
        Initializer = PyCheMPS2.PyInitialize()
        Initializer.Init()
        Group = 0
        orbirreps = np.zeros([Norb], dtype=ctypes.c_int)
        HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
        
        #Feed the 1e and 2e integral (T and V)
        for orb1 in range(Norb):
            for orb2 in range(Norb):
                HamCheMPS2.setTmat(orb1, orb2, FOCKcopy[orb1, orb2])
                for orb3 in range(Norb):
                    for orb4 in range(Norb):
                        HamCheMPS2.setVmat(orb1, orb2, orb3, orb4, self.TEI[orb1, orb3, orb2, orb4]) #From chemist to physics notation        

        assert(self.Nel % 2 == 0)
        TwoS  = self.twoS     
        Irrep = 0
       
        if self.CheMPS2print == False:
            sys.stdout.flush()
            old_stdout = sys.stdout.fileno()
            new_stdout = os.dup(old_stdout)
            devnull = os.open('/dev/null', os.O_WRONLY)
            os.dup2(devnull, old_stdout)
            os.close(devnull)        
        
        Prob  = PyCheMPS2.PyProblem(HamCheMPS2, TwoS, self.Nel, Irrep)
        OptScheme = PyCheMPS2.PyConvergenceScheme(4) # 3 instructions      
        #OptScheme.setInstruction(instruction, reduced virtual dimension D, energy convergence, maxSweeps, noisePrefactor, Davidson residual tolerance)
        OptScheme.set_instruction(0, self._D[0], self._convergence[0], 5  , self.noise,  self._davidson_tol[0])        
        OptScheme.set_instruction(1, self._D[1], self._convergence[1], 5  , self.noise,  self._davidson_tol[1])
        OptScheme.set_instruction(2, self._D[2], self._convergence[2], 5  , self.noise,  self._davidson_tol[2])
        OptScheme.set_instruction(3, self._D[3], self._convergence[3], self.max_sweep, 0.00,  self._davidson_tol[3]) # Last instruction a few iterations without noise

        theDMRG = PyCheMPS2.PyDMRG(Prob, OptScheme)
        EDMRG0 = theDMRG.Solve()  
        theDMRG.calc2DMandCorrelations()
        RDM2 = np.zeros([Norb, Norb, Norb, Norb], dtype=ctypes.c_double)
        for orb1 in range(Norb):
            for orb2 in range(Norb):
                for orb3 in range(Norb):
                    for orb4 in range(Norb):
                        RDM2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation

        RDM1 = lib.einsum('ijkk->ij', RDM2)/(self.Nel - 1)
        
        # Excited state:
        if self.nroots > 1 :      
            theDMRG.activateExcitations(self.nroots - 1)
            EDMRG  = [EDMRG0]            
            RDM1s  = [RDM1]           
            RDM2s  = [RDM2]            
            for state in range(self.nroots - 1): 
                theDMRG.newExcitation(np.abs(EDMRG0));
                EDMRG.append(theDMRG.Solve())   
                theDMRG.calc2DMandCorrelations()  
                rdm2 = np.zeros([Norb, Norb, Norb, Norb], dtype=ctypes.c_double)
                for orb1 in range(Norb):
                    for orb2 in range(Norb):
                        for orb3 in range(Norb):
                            for orb4 in range(Norb):
                                rdm2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation
                                
                rdm1 = lib.einsum('ijkk->ij', rdm2)/(self.Nel - 1)
                RDM1s.append(rdm1)                
                RDM2s.append(rdm2)    

        # theDMRG.deleteStoredMPS()
        theDMRG.deleteStoredOperators()
        del(theDMRG)
        del(OptScheme)
        del(Prob)
        del(HamCheMPS2)
        del(Initializer)    

        if self.CheMPS2print == False:        
            sys.stdout.flush()
            os.dup2(new_stdout, old_stdout)
            os.close(new_stdout)
            
        # Compute energy and RDM1      
        if self.nroots == 1:
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy          
        else: 
            e_cell = []              
            for i in range(self.nroots):                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     RDM1s[i][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state 
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, EDMRG[i], Imp_e, self.SS))                                    
                e_cell.append(Imp_e)                
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1s) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                     
            
        return (e_cell, EDMRG0, RDM1)     

########## FCI solver (not spin-adapted) ##########          
    def FCI(self):
        '''
        FCI solver from PySCF
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot 
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)        
            
        # Create and solve the fci object      
        if self.e_shift == None:        
            self.fs = fci.FCI(self.mf, self.mf.mo_coeff)                
        else:                 
            self.fs = fci.addons.fix_spin_(fci.FCI(self.mf, self.mf.mo_coeff), self.e_shift)
            
        self.fs.verbose = self.verbose
        self.fs.conv_tol       = self.fs_conv_tol  
        self.fs.conv_tol_residual = self.fs_conv_tol_residual             
        self.fs.nroots = self.nroots 
        if self.ci is not None: 
            ci0 = self.ci
        else:
            ci0 = None
        EFCI, fcivec = self.fs.kernel(ci0=ci0)         
        self.ci = fcivec

        # Compute energy and RDM1      
        if self.nroots == 1:
            if not  self.fs.converged: print('           WARNING: The solver is not converged')
            self.SS = self.fs.spin_square(fcivec, self.Norb, self.mol.nelec)[0]
            RDM1_mo , RDM2_mo = self.fs.make_rdm12(fcivec, self.Norb, self.mol.nelec)
            # Transform RDM1 , RDM2 to local basis
            RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
            RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
            RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
            RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)                
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            if not self.fs.converged.any(): print('           WARNING: The solver is not converged')
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, vec in enumerate(fcivec):
                SS = self.fs.spin_square(vec, self.Norb, self.mol.nelec)[0]   
                rdm1_mo , rdm2_mo = self.fs.make_rdm12(vec, self.Norb, self.mol.nelec)
                # Transform rdm1 , rdm2 to local basis
                rdm1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, rdm1_mo)
                rdm1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, rdm1)     
                rdm2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, rdm2_mo)
                rdm2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, rdm2)                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, EFCI[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
               
        return (e_cell, EFCI, RDM1)
        
########## SHCI solver ##########          
    def SHCI(self):
        '''
        SHCI solver from PySCF
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy()  
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)        
            
        # Create and solve the SHCI object      
        mch = shci.SHCISCF(self.mf, self.Norb, self.mol.nelectron)
        mch.fcisolver.mpiprefix = ''
        mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
        mch.fcisolver.sweep_iter = [ 0, 3 ]
        # Setting large epsilon1 thresholds highlights improvement from perturbation.
        mch.fcisolver.sweep_epsilon = [ 1e-3, 0.5e-3 ]

        # Run a single SHCI iteration with perturbative correction.
        # mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
        # mch.fcisolver.epsilon2 = 1e-8
        # shci.writeSHCIConfFile( mch.fcisolver, [self.mol.nelectron/2,self.mol.nelectron/2] , False )
        # shci.executeSHCI( mch.fcisolver )

        # Open and get the energy from the binary energy file shci.e.
        # file1 = open(os.path.join(mch.fcisolver.runtimeDir, "%s/shci.e"%(mch.fcisolver.prefix)), "rb")
        # format = ['d']*1
        # format = ''.join(format)
        # e_PT = struct.unpack(format, file1.read())

        if self.ci is not None: 
            ci0 = self.ci
            mo_coeff = self.mo_coeff
        else:
            ci0 = None
            mo_coeff = None
        e_noPT, e_cas, fcivec, mo_coeff = mch.mc1step(mo_coeff=mo_coeff, ci0=ci0)[:4] 
        ESHCI = e_noPT #TODO: this is not correct, will be modified later
        self.ci = fcivec
        self.mo_coeff = mo_coeff
        
        # Compute energy and RDM1      
        if self.nroots == 1:
            if mch.converged == False: print('           WARNING: The solver is not converged')
            self.SS = mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]  
            RDM1_mo , RDM2_mo = mch.fcisolver.make_rdm12(fcivec, self.Norb, self.mol.nelec)
            # Transform RDM1 , RDM2 to local basis
            RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
            RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
            RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
            RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)                
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            if mch.converged.any() == False: print('           WARNING: The solver is not converged')
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, vec in enumerate(fcivec):
                SS = mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]   
                rdm1_mo , rdm2_mo = mch.fcisolver.make_rdm12(fcivec, self.Norb, self.mol.nelec)
                # Transform rdm1 , rdm2 to local basis
                rdm1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, rdm1_mo)
                rdm1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, rdm1)     
                rdm2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, rdm2_mo)
                rdm2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, rdm2)                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, e[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
               
        return (e_cell, ESHCI, RDM1)
              

#########################################        
##########     CASCI solver    ##########
#########################################          
    def CASCI(self, solver = 'FCI', nevpt2_roots=None, nevpt2_nroots=10):
        '''
        CASCI with FCI or DMRG solver for a multiple roots calculation
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
    
        if self.cas == None:
            cas_nelec = self.Nel
            cas_norb = self.Norb
        else:
            cas_nelec = self.cas[0]
            cas_norb = self.cas[1]

        # Updating mc object from the new mf, mol objects     
        self.mc.mol = self.mol   
        self.mc._scf = self.mf 
        self.mc.ncas = cas_norb
        nelecb = (cas_nelec - self.mol.spin)//2
        neleca = cas_nelec - nelecb
        self.mc.nelecas = (neleca, nelecb)
        ncorelec = self.mol.nelectron - (self.mc.nelecas[0] + self.mc.nelecas[1])     
        assert(ncorelec % 2 == 0)
        self.mc.ncore = ncorelec // 2       
        self.mc.mo_coeff = self.mf.mo_coeff
        self.mc.mo_energy = self.mf.mo_energy
            
        # Define FCI solver
        if solver == 'CheMPS2':      
            self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
        elif solver == 'FCI' and self.e_shift != None:         
            target_SS = 0.5*self.twoS*(0.5*self.twoS + 1)
            self.mc.fix_spin_(shift = self.e_shift, ss = target_SS)                  
    
        self.mc.fcisolver.nroots = self.nroots 
        if self.mo is not None: 
            mo = self.mo
        elif self.molist is not None: 
            mo = mcscf.sort_mo(self.mc, self.mc.mo_coeff, self.molist, 0)
        else: 
            mo = self.mc.mo_coeff
        e_tot, e_cas, fcivec = self.mc.kernel(mo)[:3] 
        if not self.mc.converged: print('           WARNING: The solver is not converged')
        
        # Save mo for the next iterations
        self.mo_nat     = self.mc.mo_coeff           
   
        # Compute energy and RDM1      
        if self.nroots == 1:
            civec = fcivec
            self.SS = self.mc.fcisolver.spin_square(civec, self.Norb, self.mol.nelec)[0]
            RDM1_mo , RDM2_mo = self.mc.fcisolver.make_rdm12(civec, self.Norb, self.mol.nelec)
            
            ###### Get RDM1 + RDM2 #####
            core_norb = self.mc.ncore    
            core_MO = self.mc.mo_coeff[:,:core_norb]
            active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
            casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(self.mc.ci, cas_norb, self.mc.nelecas) #in CAS(MO) space    

            # Transform the casdm1_mo to local basis
            casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
            casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
            coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
            RDM1 = coredm1 + casdm1   
            
            # Transform the casdm2_mo to local basis
            casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
            casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
            casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
            casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
        
            coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
            coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

            effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
            effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                        
            RDM2 = coredm2 + casdm2 + effdm2               
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, civec in enumerate(fcivec):
                SS = self.mc.fcisolver.spin_square(civec, cas_norb, self.mc.nelecas)[0]
                
                ###### Get RDM1 + RDM2 #####
                core_norb = self.mc.ncore     
                core_MO = self.mc.mo_coeff[:,:core_norb]
                active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
                casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(civec, cas_norb, self.mc.nelecas) #in CAS(MO) space    

                # Transform the casdm1_mo to local basis
                casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
                casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
                coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
                rdm1 = coredm1 + casdm1   
                
                # Transform the casdm2_mo to local basis
                casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
                casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
                casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
                casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
            
                coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
                coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

                effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
                effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                            
                rdm2 = coredm2 + casdm2 + effdm2         
                
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, e_tot[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
        
        if nevpt2_roots is not None:
            # Run a CASCI for an excited-state wfn
            if solver == 'FCI' and self.e_shift is not None: 
                mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                mc_CASCI = mc_CASCI.fix_spin_(shift=self.e_shift, ss=target_SS) 
            else:
                mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                
            mc_CASCI.fcisolver.nroots = nevpt2_nroots
            fcivec = mc_CASCI.kernel(self.mc.mo_coeff)[2]

            # Run NEVPT2
            e_casci_nevpt = []
            for root in nevpt2_roots:
                SS = mc_CASCI.fcisolver.spin_square(fcivec[root], cas_norb, self.mc.nelecas)[0]
                e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
                if not isinstance(mc_CASCI.e_tot, np.ndarray):
                    e_CASCI = mc_CASCI.e_tot
                    e_nevpt = e_CASCI + e_corr
                else:
                    e_CASCI = mc_CASCI.e_tot[root]
                    e_nevpt = e_CASCI + e_corr
                e_casci_nevpt.append([SS, e_CASCI, e_nevpt])
                
            #Pack E_CASSCF and E_NEVPT2 into a tuple of e_tot
            e_casci_nevpt = np.asarray(e_casci_nevpt)
            e_tot = (e_tot, e_casci_nevpt)
                
        return (e_cell, e_tot, RDM1)



#########################################        
##########     CASSCF solver    ##########
#########################################  

    def CASSCF(self, solver='FCI', state_specific_=None, state_average_=None, state_average_mix_=None, nevpt2_roots=None, nevpt2_nroots=10, nevpt2_spin=0.0):
        '''
        CASSCF with FCI or DMRG solver:
            - Ground state
            - State-specfic
            - State-average
        state_specific_ is used to pass the state_id to the solver
        state_average_ is used to pass the weights to the solver
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        if self.cas is None:
            cas_nelec = self.Nel
            cas_norb = self.Norb
        else:
            cas_nelec = self.cas[0]
            cas_norb = self.cas[1]

        # Updating mc object from the new mf, mol objects    
        if state_specific_ is not None and state_average_ is None:
            state_id = state_specific_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_specific_(state_id)
        elif state_specific_ is None and state_average_ is not None and state_average_mix_ is None:
            weights = state_average_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_average_(weights)
        elif state_average_mix_ is not None:
            solver1, solver2, weights = state_average_mix_
            mcscf.state_average_mix_(self.mc, [solver1, solver2], weights)
        else:
            state_id = 0
            self.nroots = 1
            self.mc.fcisolver.nroots = self.nroots 
        
        self.mc.mol = self.mol   
        self.mc._scf = self.mf 
        self.mc.ncas = cas_norb
        nelecb = (cas_nelec - self.mol.spin)//2
        neleca = cas_nelec - nelecb
        self.mc.nelecas = (neleca, nelecb)
        ncorelec = self.mol.nelectron - (self.mc.nelecas[0] + self.mc.nelecas[1])     
        assert(ncorelec % 2 == 0)
        self.mc.ncore = ncorelec // 2       
        self.mc.mo_coeff = self.mf.mo_coeff
        self.mc.mo_energy = self.mf.mo_energy
        
        # Define FCI solver
        if solver == 'CheMPS2':      
            self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
        elif solver == 'FCI' and self.e_shift is not None and state_average_mix_ is None:         
            target_SS = 0.5*self.twoS*(0.5*self.twoS + 1)
            self.mc.fix_spin_(shift=self.e_shift, ss=target_SS)                  
            
        if self.mo is not None: 
            mo = self.mo
        elif self.molist is not None: 
            if self.chkfile is not None:
                mo = lib.chkfile.load(self.chkfile, 'mcscf/mo_coeff')
            else:
                mo = self.mc.mo_coeff
            mo = mcscf.sort_mo(self.mc, mo, self.molist, base=0)
        else: 
            mo = self.mc.mo_coeff

        e_tot, e_cas, fcivec = self.mc.kernel(mo)[:3] 
        if state_specific_ is None and state_average_ is not None: 
            e_tot = np.asarray(self.mc.e_states)
            
        if not self.mc.converged: print('           WARNING: The solver is not converged')
        
        # Save mo for the next iterations
        self.mo_nat = self.mc.mo_coeff           
        self.mo = self.mc.mo_coeff  
        
    
        # Compute energy and RDM1      
        if self.nroots == 1 or state_specific_ is not None:
            civec = fcivec
            self.SS, spin_multiplicity = mcscf.spin_square(self.mc)
            
            ###### Get RDM1 + RDM2 #####
            core_norb = self.mc.ncore    
            core_MO = self.mc.mo_coeff[:,:core_norb]
            active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
            casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(civec, cas_norb, self.mc.nelecas) #in CAS(MO) space    

            # Transform the casdm1_mo to local basis
            casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
            casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
            coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
            RDM1 = coredm1 + casdm1   
            #Activate the lower block only when you need casdm2s
            """
            # Transform the casdm2_mo to local basis
            casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
            casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
            casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
            casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
        
            coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
            coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

            effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
            effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                        
            RDM2 = coredm2 + casdm2 + effdm2               
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            """
            ImpurityEnergy = 0.0
            e_cell = self.kmf_ecore + ImpurityEnergy         
            print('       State %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (state_id, e_tot, ImpurityEnergy, self.SS))  
        elif state_average_ is not None:
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            rdm1s, rdm2s = self.mc.fcisolver.states_make_rdm12(fcivec, cas_norb, self.mc.nelecas)
            SSs, spin_multiplicities = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas) 
            
            for i in range(len(weights)):
                SS, spin_multiplicity = SSs[i], spin_multiplicities[i]

                ###### Get RDM1 + RDM2 #####
                core_norb = self.mc.ncore    
                core_MO = self.mc.mo_coeff[:,:core_norb]
                active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
                casdm1_mo, casdm2_mo = rdm1s[i], rdm2s[i]  

                # Transform the casdm1_mo to local basis
                casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
                casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
                coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
                rdm1 = coredm1 + casdm1   
                
                # Transform the casdm2_mo to local basis
                if True: 
                    # this is used to get around with the huge memory to get the IMP nergy which is not necessary for the Gamma-point embedding  
                    # TODO: generalize it 
                    Imp_Energy_state = 0
                else:
                    casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
                    casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
                    casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
                    casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
                
                    coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb], dtype=np.float64)  # TODO: this is impractical for the big embedding space. Lots of memory
                    coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
                    coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

                    effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb], dtype=np.float64)   #
                    effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
                    effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                                
                    rdm2 = coredm2 + casdm2 + effdm2         
                    
                    Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])

                                  
                Imp_e = self.kmf_ecore + Imp_Energy_state  
                if state_average_ is not None:
                    print('       State %d (%5.3f): E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, weights[i], e_tot[i], Imp_e, SS))  

                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)    

            RDM1 = lib.einsum('i,ijk->jk',state_average_, RDM1) 
            e_cell = lib.einsum('i,i->',state_average_, e_cell) 
            # self.SS = tot_SS/self.nroots  
            
        if nevpt2_roots is not None:

            # Run a CASCI for an excited-state wfn
            # if solver == 'FCI' and self.e_shift is not None: 
                # mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                # mc_CASCI = mc_CASCI.fix_spin_(shift=self.e_shift, ss=target_SS) 
            # else:
                # mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                
            self.mf.spin = nevpt2_spin
            nelecb = (cas_nelec - self.mf.spin)//2
            neleca = cas_nelec - nelecb
            nelecas = (neleca, nelecb)
            mc_CASCI = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
            mc_CASCI.fcisolver.nroots = nevpt2_nroots
            fcivec = mc_CASCI.kernel(self.mc.mo_coeff)[2]
            ground_state = fcivec[0]
            
            # Run NEVPT2
            e_casci_nevpt = []
            t_dm1s = []
            from pyscf.fci import cistring
            print("=====================================")
            if len(nevpt2_roots) > len(fcivec): nevpt2_roots = np.arange(len(fcivec))
            for root in nevpt2_roots:
                ci = fcivec[root]
                SS = mc_CASCI.fcisolver.spin_square(ci, cas_norb, nelecas)[0]
                e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
                if not isinstance(mc_CASCI.e_tot, np.ndarray):
                    e_CASCI = mc_CASCI.e_tot
                    e_nevpt = e_CASCI + e_corr
                else:
                    e_CASCI = mc_CASCI.e_tot[root]
                    e_nevpt = e_CASCI + e_corr
                e_casci_nevpt.append([SS, e_CASCI, e_nevpt])
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
                rdm1 = mc_CASCI.fcisolver.make_rdm12(ci, cas_norb, nelecas)[0]
                e, v = np.linalg.eig(rdm1)
                # Find the two SDs with most contribution 
                strsa = np.asarray(cistring.make_strings(range(cas_norb), neleca))
                strsb = np.asarray(cistring.make_strings(range(cas_norb), nelecb))    
                na = len(strsa)
                nb = len(strsb)
                
                idx_1st_max = abs(ci).argmax()
                c1 = ci.flatten()[idx_1st_max]
                stra_1st = strsa[idx_1st_max // nb]
                strb_1st = strsb[idx_1st_max % nb ]
                
                abs_fcivec = abs(ci).flatten()
                abs_fcivec[idx_1st_max] = 0.0
                idx_2nd_max = abs_fcivec.argmax()
                c2 = ci.flatten()[idx_2nd_max]
                stra_2nd = strsa[idx_2nd_max // nb]
                strb_2nd = strsb[idx_2nd_max % nb ]
                
                abs_fcivec[idx_2nd_max] = 0.0
                idx_3rd_max = abs_fcivec.argmax()
                c3 = ci.flatten()[idx_3rd_max]
                stra_3rd = strsa[idx_3rd_max // nb]
                strb_3rd = strsb[idx_3rd_max % nb ]

                abs_fcivec[idx_3rd_max] = 0.0
                idx_4th_max = abs_fcivec.argmax()
                c4 = ci.flatten()[idx_4th_max]
                stra_4th = strsa[idx_4th_max // nb]
                strb_4th = strsb[idx_4th_max % nb ]
                
                print("== State {0:d}: {1:2.4f}|{2:s},{3:s}> + {4:2.4f}|{5:s},{6:s}> + {7:2.4f}|{8:s},{9:s}> + {10:2.4f}|{11:s},{12:s}>".format(root, c1, bin(stra_1st)[2:], bin(strb_1st)[2:], c2, bin(stra_2nd)[2:], bin(strb_2nd)[2:], c3, bin(stra_3rd)[2:], bin(strb_3rd)[2:], c4, bin(stra_4th)[2:], bin(strb_4th)[2:]))
                print("   Occupancy:", e)
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
                
                ''' Calculate Transform density matrix '''
                t_dm1 = mc_CASCI.fcisolver.trans_rdm1(ground_state, ci, mc_CASCI.ncas, mc_CASCI.nelecas)
                # transform density matrix to EO representation
                orbcas = mc_CASCI.mo_coeff[:,mc_CASCI.ncore:mc_CASCI.ncore+mc_CASCI.ncas]
                t_dm1_emb = orbcas @ t_dm1 @ orbcas.T
                t_dm1s.append(t_dm1_emb)
                
            print("=====================================") 
                
            #Pack E_CASSCF and E_NEVPT2 into a tuple of e_tot
            e_casci_nevpt = np.asarray(e_casci_nevpt)
            e_tot = (e_tot, e_casci_nevpt, t_dm1s)
                
        return (e_cell, e_tot, RDM1)  
        
#########################################        
##########     CASPDFT solver    ##########
#########################################  

    import numpy as np
    from scipy import linalg
    from mrh.my_dmet import localintegrals
    import os, time
    import sys, copy
    from pyscf import gto, scf, ao2mo, mcscf, fci, lib, dft
    import time
    from pyscf import dft, ao2mo, fci, mcscf
    from pyscf.lib import logger, temporary_env
    from pyscf.mcscf import mc_ao2mo
    from pyscf.mcscf.addons import StateAverageMCSCFSolver
    from pyscf.mcscf.addons import StateAverageFCISolver
    from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
    from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
    from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
    from mrh.my_pyscf.mcpdft.otfnal import transfnal
    from mrh.my_pyscf.mcpdft.mcpdft import energy_tot 
    from mrh.my_pyscf import mcpdft
    from mrh.my_pyscf.mcpdft import _dms 
    
    
    def CASPDFT(self, solver='FCI', state_specific_=None, state_average_=None, state_average_mix_=None, nevpt2_roots=None, cell=None, mc_dup=None, nevpt2_nroots=10, nevpt2_spin=0.0, kmf=None, w90=None, emb_orbs=None, ao2eo=None, mask4Gamma=None, OEH_type=None, emb_core_orbs = None, core_orbs=None):
        '''
        CASSCF with FCI or DMRG solver:
            - Ground state
            - State-specfic
            - State-average
        state_specific_ is used to pass the state_id to the solver
        state_average_ is used to pass the weights to the solver
        '''      
        # scell, self.phase = self.get_phase(self.cell, self.kpts, self.kmesh)
        self.kmf=kmf
        self._is_ROHF = self.kmf._is_ROHF
        self.local = localbasis.Local(cell, self.kmf, w90, self._is_ROHF, self.xc_omega)
        self.emb_orbs = emb_orbs
        self.core_orbs = core_orbs
        self.emb_core_orbs = emb_core_orbs
        self.ao2eo = ao2eo
        self.mc_dup=mc_dup
        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy     
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb) #just a diagonal matrix with the correct dimensions
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb) #not same in two different calculations; which basis is this in?
        self.mf.scf(self.DMguess)      
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        DM1loc = self.mf.make_rdm1()
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if self.cas is None:
            cas_nelec = self.Nel
            cas_norb = self.Norb
        else:
            cas_nelec = self.cas[0]
            cas_norb = self.cas[1]

        # Updating mc object from the new mf, mol objects    
        if state_specific_ is not None and state_average_ is None:
            state_id = state_specific_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_specific_(state_id)
        elif state_specific_ is None and state_average_ is not None and state_average_mix_ is None:
            weights = state_average_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_average_(weights)
        elif state_average_mix_ is not None:
            solver1, solver2, weights = state_average_mix_
            mcscf.state_average_mix_(self.mc, [solver1, solver2], weights)
        else:
            state_id = 0
            self.nroots = 1
            self.mc.fcisolver.nroots = self.nroots 
        
        self.mc.mol = self.mol  
        self.mc._scf = self.mf 
        self.mc.ncas = cas_norb
        nelecb = (cas_nelec - self.mol.spin)//2
        neleca = cas_nelec - nelecb
        self.mc.nelecas = (neleca, nelecb)
        ncorelec = self.mol.nelectron - (self.mc.nelecas[0] + self.mc.nelecas[1])        
        assert(ncorelec % 2 == 0)
        self.mc.ncore = ncorelec // 2   
        self.mc.mo_coeff = self.mf.mo_coeff
        self.mc.mo_energy = self.mf.mo_energy
        
        # Define FCI solver
        if solver == 'CheMPS2':      
            self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
        elif solver == 'FCI' and self.e_shift is not None and state_average_mix_ is None:         
            target_SS = 0.5*self.twoS*(0.5*self.twoS + 1)
            self.mc.fix_spin_(shift=self.e_shift, ss=target_SS)                  
            
        if self.mo is not None: 
            mo = self.mo
        elif self.molist is not None: 
            if self.chkfile is not None:
                mo = lib.chkfile.load(self.chkfile, 'mcscf/mo_coeff')
            else:
                mo = self.mc.mo_coeff
            mo = mcscf.sort_mo(self.mc, mo, self.molist, base=0)
        else: 
            mo = self.mc.mo_coeff
        e_tot, e_cas, fcivec = self.mc.kernel(mo)[:3]
        if state_specific_ is None and state_average_ is not None: 
            e_tot = np.asarray(self.mc.e_states)  
        if not self.mc.converged: print('           WARNING: The solver is not converged')
        if self.mc.converged: print('          The solver has converged')
        # Save mo for the next iterations
        self.mo_nat = self.mc.mo_coeff           
        self.mo = self.mc.mo_coeff 
        # Compute energy and RDM1      
        if self.nroots == 1 or state_specific_ is not None:
            civec = fcivec
            self.SS, spin_multiplicity = mcscf.spin_square(self.mc)           
            ###### Get RDM1 + RDM2 #####
            core_norb = self.mc.ncore              
            core_MO = self.mc.mo_coeff[:,:core_norb]
            active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
            casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(civec, cas_norb, self.mc.nelecas) #in CAS(MO) space  
            rdm1spin_sep_a, rdm1spin_sep_b = self.mc.fcisolver.make_rdm1s(fcivec, cas_norb, self.mc.nelecas) #get spin-separated RDMs
            rdm1spin_sep_ = [rdm1spin_sep_a,rdm1spin_sep_b]
            casdm1sa_mo, casdm1sb_mo = rdm1spin_sep_a, rdm1spin_sep_b 
            # Transform the casdm1_mo to local basis
            casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
            casdm1sa = lib.einsum('ap,pq->aq', active_MO, casdm1sa_mo)
            casdm1sb = lib.einsum('ap,pq->aq', active_MO, casdm1sb_mo)
            casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
            casdm1sa = lib.einsum('bq,aq->ab', active_MO, casdm1sa)
            casdm1sb = lib.einsum('bq,aq->ab', active_MO, casdm1sb)
            # print("coredm1",coredm1)
            coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
            #Transform all the density matrices to the ao basis so that you can construct the density and on-top densities for mc-pdft
            ao_basis_casdm1sa = lib.einsum('im,mn,jn->ij', self.ao2eo[0].real, casdm1sa, self.ao2eo[0].real.conj())
            ao_basis_casdm1sb = lib.einsum('im,mn,jn->ij', self.ao2eo[0].real, casdm1sb, self.ao2eo[0].real.conj())
            ao_basis_casdm1s = np.asarray([ao_basis_casdm1sa,ao_basis_casdm1sb])
            RDM1 = coredm1 + casdm1  
            RDM1Sa = coredm1/2 + casdm1sa
            RDM1Sb = coredm1/2 + casdm1sb              
            RDM1s = [RDM1Sa,RDM1Sb]
            nelectrons = np.trace(RDM1[:,:])
            casdm1s = [casdm1sa_mo,casdm1sb_mo]
            ao_basis_RDM1 = lib.einsum('im,mn,jn->ij', self.ao2eo[0].real, RDM1, self.ao2eo[0].conj().real)
            ao_basis_RDM1sa = lib.einsum('Rim,mn,jn->Rij', self.ao2eo.real, RDM1Sa, self.ao2eo[0].conj().real)
            ao_basis_RDM1sb = lib.einsum('Rim,mn,jn->Rij', self.ao2eo.real, RDM1Sb, self.ao2eo[0].conj().real)
            ao_basis_RDM1s = [ao_basis_RDM1sa,ao_basis_RDM1sb]
            #create an aobasis_coreDM1
            #Activate the lower block only when you need casdm2s
            """
            # casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
            # casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
            # casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
            # casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
        
            # coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            # coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
            # coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

            # effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            # effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
            # effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)            
                        
            # RDM2 = coredm2 + casdm2 + effdm2   
            
            # ao_basis_RDM2 = lib.einsum('ap,pqrs->aqrs', self.ao2eo[0].real, RDM2)
            # ao_basis_RDM2 = lib.einsum('bq,aqrs->abrs', self.ao2eo[0].real, ao_basis_RDM2)
            # ao_basis_RDM2 = lib.einsum('cr,abrs->abcs', self.ao2eo[0].real, ao_basis_RDM2)
            # ao_basis_RDM2 = lib.einsum('ds,abcs->abcd', self.ao2eo[0].real, ao_basis_RDM2)
            
            
            # print("RDM2.shape",RDM2.shape)
            # print("ao_basis_RDM2.shape",ao_basis_RDM2.shape)           
            # ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       # + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       # + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       # + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       # + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            """
            ImpurityEnergy = 0.0
            e_cell = self.kmf_ecore + ImpurityEnergy  
            print('       State %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (state_id, e_tot, e_cell, self.SS))           
            self.mask4Gamma = mask4Gamma
            self.OEH_type = OEH_type
            self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(0., self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=None)
            emb_core_1RDM_for_mcpdft = self.local.make_emb_space_RDM(self.loc_1RDM_R0, self.emb_orbs, self.core_orbs, self.emb_core_orbs)
            emb_core_1RDM_for_mcpdft1sa=emb_core_1RDM_for_mcpdft/2
            emb_core_1RDM_for_mcpdft1sb=emb_core_1RDM_for_mcpdft/2
            emb_core_1RDM_for_mcpdft1sa[:self.Norb,:self.Norb] = RDM1Sa
            emb_core_1RDM_for_mcpdft1sb[:self.Norb,:self.Norb] = RDM1Sb
            # emb_core_1RDM_for_mcpdft[:self.Norb,:self.Norb] = RDM1 #In case one wants to implement non-spin separated
            self.loc_1RDM_R0_modified = self.loc_1RDM_R0
            ao_basis_emb_core_1RDM_for_mcpdft = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft)
            ao_basis_emb_core_1RDM_for_mcpdft_1sa = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft1sa)
            ao_basis_emb_core_1RDM_for_mcpdft_1sb = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft1sb)
            self.loc_1RDM_R0_modified[0][:self.Norb,:self.Norb] = RDM1
            self.loc_1RDM_R0_modified_ao_basis = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft])
            self.loc_1RDM_R0_modified_ao_basis_1sa = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft_1sa])
            self.loc_1RDM_R0_modified_ao_basis_1sb = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft_1sb])
            self.loc_1RDM_R0_modified_ao_basis_1s = ([self.loc_1RDM_R0_modified_ao_basis_1sa,self.loc_1RDM_R0_modified_ao_basis_1sb])
            e_pdft = get_dmet_pdft (self.mc, RDM1, 'tPBE', casdm1s, casdm1_mo, casdm2_mo, rdm1spin_sep_a, rdm1spin_sep_b, cell, mc_dup=mo, ao_basis_RDM1s=self.loc_1RDM_R0_modified_ao_basis_1s, ao2eo=self.ao2eo, kmf=self.kmf)           
            print("The MC-PDFT energy is ", e_pdft)
        elif state_average_ is not None: #still under development
            nroots = self.mc.fcisolver.nroots = self.nroots
            civec = fcivec
            tot_SS = 0 
            RDM1 = []  
            e_cell = []  
            rdm1spin_sep_a, rdm1spin_sep_b = self.mc.fcisolver.states_make_rdm1s(fcivec, cas_norb, self.mc.nelecas)
            rdm1spin_sep_ = [rdm1spin_sep_a[0],rdm1spin_sep_b[0]]
            rdm1s, rdm2s = self.mc.fcisolver.states_make_rdm12(fcivec, cas_norb, self.mc.nelecas)   #This is rdm1 and rdm2, not 1s and 2s   
            # dm1s = np.asarray ( self.mc.make_rdm1s() )
            SSs, spin_multiplicities = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas) 
            # print("Am I here? -1167")
            casdm1s_append = []
            for i in range(len(weights)):
                weights = state_average_
                if not 'FakeCISolver' in str(self.mc.fcisolver):
                    self.mc = self.mc.state_average_(weights)
                
                SS, spin_multiplicity = SSs[i], spin_multiplicities[i]
                
                ###### Get RDM1 + RDM2 #####
                core_norb = self.mc.ncore    
                core_MO = self.mc.mo_coeff[:,:core_norb]
                active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
                casdm1_mo, casdm2_mo = rdm1s[i], rdm2s[i] 
                casdm1sa_mo, casdm1sb_mo = rdm1spin_sep_a[i], rdm1spin_sep_b[i] 
                casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
                casdm1sa = lib.einsum('ap,pq->aq', active_MO, casdm1sa_mo)
                casdm1sb = lib.einsum('ap,pq->aq', active_MO, casdm1sb_mo)
                casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
                casdm1sa = lib.einsum('bq,aq->ab', active_MO, casdm1sa)
                casdm1sb = lib.einsum('bq,aq->ab', active_MO, casdm1sb)
                coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
                rdm1 = coredm1 + casdm1     
                rdm1Sa = coredm1/2 + casdm1sa
                rdm1Sb = coredm1/2 + casdm1sb              
                rdm1S = [rdm1Sa,rdm1Sb]
                ao_basis_rdm1sa = lib.einsum('Rim,mn,jn->Rij', self.ao2eo.real, rdm1Sa, self.ao2eo[0].conj().real)
                ao_basis_rdm1sb = lib.einsum('Rim,mn,jn->Rij', self.ao2eo.real, rdm1Sb, self.ao2eo[0].conj().real)
                ao_basis_rdm1s = [ao_basis_rdm1sa,ao_basis_rdm1sb]
                self.mask4Gamma = mask4Gamma
                self.OEH_type = OEH_type
                self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(0., self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=None)
                emb_core_1RDM_for_mcpdft = self.local.make_emb_space_RDM(self.loc_1RDM_R0, self.emb_orbs, self.core_orbs, self.emb_core_orbs)
                emb_core_1RDM_for_mcpdft1sa=emb_core_1RDM_for_mcpdft/2
                emb_core_1RDM_for_mcpdft1sb=emb_core_1RDM_for_mcpdft/2
                emb_core_1RDM_for_mcpdft1sa[:self.Norb,:self.Norb] = rdm1Sa
                emb_core_1RDM_for_mcpdft1sb[:self.Norb,:self.Norb] = rdm1Sb
                # emb_core_1RDM_for_mcpdft[:self.Norb,:self.Norb] = RDM1 #In case one wants to implement non-spin separated
                self.loc_1RDM_R0_modified = self.loc_1RDM_R0
                ao_basis_emb_core_1RDM_for_mcpdft = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft)
                ao_basis_emb_core_1RDM_for_mcpdft_1sa = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft1sa)
                ao_basis_emb_core_1RDM_for_mcpdft_1sb = self.local.loc_kpts_to_emb_trial_2(self.loc_1RDM_R0_modified, self.emb_orbs, self.core_orbs, self.emb_core_orbs, emb_core_1RDM_for_mcpdft1sb)
                self.loc_1RDM_R0_modified[0][:self.Norb,:self.Norb] = rdm1
                self.loc_1RDM_R0_modified_ao_basis = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft])
                self.loc_1RDM_R0_modified_ao_basis_1sa = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft_1sa])
                self.loc_1RDM_R0_modified_ao_basis_1sb = np.asarray([ao_basis_emb_core_1RDM_for_mcpdft_1sb])
                self.loc_1RDM_R0_modified_ao_basis_1s = ([self.loc_1RDM_R0_modified_ao_basis_1sa,self.loc_1RDM_R0_modified_ao_basis_1sb]) 
                
                # Transform the casdm2_mo to local basis
                
                if True: 
                    # this is used to get around with the huge memory to get the IMP nergy which is not necessary for the Gamma-point embedding  
                    # TODO: generalize it 
                    Imp_Energy_state = 0.0
                else:
                    casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
                    casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
                    casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
                    casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
                
                    coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb], dtype=np.float64)  # TODO: this is impractical for the big embedding space. Lots of memory
                    coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
                    coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

                    effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb], dtype=np.float64)   #
                    effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
                    effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                                
                    rdm2 = coredm2 + casdm2 + effdm2         
                    
                    Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                                  + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])

                # print(" self.kmf_ecore",self.kmf_ecore)    
                Imp_e = self.kmf_ecore + Imp_Energy_state  
                # print(" Imp_e",Imp_e) 
                # print(" casdm1sa",casdm1sa)
                
                # print(" Imp_Energy_state",Imp_Energy_state)
                # get_dmet_sapdft (self.mc, 40.0, 'tPBE', casdm1_mo, casdm2_mo, cell, mc_dup=self.mc_dup)             
                if state_average_ is not None:
                    print('       State %d (%5.3f): E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, weights[i], e_tot[i], Imp_e, SS))  
                    # get_dmet_sapdft (self.mc, 40.0, 'tPBE', casdm1_mo, casdm2_mo, cell, mc_dup=self.mc_dup)
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)
                # epdft = get_dmet_sapdft (self.mc, 40.0, 'tPBE', casdm1_mo, casdm2_mo, cell, mc_dup=self.mc_dup,weights=weights)
                casdm1s = [casdm1sa_mo,casdm1sb_mo]
                rdm1spin_sep = [rdm1spin_sep_a[i],rdm1spin_sep_b[i]]
                epdft = get_dmet_sapdft(self.mc, rdm1, casdm1s, casdm1_mo, casdm2_mo, 'tPBE', rdm1spin_sep_a[i], rdm1spin_sep_b[i], cell=cell, root = i, mc_dup=self.mc_dup, weights=weights, ao_basis_rdm1s=self.loc_1RDM_R0_modified_ao_basis_1s, ao2eo=self.ao2eo, kmf=self.kmf)
                # epdft = get_dmet_sapdft(self.mc, rdm1, 'tPBE', rdm1s[i], rdm2s[i], casdm1_mo, casdm2_mo, rdm1spin_sep_a[i], rdm1spin_sep_b[i], cell, root = i,mc_dup=self.mc_dup, weights=weights)
                # e_pdft_core = epdft 
                print("The MC-PDFT energy of state",i, "is ", epdft)
            RDM1 = lib.einsum('i,ijk->jk',state_average_, RDM1) 
            e_cell = lib.einsum('i,i->',state_average_, e_cell) 
            self.mask4Gamma = mask4Gamma
            self.OEH_type = OEH_type
            self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(0., self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=None)
        if nevpt2_roots is not None:       
            self.mf.spin = nevpt2_spin
            nelecb = (cas_nelec - self.mf.spin)//2
            neleca = cas_nelec - nelecb
            nelecas = (neleca, nelecb)
            mc_CASCI = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
            mc_CASCI.fcisolver.nroots = nevpt2_nroots
            fcivec = mc_CASCI.kernel(self.mc.mo_coeff)[2]
            ground_state = fcivec[0]
            
            # Run NEVPT2
            e_casci_nevpt = []
            t_dm1s = []
            from pyscf.fci import cistring
            print("=====================================")
            if len(nevpt2_roots) > len(fcivec): nevpt2_roots = np.arange(len(fcivec))
            for root in nevpt2_roots:
                ci = fcivec[root]
                SS = mc_CASCI.fcisolver.spin_square(ci, cas_norb, nelecas)[0]
                e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
                if not isinstance(mc_CASCI.e_tot, np.ndarray):
                    e_CASCI = mc_CASCI.e_tot
                    e_nevpt = e_CASCI + e_corr
                else:
                    e_CASCI = mc_CASCI.e_tot[root]
                    e_nevpt = e_CASCI + e_corr
                e_casci_nevpt.append([SS, e_CASCI, e_nevpt])
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
                rdm1 = mc_CASCI.fcisolver.make_rdm12(ci, cas_norb, nelecas)[0]
                e, v = np.linalg.eig(rdm1)
                # Find the two SDs with most contribution 
                strsa = np.asarray(cistring.make_strings(range(cas_norb), neleca))
                strsb = np.asarray(cistring.make_strings(range(cas_norb), nelecb))    
                na = len(strsa)
                nb = len(strsb)
                
                idx_1st_max = abs(ci).argmax()
                c1 = ci.flatten()[idx_1st_max]
                stra_1st = strsa[idx_1st_max // nb]
                strb_1st = strsb[idx_1st_max % nb ]
                
                abs_fcivec = abs(ci).flatten()
                abs_fcivec[idx_1st_max] = 0.0
                idx_2nd_max = abs_fcivec.argmax()
                c2 = ci.flatten()[idx_2nd_max]
                stra_2nd = strsa[idx_2nd_max // nb]
                strb_2nd = strsb[idx_2nd_max % nb ]
                
                abs_fcivec[idx_2nd_max] = 0.0
                idx_3rd_max = abs_fcivec.argmax()
                c3 = ci.flatten()[idx_3rd_max]
                stra_3rd = strsa[idx_3rd_max // nb]
                strb_3rd = strsb[idx_3rd_max % nb ]

                abs_fcivec[idx_3rd_max] = 0.0
                idx_4th_max = abs_fcivec.argmax()
                c4 = ci.flatten()[idx_4th_max]
                stra_4th = strsa[idx_4th_max // nb]
                strb_4th = strsb[idx_4th_max % nb ]
                
                print("== State {0:d}: {1:2.4f}|{2:s},{3:s}> + {4:2.4f}|{5:s},{6:s}> + {7:2.4f}|{8:s},{9:s}> + {10:2.4f}|{11:s},{12:s}>".format(root, c1, bin(stra_1st)[2:], bin(strb_1st)[2:], c2, bin(stra_2nd)[2:], bin(strb_2nd)[2:], c3, bin(stra_3rd)[2:], bin(strb_3rd)[2:], c4, bin(stra_4th)[2:], bin(strb_4th)[2:]))
                print("   Occupancy:", e)
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
                
                ''' Calculate Transform density matrix '''
                t_dm1 = mc_CASCI.fcisolver.trans_rdm1(ground_state, ci, mc_CASCI.ncas, mc_CASCI.nelecas)
                # transform density matrix to EO representation
                orbcas = mc_CASCI.mo_coeff[:,mc_CASCI.ncore:mc_CASCI.ncore+mc_CASCI.ncas]
                t_dm1_emb = orbcas @ t_dm1 @ orbcas.T
                t_dm1s.append(t_dm1_emb)
                
            print("=====================================") 
                
            #Pack E_CASSCF and E_NEVPT2 into a tuple of e_tot
            e_casci_nevpt = np.asarray(e_casci_nevpt)
            e_tot = (e_tot, e_casci_nevpt, t_dm1s)
 
        return (e_cell, e_tot, RDM1)  
        
        
                    
#########################################        
##########     MC-PDFT solver    ##########
#########################################  
import numpy as np
import time
import os
from scipy import linalg
from pyscf.fci import cistring
from pyscf.dft import gen_grid
from pyscf.mcscf import mc_ao2mo, mc1step
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, StateAverageMixFCISolver
from mrh.my_pyscf.mcpdft import pdft_veff, ci_scf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
import math
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
from mrh.my_dmet import localintegrals
import os, time
import sys, copy
from pyscf import gto, scf, ao2mo, mcscf, fci, lib, dft
from pyscf import gto, dft, ao2mo, fci, mcscf, lib, __config__
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft import _dms 


def get_dmet_pdft (dmetmc, rdm, my_ot, casdm1s, casdm1, casdm2,  rdm1spin_sep_a, rdm1spin_sep_b, cell=None,  mc_dup=None, ao_basis_RDM1s=None, ao2eo=None, kmf=None, my_grid=6):

    print ( 'you are doing a calculation using', my_ot)
    from pyscf.pbc import gto, scf, cc, df, mp
    from pyscf import gto, scf, cc, df, mp
    ks = dft.RKS(cell).density_fit()
    if my_ot[:1].upper () == 'T':
        ks.xc = my_ot[1:]
        otfnal = transfnal (ks)
    elif my_ot[:2].upper () == 'FT':
        ks.xc = my_ot[2:]
        otfnal = ftransfnal (ks)
    grids = dft.gen_grid.Grids(cell) 
    grids.level = my_grid
    otfnal.grids = grids
    otfnal.verbose = 3
    e_tot, E_ot = MCPDFT (dmetmc, rdm, otfnal, casdm1s, casdm1, casdm2, rdm1spin_sep_a, rdm1spin_sep_b, cell=cell, mc_dup=mc_dup, ao_basis_RDM1s=ao_basis_RDM1s, ao2eo=ao2eo, kmf=kmf)
    print ('Final on-top energy is', E_ot)
    print ('Final DMET-PDFT energy is', e_tot)
    return e_tot, E_ot
    
from mrh.my_pyscf.mcpdft.mcpdft import energy_mcwfn   
def MCPDFT (mc, rdm, ot, casdm1s, casdm1, casdm2, rdm1spin_sep_a, rdm1spin_sep_b, root=-1, cell=None, mc_dup=None, ao_basis_RDM1s=None, ao2eo=None, kmf=None):
    ''' Calculate MC-PDFT total energy
        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-PDFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py
        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-PDFT results (i.e., using state-averaged density matrices)
        Returns:
            Total MC-PDFT energy including nuclear repulsion energy.
    '''
    # e_mcwfn_MCPDFT = energy_mcwfn(mc, ot=ot, casdm1s=casdm1s, casdm2=casdm2)
    t0 = (time.clock (), time.time ())
    dm1s = np.asarray ( mc.make_rdm1s() )
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    Vnn = cell.energy_nuc()
    hyb_x, hyb_c = hyb
    """
    h = mc._scf.get_hcore ()
    dm1 = dm1s[0] + dm1s[1]
    ao_basis_RDM1 = ao_basis_RDM1s[0] + ao_basis_RDM1s[1]
    if ot.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        vj, vk = mc._scf.get_jk (dm=rdm)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j (dm=rdm)
    Te_Vne = np.tensordot (h, rdm)
    E_j = np.tensordot (vj, rdm) / 2
    """   
    h = kmf.get_hcore ()
    dm1 = dm1s[0] + dm1s[1]
    ao_basis_RDM1 = ao_basis_RDM1s[0] + ao_basis_RDM1s[1]
    if ot.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        vj, vk = kmf.get_jk (dm_kpts=np.asarray(ao_basis_RDM1s[0]), hermi=1)
        vj = vj[0] + vj[1]
    else:
        vj = kmf.get_j (dm_kpts=np.asarray(ao_basis_RDM1[0]), hermi=1)
    Te_Vne = np.tensordot (h, np.asarray(ao_basis_RDM1[0]))
    E_j = np.tensordot (vj, np.asarray(ao_basis_RDM1[0])) / 2  
    if ot.verbose >= logger.DEBUG  or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        E_x = -(np.tensordot (vk[0], dm1s[0]) + np.tensordot (vk[1], dm1s[1])) / 2
    else:
        E_x = 0
    logger.debug (ot, 'CAS energy decomposition:')
    logger.debug (ot, 'Vnn = %s', Vnn)
    logger.debug (ot, 'Te + Vne = %s', Te_Vne)
    logger.debug (ot, 'E_j = %s', E_j)
    logger.debug (ot, 'E_x = %s', E_x)
    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 > 1e-10:
        logger.debug (ot, 'Adding %s * %s CAS exchange to E_ot', hyb, E_x)
    t0 = logger.timer (ot, 'Vnn, Te, Vne, E_j, E_x', *t0)   
    E_ot = get_E_ot (mc, rdm, ot, casdm1s, casdm1, casdm2, rdm1spin_sep_a, rdm1spin_sep_b, mc_dup=mc_dup, ao_basis_RDM1s=ao_basis_RDM1s, ao2eo=ao2eo)
    t0 = logger.timer (ot, 'E_ot', *t0)
    E_c = 0
    e_tot = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) + E_ot 
    print ("This is the breakdown")
    print ("Vnn",Vnn)
    print ("Te_Vne",Te_Vne)
    print ("E_j",E_j)
    print ("E_x",E_x)
    print ("E_ot",E_ot)
    print ("e_tot",e_tot)    
    logger.info (ot, 'MC-PDFT E = %s, Eot(%s) = %s', e_tot, ot.otxc, E_ot)
    return e_tot, E_ot

def get_E_ot (mc, rdm, ot, casdm1s, casdm1, casdm2, rdm1spin_sep_a, rdm1spin_sep_b, max_memory=2000, hermi=1, root=-1, ci=None, mo_coeff=None, mc_dup=None, ao_basis_RDM1s=None, ao2eo=None):
    print("Starting on-top-energy calculation")
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal 
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    ao2eo_inv = np.linalg.pinv(ao2eo[0])
    mo_cas_ao2eo = lib.einsum('ap,pq->aq',  ao2eo[0].real, mo_cas )
    moH_cas = mo_cas.conj ().T
    mo_core = mo_coeff[:,:ncore]
    moH_core = mo_core.conj ().T
    # print("mo_cas",mo_cas)
    # print("ao2eo_inv",ao2eo_inv.real)
    # print("lib.einsum('ap,pq->aq',  ao2eo[0].real, mo_cas )",lib.einsum('ap,pq->aq',  ao2eo_inv.real, mo_cas ))
    # print("mo_cas_ao2eo",mo_cas_ao2eo)
    # print("ao2eo[0]",ao2eo[0].real)
    t0 = (time.clock (), time.time ())
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    oneCDMs = np.asarray(ao_basis_RDM1s)
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao=ao2eo.shape[1]
    E_ot = 0.0
    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        cascm2 = _dms.dm2_cumulant (casdm2, casdm1s) #Check the exact implementations here       
        Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas_ao2eo, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top exchange-correlation energy calculation', *t0)
    return E_ot
    
    
    
    
def get_energy_decomposition_mcpdft (mc, ot, mo_coeff=None, ci=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear potential, core, Coulomb, exchange,
    and correlation terms. The exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already hybrids. Most return arguments
    are lists if mc is a state average instance. '''
    e_tot, e_ot, e_mcscf, e_cas, ci, mo_coeff = mc.kernel (mo=mo_coeff, ci=ci)[:6]
    if isinstance (mc, StateAverageMCSCFSolver):
        e_tot = mc.e_states
    e_nuc = mc._scf.energy_nuc ()
    h = mc.get_hcore ()
    xfnal, cfnal = ot.split_x_c ()
    if isinstance (mc, StateAverageMCSCFSolver):
        e_core = []
        e_coul = []
        e_otx = []
        e_otc = []
        e_wfnxc = []
        nelec_root = [mc.nelecas,]* len (e_mcscf)
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            nelec_root = []
            for s in mc.fcisolver.fcisolvers:
                nelec_root.extend ([mc.fcisolver._get_nelec (s, mc.nelecas),]*s.nroots)
        for ci_i, ei_mcscf, nelec in zip (ci, e_mcscf, nelec_root):
            row = _get_e_decomp (mc, ot, mo_coeff, ci_i, ei_mcscf, e_nuc, h, xfnal, cfnal, nelec)
            e_core.append  (row[0])
            e_coul.append  (row[1])
            e_otx.append   (row[2])
            e_otc.append   (row[3])
            e_wfnxc.append (row[4])
    else:
        e_core, e_coul, e_otx, e_otc, e_wfnxc = _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, e_nuc, h, xfnal,
            cfnal, mc.nelecas)
    print ("e_nuc",e_nuc)
    print ("e_core",e_core)
    print ("e_coul",e_coul)
    print ("e_otx",e_otx)
    print ("e_otc",e_otc)
    print ("e_wfnxc",e_wfnxc)
    
    return e_nuc, e_core, e_coul, e_otx, e_otc, e_wfnxc
    
    
    
from mrh.my_pyscf.mcpdft.mcpdft import energy_mcwfn    
def get_dmet_sapdft (dmetmc, rdm, casdm1s, casdm1, casdm2, my_ot, rdm1spin_sep_a, rdm1spin_sep_b, cell=None, mc_dup=None, my_grid=6, weights=None, root = 0, cipass=None, dmet_pdft_roots=10, dmet_pdft_spin=0.0, ao_basis_rdm1s=None, ao2eo=None, kmf=None):
    
    print ( 'you are doing a calculation using', my_ot)
    from pyscf.pbc import gto, scf, cc, df, mp
    from pyscf import gto, scf, cc, df, mp
    ks = dft.RKS(cell).density_fit()
    if my_ot[:1].upper () == 'T':
        ks.xc = my_ot[1:]
        otfnal = transfnal (ks)
    elif my_ot[:2].upper () == 'FT':
        ks.xc = my_ot[2:]
        otfnal = ftransfnal (ks)
    grids = dft.gen_grid.Grids(cell) 
    grids.level = my_grid
    otfnal.grids = grids
    otfnal.verbose = 3
    e_tot, E_ot = SAMCPDFT (dmetmc, rdm, casdm1s, casdm1, casdm2, otfnal, rdm1spin_sep_a, rdm1spin_sep_b, cell=cell, root=root, mc_dup=mc_dup, weights=weights, dmet_pdft_spin=dmet_pdft_spin, ao_basis_rdm1s=ao_basis_rdm1s, ao2eo=ao2eo, kmf=kmf)
    print ('Final on-top energy is', E_ot)
    print ('Final DMET-PDFT energy is', e_tot)
    return e_tot, E_ot

        
    
def SAMCPDFT (mc, rdm, casdm1s, casdm1, casdm2, ot, rdm1spin_sep_a, rdm1spin_sep_b, root=0, cell=None, mc_dup=None, weights=None, dmet_pdft_spin=0.0, cipass=None, ao_basis_rdm1s=None, ao2eo=None, kmf=None):
    ''' Calculate MC-PDFT total energy
        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-PDFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py
        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-PDFT results (i.e., using state-averaged density matrices)
        Returns:
            Total MC-PDFT energy including nuclear repulsion energy.
    '''
    # Vnn = cell.energy_nuc()
    # e_mcwfn_MCPDFT = energy_mcwfn(mc, ot=ot, casdm1s=casdm1s, casdm2=casdm2, state=root)
    t0 = (time.clock (), time.time ())    
    verbose = mc.verbose
    log = logger.new_logger (mc, verbose=verbose)
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    Vnn = cell.energy_nuc()
    hyb_x, hyb_c = hyb
    """
    h = mc._scf.get_hcore ()
    ao_basis_rdm1 = ao_basis_rdm1s[0] + ao_basis_rdm1s[1]
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        vj, vk = mc._scf.get_jk (dm=rdm)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j (dm=rdm)
    Te_Vne = np.tensordot (h, rdm)
    E_j = np.tensordot (vj, rdm) / 2
    """
    ao_basis_RDM1s = ao_basis_rdm1s   
    h = kmf.get_hcore ()
    ao_basis_RDM1 = ao_basis_RDM1s[0] + ao_basis_RDM1s[1]
    if ot.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        vj, vk = kmf.get_jk (dm_kpts=np.asarray(ao_basis_RDM1[0]))
        vj = vj[0] + vj[1]
    else:
        vj = kmf.get_j (dm_kpts=np.asarray(ao_basis_RDM1[0]))
    Te_Vne = np.tensordot (h, np.asarray(ao_basis_RDM1[0]))
    E_j = np.tensordot (vj, np.asarray(ao_basis_RDM1[0])) / 2  
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    if log.verbose >= logger.DEBUG  or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 :
        E_x = -(np.tensordot (vk[0], rdm1spin_sep_a[0]) + np.tensordot (vk[1], rdm1spin_sep_b[1])) / 2
    else:
        E_x = 0
    logger.debug (ot, 'CAS energy decomposition:')
    logger.debug (ot, 'Vnn = %s', Vnn)
    logger.debug (ot, 'Te + Vne = %s', Te_Vne)
    logger.debug (ot, 'E_j = %s', E_j)
    logger.debug (ot, 'E_x = %s', E_x)
    E_c = 0
    if log.verbose >= logger.DEBUG or abs (hyb_c) > 1e-10:
        # g_pqrs * l_pqrs / 2
        #if log.verbose >= logger.DEBUG:
        aeri = ao2mo.restore (1, mc.get_h2eff (mo_coeff), mc.ncas)
        E_c = np.tensordot (aeri, cascm2, axes=4) / 2
        log.info ('E_c = %s', E_c)   
    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 > 1e-10:
        logger.debug (ot, 'Adding %s * %s CAS exchange to E_ot', hyb, E_x)
    t0 = logger.timer (ot, 'Vnn, Te, Vne, E_j, E_x', *t0)
    E_ot = sa_get_E_ot (mc, rdm, casdm1s, casdm1, casdm2, ot, rdm1spin_sep_a, rdm1spin_sep_b, mc_dup=mc_dup, root=root, weights=weights, dmet_pdft_spin=dmet_pdft_spin, ao_basis_rdm1s=ao_basis_rdm1s, ao2eo=ao2eo)
    t0 = logger.timer (ot, 'E_ot', *t0)
    e_tot = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) + E_ot 
    print ("This is the breakdown")
    print ("Vnn",Vnn)
    print ("Te_Vne",Te_Vne)
    print ("E_j",E_j)
    print ("E_x",E_x)
    print ("E_c",E_c)
    print ("E_ot",E_ot)
    print ("e_tot",e_tot)    
    logger.info (ot, 'MC-PDFT E = %s, Eot(%s) = %s', e_tot, ot.otxc, E_ot)
    return e_tot, E_ot

def sa_get_E_ot (mc, rdm, casdm1s, casdm1, casdm2, ot, rdm1spin_sep_a, rdm1spin_sep_b, max_memory=2000, hermi=1, root=0, ci=None, mo_coeff=None, mc_dup=None, weights=None, dmet_pdft_spin=0.0, ao_basis_rdm1s=None, ao2eo=None): 
    print("Starting on-top-energy calculation")
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff #2 change
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    mo_cas_ao2eo = lib.einsum('ap,pq->aq', ao2eo[0].real, mo_cas)
    mo_cas2 = mc.mo_coeff[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T
    mo_core = mo_coeff[:,:ncore]
    moH_core = mo_core.conj ().T
    t0 = (time.clock (), time.time ())
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    hyb_x, hyb_c = hyb
    t0 = (time.clock (), time.time ())
    oneCDMs = np.asarray(ao_basis_rdm1s)
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2eo.shape[1]
    E_ot = 0.0
    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        cascm2 = _dms.dm2_cumulant (casdm2, casdm1s) #Check the exact implementations here       
        Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas_ao2eo, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top exchange-correlation energy calculation', *t0)
    return E_ot
    
    
    
    
