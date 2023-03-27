'''
Created on Feb 8, 2019

@author: albertgo
'''

import numpy as np
import glob
import logging
import sys
import torch
import re
from typing import List, Sequence, Any, Dict

from cdd_chem.util import io
from t_opt.atom_info import NumToName
from t_opt.coordinates_batch import SameSizeCoordsBatch
from t_opt import atom_info
from t_opt.abstract_NNP_computer import AbstractNNPComputer,\
    EnergyAndGradHelperHarmConstraint, EnergyAndGradHelperFixedAtoms,\
    AbstractEnergyAndGradHelper, EnergyAndGradHelperInterface
from t_opt.unit_conversion import Units, KCAL_MOL_to_AU, bohr2ang

log = logging.getLogger(__name__)


class ANIComputer(AbstractNNPComputer):
    '''
    Energy and Force computer using neurochem c++ apI

    ALL MOLECULES MUST HAVE the SAME SEQUENCE OF ATOMS TYPE == BE ISOMERS with same atom ordering
    '''


    def __init__(self, net_dir:str, outputGrads:bool, compute_stdev:bool, gpuid:Sequence[int] = [0],
                 sinet:bool = False, energyOutUnits=Units.KCAL):
        # pylint: disable=R0913, R0914, W0102
        '''
        Constructor
        '''
        self.device = torch.device("cuda")
        log.info(f"device={self.device} cuda.isavaialble={torch.cuda.is_available()}") # pylint: disable=W1203

        try:
            import pyNeuroChem as neuro # pylint: disable=C0415
        except ImportError as e:
            log.warning("NeuroChem shared library not found. Make sure your LD_LIBRARY_PATH is set correctly")
            raise e


        paramFile = glob.glob1(net_dir,"*.params")
        if len(paramFile) != 1:
            # pylint: disable=W1203
            log.critical(f"paramFile does not exist or is not uique: {paramFile}")
            sys.exit(1)
        paramFileStr = f"{net_dir}/{paramFile[0]}"

        atFitFile = glob.glob1(net_dir,"*.dat")
        if len(atFitFile) != 1:
            # pylint: disable=W1203
            log.critical(f"atFitFile does not exist or is not uique: {atFitFile}")
            sys.exit(1)
        atFitFileStr = f"{net_dir}/{atFitFile[0]}"

        nEnsembles = len(glob.glob1(net_dir,"train[1-9]") + glob.glob1(net_dir,"train[1-9][0-9]"))
        if nEnsembles <= 0:
            # pylint: disable=W1203
            log.critical(f"No train* directories found in {net_dir}")
            sys.exit(1)

        # Number of networks
        self.nEnsembles = nEnsembles

        gpua = [gpuid[int(np.floor(i/(nEnsembles/len(gpuid))))] for i in range(nEnsembles)]

        # Construct pyNeuroChem class
        netWorkDir = f"{net_dir}/train%%s/networks/"
        self.nnp_ensembles = []
        for i in range(nEnsembles):
            print(f"paramFile={paramFileStr}, atFile={atFitFileStr}, net_dir={netWorkDir%str(i)}, gpu={gpua[i]}, sinet={sinet}")
            #self.nnp_ensembles.append(
            #    neuro.conformers(paramFile, atFitFile, netWorkDir % str(i), gpua[i], sinet))
            self.nnp_ensembles.append(
                neuro.conformers(paramFileStr, atFitFileStr, netWorkDir % str(i), gpua[i], False))

        if energyOutUnits is Units.KCAL:
            self.eFactor = 1./KCAL_MOL_to_AU      # if KCAL requested convert after calling NNP
            self.fFactor = 1./KCAL_MOL_to_AU      # if KCAL requested convert after calling NNP
        elif energyOutUnits is Units.AU:
            self.eFactor = 1.
            self.fFactor = bohr2ang               # if AU requested convert after calling NNP
        else:
            raise TypeError(f"Unknown energy unit: {energyOutUnits}")

        conf_file = glob.glob1(net_dir,"*.params")
        if len(conf_file) != 1:
            # pylint: disable=W1203
            log.critical(f"conf_file does not exist or is not uique: {conf_file}")
            sys.exit(1)
        conf_fileStr = f"{net_dir}/{conf_file[0]}"
        conf = io.read_file(conf_fileStr)
        atomTypes = re.search("Atyp *= *\\[([^]]+)]", conf).group(1) # type: ignore
        atomTypes = re.split(" *, *", atomTypes)
        atomTypes = [atom_info.NameToNum[at] for at in atomTypes]
        self.allowed_atom_num = frozenset(atomTypes)

        batch_by_atom_order = True # neurochem requires batches with same atom sequence
        mol_mem_GB    = 13
        memParam:Dict[str,float] = { "bytes_per_mol": 180 }

        super().__init__(atomTypes, outputGrads, compute_stdev, torch.float, mol_mem_GB,
                         memParam, batch_by_atom_order, self.device, energyOutUnits)


    def maxConfsPerBatch(self, nAtom:int) -> int:
        """
        :param nAtom: given a number of atoms compute how many conformations can be processed in one batch on the GPU

        May use self.MEM_GB and self.MEM_PARAM
        """
        if self.MEM_PARAM is None:
            return 100000
        return max(1,int(self.MEM_GB / self.MEM_PARAM['bytes_per_mol']))


    def create_energy_helper(self, coords_batch:SameSizeCoordsBatch,
                                  harm_constraint:Sequence[float] = None,
                                  fixed_atoms_list:List[bool] = None )  -> EnergyAndGradHelperInterface:
        """ Create an energy helper class that can compute energies and gradients for batches
            of conformations depending on the presence of constraints

            arguments:
            harm_constraint: list of harmonic constraint.
                                mols.shape[0] must be same size harm_constraint
        """

        energyHelper = EnergyAndGradHelper(self.nnp_ensembles, coords_batch, self.eFactor, self.fFactor)
        e_helper:EnergyAndGradHelperInterface = energyHelper

        if harm_constraint:
            if fixed_atoms_list is not None and len(fixed_atoms_list) > 0:
                raise TypeError("HarmConstraint not supported with additional constraints")
            e_helper = EnergyAndGradHelperHarmConstraint(energyHelper, harm_constraint)
        elif fixed_atoms_list is not None and len(fixed_atoms_list) > 0:
            fixed_atoms = torch.tensor(fixed_atoms_list, dtype=torch.uint8, device=self.device).bool()
            e_helper = EnergyAndGradHelperFixedAtoms(energyHelper, fixed_atoms)

        return e_helper


    def computeBatch(self, mols):
        """ Return iterator returning (mol,E, grad) for all mols

            Parameter:
            mols: array of Mol Objects, currently must be conformers of same molecule
        """

        class ComputeResultIterator():
            """ iterator over results """
            def __init__(self, nnpCmptr, mols):
                """
                Attributes:
                  _mol_batch    array of cdd_chem.mol.BaseMol's
                  _energy_batch np.array for float
                  _coords_batch np.
                """
                self._mol_batch    = mols
                self._std_batch    = None
                self._energy_batch = None
                self._force_batch  = None
                self._curentMol = 0
                self.outputGrads = nnpCmptr.outputGrads
                self.nnpCmptr = nnpCmptr

                nAtoms = mols[0].num_atoms
                nConfs = len(mols)
                nEnsembles = nnpCmptr.nEnsembles

                # convert coordinates to ANI input
                atomSymbols = mols[0].atom_symbols
                coordList = []
                for mol in mols:
                    assert nAtoms == mol.num_atoms
                    assert atomSymbols == mol.atom_symbols
                    xyz = mol.coordinates
                    coordList.append(xyz)   # ANI needs ANGSTROM, no conversion

                # run ANI and get energies and forces
                coordsA = np.array(coordList,dtype=np.float32)
                energies = np.zeros((nEnsembles, nConfs), dtype=np.float64)
                forces   = np.zeros((nEnsembles, nConfs, nAtoms, 3), dtype=np.float32)
                for i, nnp in enumerate(nnpCmptr.nnp_ensembles):
                    nnp.setConformers(confs=coordsA,types=atomSymbols)
                    E = nnp.energy().copy()
                    energies[i] = E

                    if nnpCmptr.outputGrads:
                        F = nnp.force().copy()
                        forces[i]   = -F

                self._energy_batch = np.mean(energies,axis=0) # Hatree
                self._force_batch  = np.mean(forces,axis=0)   # Hartree / A
                if nnpCmptr.compute_stdev:
                    self._std_batch = np.std(energies,ddof=1, axis=0)


                if self.nnpCmptr.eFactor != 1.:
                    self._energy_batch *= self.nnpCmptr.eFactor
                    if nnpCmptr.compute_stdev:
                        self._std_batch    *= self.nnpCmptr.eFactor
                if self.nnpCmptr.fFactor != 1.:
                    self._force_batch  *= self.nnpCmptr.fFactor


            def __iter__(self):
                return self

            def __next__(self):
                """ returns tuple molecule (unchanged, energy float, gradient(numpy))
                    gradient will be None unless the nnpCmpr was configured with outputGrads True
                """
                if self._curentMol >= len(self._mol_batch):
                    raise StopIteration

                mol  = self._mol_batch[self._curentMol]
                e    = self._energy_batch[self._curentMol]
                if self.nnpCmptr.compute_stdev:
                    std = self._std_batch[self._curentMol]
                else:
                    std = None

                grad = None
                if self.outputGrads:
                    grad   = self._force_batch[self._curentMol]

                self._curentMol += 1
                log.debug("coodsA=%s e=%.6f g=%s [%s]",
                    mol.coordinates.reshape(-1),e,grad,self.nnpCmptr.energyOutUnits)
                return mol,e,std,grad

        return ComputeResultIterator(self, mols)


class EnergyAndGradHelper(AbstractEnergyAndGradHelper):
    """ Helper to compute ANI Energies and gradients """

    def __init__(self, nnp_ensembles:List[Any], coords_batch:SameSizeCoordsBatch,
                 eFactor:float, fFactor:float):
        """
            Helper Class for computing energies adn gradients of a batch
            for conformations with same atom count and atom order.

            Arguments:
                nnp_ensembles: ANI models
                coords_batch: batch of coordinates
                eFactor: multiply energies with eFactor to get energies in correct unit
                fFactor: multiply forces with fFactor to get forces in correct unit
        """
        super().__init__(coords_batch, eFactor, fFactor)

        self._nnp_ensembles = nnp_ensembles
        self.energies_noconstraint = None

        self.coords_batch.coords.detach_()


    def _compute_energy(self, corrds_A:np.ndarray):
        ''' This will work only if the conformers have the same number
            and the same sequence of atom types.
        '''

        n_ensembles  = len(self._nnp_ensembles)
        n_confs      = self.coords_batch.n_confs
        atom_sym     = [ NumToName[at] for at in self.coords_batch.atom_types[0].cpu().numpy()]
        energies = np.zeros((n_ensembles, n_confs), dtype=np.float64)

        for i, nnp in enumerate(self._nnp_ensembles):
            nnp.setConformers(confs=corrds_A, types=atom_sym) # type: ignore
            E = nnp.energy().copy()                           # type: ignore
            energies[i] = E

        e = np.mean(energies,axis=0) # Hatree
        std = np.std(energies,axis=0) # Hatree
        if self.eFactor != 1.:
            e *= self.eFactor
            std *= self.eFactor

        # pylint: disable=E1102
        self.energies_noconstraint = torch.tensor(e, dtype=torch.float32,
                                                  device=self.coords_batch.coords.device)
        std = torch.tensor(std, dtype=torch.float32, device=self.coords_batch.coords.device)
        return self.energies_noconstraint, std


    def compute_energy(self) -> torch.tensor:
        corrds_A     = self.coords_batch.coords.to(dtype=torch.float32).cpu().numpy()
        return self._compute_energy(corrds_A)


    def compute_energy_with_filter(self, fltr:torch.tensor) -> torch.tensor:
        """
            Compute Energies for conformations for which fltr is 1
        """
        corrds_A     = self.coords_batch.coords[fltr].to(dtype=torch.float32).cpu().numpy()
        return self._compute_energy(corrds_A)


    def energy_no_constraint(self):
        """ Return energies without any imposed constraints
            To be overwritten
        """
        return self.energies_noconstraint


    def compute_grad(self):
        n_ensembles  = len(self._nnp_ensembles)
        n_confs      = self.coords_batch.n_confs
        n_at_in_conf = self.coords_batch.n_atom_per_conf
        forces   = np.zeros((n_ensembles, n_confs, n_at_in_conf, 3), dtype=np.float32)

        for i, nnp in enumerate(self._nnp_ensembles):
            F = nnp.force().copy()
            forces[i]   = -F

        grad  = np.mean(forces,axis=0)   # Hartree / A

        if self.fFactor != 1.:
            grad  *= self.fFactor

        return torch.tensor(grad, dtype=torch.float32, device=self.coords_batch.coords.device)
