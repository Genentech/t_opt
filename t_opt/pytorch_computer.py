"""
Created on Feb 8, 2019

@author: albertgo
"""

import torch
import logging
import numpy as np
from typing import List, Sequence, Iterable, Tuple, Dict, Optional

from t_opt.abstract_NNP_computer import EnergyAndGradHelperHarmConstraint, EnergyAndGradHelperFixedAtoms,\
    AbstractNNPComputer, AbstractEnergyAndGradHelper,\
    EnergyAndGradHelperInterface
from t_opt.coordinates_batch import SameSizeCoordsBatch, CoordinateModelInterface
from t_opt.unit_conversion import Units


log = logging.getLogger(__name__)


class PytorchComputer(AbstractNNPComputer):
    """
        An NNP computer that can be used for pytorch based NNP's that implement the
        CoordinateModelInterface.
    """

    def __init__(self, model:CoordinateModelInterface,
                 allowed_atom_num:Iterable[int], atomization_energies:torch.tensor,
                 outputGrads:bool, compute_stdev:bool, coords_dtype,
                 mem_gb, memparam:Optional[Dict[str,float]], batch_by_atom_order,
                 eFactor:float = 1., fFactor:float = 1.,
                 energyOutUnits:Units = Units.KCAL ): # pylint: disable=R0913
        """
            Arguments:
            model:       pytorch module that takes SameSizeCoordsBatch and output energy
            allowed_atom_num: atomic numbers supported by this network
            atomization_energies: indexed by atomic number with atomization energies atoms up to max(allowed_atom_num)
            outputGrads: output gradients
            compute_stdev: compute standard deviation
            coords_dtype:   torch dtype of coordinates
            energyOutUnits: units of energies to output
            mol_mem_GB:     GB of memory available for computations
            bytes_per_mol:  approximate bytes required for each conformation
            batch_by_atom_order: if false input molecules are grouped by atom count
                                 if true in addtion they must have the same atom
                                    types in the same order
        """
        super().__init__(allowed_atom_num, outputGrads, compute_stdev, coords_dtype,
                         mem_gb, memparam, batch_by_atom_order, atomization_energies.device,
                         energyOutUnits)

        self.model = model
        self.allowed_atom_num     = frozenset(allowed_atom_num)
        self.atomization_energies = atomization_energies
        self.device = atomization_energies.device
        self.eFactor = eFactor
        self.fFactor = fFactor


    def maxConfsPerBatch(self, nAtom:int) -> int:
        """
        :param nAtom: given a number of atoms compute how many conformations can be processed in one batch on the GPU

        May use self.MEM_GB and self.MEM_PARAM

        Overwrite this to compute more accurate max number of conforamtion based on atom count
        """
        return 1000

    def create_energy_helper(self, coords_batch:SameSizeCoordsBatch,
                                  harm_constraint:Sequence[float] = None,
                                  fixed_atoms_list:List[bool]     = None ) -> EnergyAndGradHelperInterface:
        """ Create an energy helper class that can compute energies and gradients for batches
            of conformations depending on the presence of constraints

            arguments:
            harm_constraint: list of harmonic constraint.
                                mols.shape[0] must be same size harm_constraint
        """

        e_helper:AbstractEnergyAndGradHelper = \
            EnergyAndGradHelper(self.atomization_energies, self.model, coords_batch,
                                       self.eFactor,self.fFactor)
        e_helper_o: EnergyAndGradHelperInterface = e_helper

        if harm_constraint:
            dType  = self.atomization_energies.dtype
            harm_constraint = torch.tensor(harm_constraint,dtype=dType, device=self.device)
            if fixed_atoms_list is not None and len(fixed_atoms_list) > 0:
                raise TypeError("HarmConstraint not supported with additional constraints")
            e_helper_o = EnergyAndGradHelperHarmConstraint(e_helper, harm_constraint)
        elif fixed_atoms_list is not None and len(fixed_atoms_list) > 0:
            fixed_atoms = torch.tensor(np.array(fixed_atoms_list), dtype=torch.bool, device=self.device)
            e_helper_o = EnergyAndGradHelperFixedAtoms(e_helper, fixed_atoms)

        return e_helper_o



class EnergyAndGradHelper(AbstractEnergyAndGradHelper):
    """ Helper to compute Energy and gradients for Pytorch based NNP """

    def __init__(self, atomization_energies:torch.tensor,
                 model:CoordinateModelInterface,
                 coords_batch:SameSizeCoordsBatch,
                 eFactor:float, fFactor:float): # pylint: disable=R0913
        super().__init__(coords_batch, eFactor, fFactor)

        self._model = model     # could be ensemble model
        self.energies_noconstraint: torch.tensor
        self.nconf_ones = torch.ones(self.n_confs, dtype=atomization_energies.dtype, device=coords_batch.coords.device)
        self.atomization_energies = atomization_energies


    def compute_energy(self) -> Tuple[torch.tensor, torch.tensor]:
        self.coords_batch.zero_grad()

        self.energies_noconstraint, std = self._model.forward(self.coords_batch)
        self.energies_noconstraint += self.atomization_energies[self.coords_batch.atom_types].sum(dim=-1)
        if self.eFactor != 1.:
            self.energies_noconstraint *= self.eFactor
            if std is not None:
                std *= self.eFactor

        return self.energies_noconstraint, std


    def compute_energy_with_filter(self, fltr:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
            Compute Energies for conformations for which fltr is 1
            Note: you can not compute the gradient using this method!
        """
        self.coords_batch.coords[fltr].detach_()
        coords_batch = self.coords_batch.filter(fltr)

        coords_batch.zero_grad()

        e, std = self._model.forward(coords_batch)
        if self.eFactor != 1.:
            e *= self.eFactor
            if std is not None:
                std *= self.eFactor

        self.energies_noconstraint[fltr] = e
        return e, std


    def energy_no_constraint(self) -> torch.tensor:
        """ Return energies without any imposed constraints
            To be overwritten
        """
        return self.energies_noconstraint


    def compute_grad(self) -> torch.tensor:
        self.energies_noconstraint.backward(self.nconf_ones, retain_graph=False)
        self.energies_noconstraint.detach_()
        self.coords_batch.coords.grad.detach_()
        grad = self.coords_batch.coords.grad.data.clone()
        self.coords_batch.coords.grad.zero_()

        if self.fFactor != 1.: grad /= self.fFactor

        return grad

    def filter_(self, fltr:torch.tensor):
        super().filter_(fltr)
        self.nconf_ones = self.nconf_ones[fltr]



class DummyNet(CoordinateModelInterface):
    """
        A dummy pytoch module that computes a potential that pulls all atoms
        towards having coordinate = -0.703
    """

    def forward(self, same_size_coords_batch:SameSizeCoordsBatch) -> Tuple[torch.tensor,torch.tensor]:
        c = same_size_coords_batch.coords
        c = c*5
        e = c.pow(2) + c.exp()
        e = e.reshape(c.shape[0],-1).sum(-1)
        # min (y=(5x)^2 + e^(5x)) ~ y(-0.703) = 0.8272
        return e, e   # fake stdev with e, will not affect tests
