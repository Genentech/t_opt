# encoding: utf-8
"""
Factory class to hide the actual implementation of the NNP computer.

@author:     albertgo

@copyright:  2019 Genentech Inc.

"""

import logging
import os
import errno
from abc import ABCMeta
from abc import abstractmethod
import torch

from t_opt.unit_conversion import Units
from t_opt.pytorch_computer import PytorchComputer
from t_opt.abstract_NNP_computer import AbstractNNPComputer

log = logging.getLogger(__name__)


class NNPComputerFactoryInterface(metaclass=ABCMeta):
    """
        Interface for a Factory of NNP Computers.
        This class must be overwritten when adapting a new NNP computer to be used
        for the optimizer.
    """


    @abstractmethod
    def __init__(self, nnpName:str):
        """
            use the nnpName parameter to instantiate the correct NNPComputer.
        """


    @abstractmethod
    def createNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs) \
            -> AbstractNNPComputer :
        pass



class ExampleNNPComputerFactory(NNPComputerFactoryInterface):
    """ Example Factory class for NNP_Computer """

    def __init__(self, nnpName:str):
        """
            Factory class that creates an AbstractNNPComputer given a NNP name.
            If the nnpName is a directory name it will assume that the NNP configuration is
            for the neurochem implementation of ANI.
            If the nnPName is a file name if it will raise an exception
            If it is neither a directory nor a file name it will create a
            Dummy NNP just for demo purposes.
        """
        super().__init__(nnpName)

        # try path as given, if this does not work try prepending $NNP_PATH
        if not os.path.exists(nnpName):
            if os.path.exists(os.environ.get("NNP_PATH","") + "/" + nnpName):
                nnpName = os.environ.get("NNP_PATH","") + "/" + nnpName

        self.nnp_name = nnpName

        if os.path.isdir(nnpName):
            self.__setattr__("createNNP", self._createANI)

        elif os.path.isfile(nnpName):
            self.__setattr__("createNNP", self._otherNNP)

        elif self.nnp_name == 'torchani-ani1x':
            self.__setattr__("createNNP", self._ANI1xNNP)

        elif self.nnp_name == 'torchani-ani2x':
            self.__setattr__("createNNP", self._ANI2xNNP)

        else:
            self.__setattr__("createNNP", self._dummyNNP)



    def createNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """ to be replaced by specific implementation"""
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.nnp_name)


    def _createANI(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        # pylint: disable=C0415, W0613
        # import here to avoid dependency collision pytorch vs ANI
        from t_opt import ANI_computer
        return ANI_computer.ANIComputer(self.nnp_name,
                                        outputGrad, compute_stdev, energyOutUnits=energyOutUnits)


    def _ANI1xNNP(self, outputGrad: bool, compute_stdev: bool, energyOutUnits: Units = Units.KCAL, **kwArgs):
        # pylint: disable=C0415, W0613, R0201
        # import here to avoid dependency collision pytorch vs ANI
        """
           just a call to use ANI2x pytorch potential
        """
        from t_opt.torchani_computer import ANI1xNet

        log.warning('Using Torchani-ANI1x model!!!!!')

        net = ANI1xNet()
        atoms = net.atoms
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        atomization_e = torch.tensor([0] * (max(atoms) + 1), dtype=torch.float).to(device)

        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, None, False)


    def _ANI2xNNP(self, outputGrad: bool, compute_stdev: bool, energyOutUnits: Units = Units.KCAL, **kwArgs):
        # pylint: disable=C0415, W0613, R0201
        # import here to avoid dependency collision pytorch vs ANI
        """
           just a call to use ANI2x pytorch potential
        """
        from t_opt.torchani_computer import ANI2xNet

        log.warning('Using Torchani-ANI2x model!!!!!')

        net = ANI2xNet()
        atoms = net.atoms
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        atomization_e = torch.tensor([0] * (max(atoms) + 1), dtype=torch.float).to(device)

        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, None, False)


    def _dummyNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        # pylint: disable=W0613, R0201, C0415
        """
           just a dummy pytorch potential that pulls all atom coordiantes to -0.703
        """
        from t_opt.pytorch_computer import DummyNet

        log.warning('Using DummyNet just for testing!!!!!')

        net = DummyNet()
        atoms = [1,6,7,8,9,16,17]
        atomization_e = torch.tensor([0]*(max(atoms)+1), dtype=torch.float)

        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, None, False)

    def _otherNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """
           Here we could call the constructor for a NNP_Computer implemented differently than NeuroChem
        """

        raise NotImplementedError("No other NNP was implemented")
