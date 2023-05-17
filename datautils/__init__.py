from .ogbg import OGBG_DataModule
from .ogbg_nonce import OGBG_Nonce_DataModule
from .peptides import PeptidesDataModule
from .tu import TU_DataModule
from .ged import GED_DataModule
from .pcqm4mv2 import PCQM4Mv2_DataModule
from .CSL import CSL_DataModule
from .pcqmcontact import PCQMContactDataModule

__all__ = [
    'OGBG_DataModule',
    'OGBG_Nonce_DataModule',
    'PeptidesDataModule',
    'TU_DataModule',
    'GED_DataModule',
    'PCQM4Mv2_DataModule',
    'CSL_DataModule',
    'PCQMContactDataModule'
]
