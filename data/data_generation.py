from tovpy import EOS
from tovpy import Units
uts = Units()
from tovpy import Utils
from scipy.constants import c, G
import numpy as np
import os

EOS_FILE_NAME = [
    '2B.rns',
    '2H.rns',
    'ALF2.rns',
    'BHB_lp_25-Sept-2017.rns',
    'BLQ_30-Jan-2020.rns',
    'BLh_gibbs_180_0.35_10-Jul-2019.rns',
    'DD2_22-Jun-2018.rns',
    'DD2_25-Sept-2017.rns',
    'ENG.rns',
    'H4.rns',
    'LS220B0.rns',
    'LS220B0v2.rns',
    'LS_220_25-Sept-2017.rns',
    'MPA1.rns',
    'MS1.rns',
    'MS1b.rns',
    'NL3_05-Oct-2017.rns',
    'SFHo+BL_01-Apr-2019.rns',
    'SFHo_09-Feb-2019.rns',
    'SFHo_25-Sept-2017.rns',
    'SLy.rns',
    'SLy4.rns',
    'SLy4_15-Jun-2018.rns',
    'TM1_05-Oct-2017.rns',
    'TMA_05-Oct-2017.rns',
    'eosA',
    'eosAPR_fit',
    'eosAU',
    'eosB',
    'eosC',
    'eosF',
    'eosFPS_fit',
    'eosFPS_old',
    'eosG',
    'eosL',
    'eosN',
    'eosNV',
    'eosO',
    'eosSLy_fit',
    'eosUU',
    'eosWS',
    'eos_DD2_adb.rns',
    'eos_SFHo_adb.rns',
    'eos_SFHx_adb.rns'
]
# Note that LS1800B0.rns, eosAPR, eosFP, eosFPS, eosSLy, eosWNV are not included in the list above
from tovpy import TOV
pc = np.logspace(-12, -8, 200)
for eos_file in EOS_FILE_NAME:
    print(eos_file)
    eos = EOS('tabular',name="from_file",filename=eos_file)
    module_dir = os.path.dirname(__file__)
    utils = Utils(eos=eos, path=module_dir, p = pc)
    leven, lodd = [2,3], [2,3]
    utils.Love_txt(leven=leven, lodd=lodd, filename=eos_file+'.txt')
 