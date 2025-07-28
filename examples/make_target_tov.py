################################################################################

import os
import argparse
from sys import stderr
import numpy as np
from tovpy.utils import Target, msol
from tovpy.eos import EOS

################################################################################

parser = argparse.ArgumentParser(
    description="Computes a TOV sequence given an EOS table"
)
parser.add_argument("eos", help="EOS table in RNS format")
parser.add_argument("-m", "--m_target",  nargs='*', default=[], type=float,
                    help="Target TOV masses in Msol")
parser.add_argument("-p", "--pc_guess",  default=2e-9, type=float,
                    help="Initial pressure guess for maximum mass")
parser.add_argument("-o", "--output", default="tov.txt",
                    help="Output filename")
args = parser.parse_args()

################################################################################

eos = EOS("tabular", "", filename=args.eos)
leven = [2, 3]
lodd = [2, 3]
target = Target(eos=eos, leven=leven, lodd=lodd,
                ode_atol=1e-10, ode_rtol=1e-10, dhfact=-1e-12)

if not os.path.isfile(args.output):
  with open(args.output, 'a') as of:
      of.write(f"{'M':>11s} {'pc':>22s} {'C':>11s} {'R':>12s} {'k2':>14s} "
               f"{'k3':>14s} {'j2':>14s} {'j3':>14s}{'Lambda':>15s}\n")

def output_tov(tov, pc):
    print(f"pc={pc:10e} ", end='')
    M, R, C, k, h, j = tov.solve(pc)
    print(f"M={M/msol:.4f}")
    Lam = 2/3 * k[2]/C**5
    with open(args.output, 'a') as of:
        of.write(f"{M:11.8f} {pc:22.16e} {C:11.8f} {R:12.8f} {k[2]:14.8e} "
                 f"{k[3]:14.8e} {j[2]:14.8e} {j[3]:14.8e} {Lam:15.8f}\n")

def output_fail(Mtarg):
    print(f"pc={np.nan:10e} M={Mtarg/msol:.4f}")
    with open(args.output, 'a') as of:
        of.write(f"{Mtarg:11.8f} {np.nan:22.16e} {np.nan:11.8f} {np.nan:12.8f} "
                 f"{np.nan:14.8e} {np.nan:14.8e} {np.nan:14.8e} {np.nan:14.8e} "
                 f"{np.nan:15.8f}\n")

################################################################################

res = target.maximum_mass(p0=args.pc_guess, method='Nelder-Mead',
                          options={'fatol': 1e-5})
output_tov(target.tov, target.pc_max)

for m in args.m_target:
    try:
        pc_target = target.specific_mass(m, xtol=1e-5)
        output_tov(target.tov, pc_target)
    except ValueError as err:
        print(err, file=stderr)
        output_fail(m)
