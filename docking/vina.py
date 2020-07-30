#!/usr/bin/env python3
import os
import subprocess
import sys
from subprocess import PIPE
from itertools import zip_longest
from multiprocessing import Pool

def call(args, verbose=True, **kwargs):
    stdin=None
    input=None
    if "input" in kwargs:
        input=kwargs["input"]
        kwargs.pop("input")
        stdin=PIPE

    p = subprocess.Popen(args.split(), stdin=stdin, stderr=PIPE, stdout=PIPE,
        universal_newlines=True, bufsize=1, **kwargs)
    out,err = p.communicate(input)
    if verbose:
        if out != "":
            print(out)

    if err != "":
        print("ERROR")
        print(err)

def extract_frame(i, parm_fnm, traj_fnm, force=False):
    pdb_fnm="prot.md.frame.{:d}.pdb".format(i)
    inp="""
    trajin {2:s} {0:d} {1:d}
    trajout {3:s} pdb
    go""".format(i, i, traj_fnm, pdb_fnm)

    cpptraj = "cpptraj -p {:s}".format(parm_fnm)
    if force or not os.path.exists(pdb_fnm):
        call(cpptraj, input=inp, verbose=True)
        print("Wrote", pdb_fnm)
    return pdb_fnm
    #print(out.stdout)

def to_pdbqt(in_fnm, out_fnm, rigid=True, force=False):
    rigid_flag = "-xr" if rigid else ""
    obabel="obabel -ipdb {:s} -opdbqt -O {:s} {:s}"
    obabel=obabel.format(in_fnm, out_fnm, rigid_flag)
    if force or not os.path.exists(out_fnm):
        call(obabel, verbose=False)
        print("Wrote", out_fnm)

def to_pdb(in_fnm, out_fnm):
    obabel="obabel -ipdbqt {:s} -opdb -O {:s}".format(in_fnm, out_fnm)
    print("Wrote", out_fnm)
    call(obabel, verbose=False)


def dock_structure(prot_fnm, lig_fnm, center, size, log=None):
    config = """
    receptor = {0:s}
    ligand = {1:s}

    out = {9:s}
    log = {2:s}

    center_x = {3:f}
    center_y = {4:f}
    center_z = {5:f}

    size_x = {6:f}
    size_y = {7:f}
    size_z = {8:f}

    exhaustiveness = 8
    num_modes = 20

    energy_range = 100
    cpu = 1
    """
    if log is None:
        log = "log"
    vina = "vina --config /dev/stdin"
    out = log.rsplit('.',1)[0] + '.pdbqt'
    inp = config.format(prot_fnm, lig_fnm, log, *center, *size, out)
    

    #breakpoint()
    call(vina, input=inp)
    
    print("Wrote", out, log)


def run_single(i, c, parm_fnm, traj_fnm, lig_fnm, center, size):

    lig_name = lig_fnm.rsplit('.',1)[0]
    
    print("\rExtracting frame {:7d}/{:-8d}".format(i, i), end="")
    #print("\r{:80s}".format(""), end="")
    pdb_fnm = extract_frame(i, parm_fnm, traj_fnm)
    pdbqt_fnm = pdb_fnm.rsplit('.', 1)[0] + '.pdbqt'


    print("\rConverting frame {:7d}/{:-8d}".format(i, i), end="")
    to_pdbqt(pdb_fnm, pdbqt_fnm, rigid=True)

    dockout = "\rDocking center {:d} {:8.6} {:8.6} {:8.6}, frame {:7d}/{:-8d}"
    print(dockout.format(c, *center,i, i), end="")
    #print("\r{:80s}".format(""), end='')
    log_fnm = pdb_fnm.rsplit('.', 1)[0]+'.c{:d}.{:s}.log'.format(c, lig_name)
    dock_structure(pdbqt_fnm, lig_fnm, center, size, log=log_fnm)

    out_fnm = pdb_fnm.rsplit('.', 1)[0]+'.c{:d}.{:s}.pdbqt'.format(c, lig_name)
    out_as_pdb_fnm = pdb_fnm.rsplit('.', 1)[0] \
        +'.c{:d}.{:s}.pdb'.format(c, lig_name)
    to_pdb(out_fnm, out_as_pdb_fnm)


# def dock(origin, grid_lens, frame, parm_fnm, traj_fnm, frame_start, frames=1):

#     if False:
#         with Pool() as pool:
#             work = []
#             for i in range(frame_start,frames+1):
#                 pdb_fnm = extract_frame(i, parm_fnm, traj_fnm)
#                 pdbqt_fnm = pdb_fnm.rsplit('.', 1)[0] + '.pdbqt'
#                 work.append(pool.apply_async(to_pdbqt, (pdb_fnm, pdbqt_fnm)))
#             out = [result.get() for result in work]
#             print(out)
#         exit()
#     if False:
#         run_single(1, 1, parm_fnm, traj_fnm, lig_fnm, centers[0], size)
#         exit()
#     if True:
#         with Pool() as pool:
#             work = []
#             for i in range(frame_start,frames+1):
#                 for c,center in enumerate(centers,1):
#                     args =  (i, c, parm_fnm, traj_fnm, lig_fnm, center, size)
#                     work.append(pool.apply_async(run_single, args))
#             out = [result.get() for result in work]
#         return out

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="""
        Dock a compound to an trajectory using OBABEL/CPPTRAJ/VINA""")

    parser.add_argument("--cx", "--center-x", type=float,
        description="x coordinate of the binding grid origin")
    parser.add_argument("--cy", "--center-y", type=float,
        description="y coordinate of the binding grid origin")
    parser.add_argument("--cz", "--center-z", type=float,
        description="z coordinate of the binding grid origin")

    parser.add_argument("--lx", "--length-x", type=float,
        description="x coordinate of the binding grid origin")
    parser.add_argument("--ly", "--length-y", type=float,
        description="y coordinate of the binding grid origin")
    parser.add_argument("--lz", "--length-z", type=float,
        description="z coordinate of the binding grid origin")

    parser.add_argument("-p", "--params", type=str,
        description="Parm7 file to give to cpptraj")
    parser.add_argument("-y", "--trajectory", type=str,
        description="Trajectory to load")
    parser.add_argument("-l", "--ligand", type=str,
        description="Ligand file in PDBQT format")

    parser.add_argument("-f", "--frame", type=str,
        description="Frame to extract from trajectory (1-index)")

    args = parser.parse_args()

    parm = args.params
    traj = args.trajectory
    lig = args.ligand
    center = [args.center_x, args.center_y, args.center_z]
    lens = [args.length_x, args.length_y, args.length_z]

    run_single(args.frame, args.frame, parm, traj, lig, center, lens)

    # centers=[
    #     [48.86,124.23,120.34],
    #     [146.86,132.23,120.34],
    #     [104.86,45.23,120.34],
    #     [100.19,100.56,130.34]
    # ]
    # size=[50,50,50]
    # frame_start = int(sys.argv[1])
    # frames = int(sys.argv[2])


    # parm_fnm = "top.noh.pdb"
    # traj_fnm = "traj.noh.pdb"
    # lig_fnm  = "lig2h.am1.pdbqt"