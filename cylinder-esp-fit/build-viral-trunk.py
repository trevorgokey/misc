#!/usr/bin/env python3
import numpy as np
import sys

def_angle = 17.2 * np.pi / 180.0
def_translation = np.array([[0, 0, 40.6]])

# copies = 11
def_replicas = 23


def load_xyz(fnm):
    N = 0
    xyz = None
    with open(fnm, "r") as fd:
        N = int(fd.readline())
        fd.readline()
        xyz = np.empty((N, 3))
        sym = []
        for i in range(N):
            line = fd.readline().split()
            pos = line[1:]
            sym.append(line[0])
            xyz[i][:] = list(map(float, pos))
    return sym, xyz


def save_xyz(fnm, xyz, comment="", syms=None):
    out_str = "{:4s}    " + ("{:12.4f}" * 3) + "\n"
    if syms is None:
        syms = list(["C"] * len(xyz))
    with open(fnm, "w") as fd:
        fd.write("{:8d}\n".format(xyz.shape[0]))
        fd.write("{:s}\n".format(comment))
        [fd.write(out_str.format(s, *pos)) for s, pos in zip(syms, xyz)]


def rot_from_angle(angle):
    rotation = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation


class ViralSheathBuilder:
    def __init__(
        self,
        disc,
        mass,
        replicas=0,
        translate=[0.0, 0.0, 0.0],
        rotate=[0.0, 0.0, 0.0],
        align=False,
        center=False,
        quiet=False,
        verbose=False,
    ):
        self.disc = np.atleast_2d(disc)
        self.sheath = None
        self.replicas = replicas
        self.translate = np.array(translate)
        self.rotate = np.array(rotate)
        self.align = align
        self.center = center
        if mass is not None:
            self.mass = np.ones((self.disc.shape[0], 1))
        else:
            self.mass = mass
            assert mass.shape[0] == self.disc.shape[0], "Crds/Mass len mismatch"

        self.verbose = verbose
        self.quiet = quiet

    def center_of_mass(self):
        com = np.sum(self.mass / self.mass.sum() * self.disc, axis=0)
        return com

    def center_crd(self):
        com = self.center_of_mass()
        if self.verbose:
            print("COM: translated by", com)
        self.disc -= com

    def align_princpal_axes(self):
        I_mat = self._interia_tensor()
        _, prin_axes = np.linalg.eigh(I_mat)

        ca, cb, cc = prin_axes[:, 0]
        a, b, c = np.arccos([ca, cb, cc])

        R = self._rot_mat_xyz([a, b, c])

        # put the largest axis on z
        align_z = self._rot_mat_xyz([0.0, -np.pi/2, 0.0])

        if self.verbose:
            print("ROT: rotate using", R)
        self.disc[:, :] = np.dot(align_z,np.dot(R, self.disc.T)).T

    def _rot_mat(self, angles):
        (ca, cb, cc) = np.cos(angles)
        (sa, sb, sc) = np.sin(angles)

        rot_mat = np.array(
            [
                [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
                [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
                [-sb, cb * sc, cb * cc],
            ]
        )

        return rot_mat.T

    def _rot_mat_xyz(self, angles):
        rot = np.dot(
            self._rot_mat_x(angles[0]),
            np.dot(self._rot_mat_y(angles[1]), self._rot_mat_z(angles[2])),
        )
        return rot

    def _rot_mat_z(self, angle):
        ca = np.cos(angle)
        sa = np.sin(angle)

        rot_mat = np.array([[ca, -sa, 0], [ sa, ca, 0], [0, 0, 1],])

        return rot_mat

    def _rot_mat_y(self, angle):
        ca = np.cos(angle)
        sa = np.sin(angle)

        rot_mat = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca],])

        return rot_mat

    def _rot_mat_x(self, angle):
        ca = np.cos(angle)
        sa = np.sin(angle)

        rot_mat = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca],])

        return rot_mat

    def _interia_tensor(self):

        I_mat = np.zeros((3, 3))
        self.center_crd()
        for (x, y, z), m in zip(self.disc, self.mass):
            I_mat[0, 0] += m * (y * y + z * z)
            I_mat[0, 1] -= m * (x * y)
            I_mat[0, 2] -= m * (x * z)
            I_mat[1, 1] += m * (x * x + z * z)
            I_mat[1, 2] -= m * (y * z)
            I_mat[2, 2] += m * (x * x + y * y)

        I_mat[1, 0] = I_mat[0, 1]
        I_mat[2, 0] = I_mat[0, 2]
        I_mat[2, 1] = I_mat[1, 2]
        return I_mat

    def center_of_mass_per_chunk(self, chunk_len=1):
        mass_chunks = np.atleast_3d(self.mass).reshape(chunk_len, -1, 1 )
        crd_chunks = np.atleast_3d(self.disc).reshape(chunk_len, -1, 3)

        com_lst = []
        for i, (mass, chnk) in enumerate(zip(mass_chunks, crd_chunks)):
            com = np.sum(mass / mass.sum() * chnk, axis=0)
            com_lst.append(com)
        return np.array(com_lst)

    def print_com_by_chunk(self, chunk_len=1):
        import json
        chunks = self.center_of_mass_per_chunk(chunk_len)
        print("Chunk COMs:")
        print(chunks)
        with open("out.json", 'w') as fid:
            json.dump(chunks.tolist(), fid)

    def generate_sheath(self):
        """
        Calculate the sheath, storing in the variable sheath
        """

        if self.center:
            self.center_crd()
        if self.align:
            self.align_princpal_axes()
        self.sheath = np.zeros_like(self.disc).reshape((1, -1, 3))
        self.sheath = np.repeat(self.sheath, self.replicas, axis=0)
        self.sheath[0, :, :] = self.disc[:, :]
        for i in range(1, self.replicas+1):

            angles = self.rotate * i
            rot = self._rot_mat_xyz(angles)
            # rot = self._rot_mat(angles)
            if self.verbose:
                print("rot is", rot)
            disc = np.dot(rot, self.disc.T).T
            disc += i * self.translate
            self.sheath[i-1, :, :] = disc[:, :]

        self.sheath = self.sheath.reshape(np.prod(self.sheath.shape[:-1]), -1)
        # tube -= tube.mean( axis=0)
        # save_xyz(sys.argv[2], tube, comment="", atom="C")


def build_trunk(
    input_fnm,
    translate,
    rotate,
    replicas,
    monomers=1,
    align=True,
    center=True,
    quiet=False,
    verbose=False,
    out_xyz=None,
):

    mass_table = {
        "C": 12.000,
        "H": 1.000,
        "N": 14.000,
        "O": 16.0,
        "P": 31.00,
        "S": 32.00,
        "F": 19.0,
        "Cl": 35,
        "B": 11.00,
    }
    syms, disc = load_xyz(input_fnm)
    mass = np.atleast_2d([mass_table[s] for s in syms])

    vsb = ViralSheathBuilder(
        disc,
        mass,
        replicas,
        translate,
        rotate,
        align=align,
        center=center,
        quiet=quiet,
        verbose=verbose,
    )

    if verbose:
        vsb.print_com_by_chunk(monomers)

    vsb.generate_sheath()

    sheath = vsb.sheath
    if verbose:
        print("Sheath is now shape:", sheath.shape)

    if out_xyz is None and (not quiet):
        for line in sheath:
            print(line)
    else:
        if verbose:
            print("Saving xyz file:", out_xyz)
        rep_syms = np.tile(syms, vsb.replicas)
        save_xyz(out_xyz, sheath, comment="", syms=rep_syms)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a complete viral sheath from a single disc"
    )
    parser.add_argument(
        "input_xyz",
        metavar="input",
        type=str,
        help="input filename containing coordinates",
    )
    parser.add_argument("-o", "--out_xyz", type=str)
    parser.add_argument("-x", "--translate-x", type=float, default=0.0)
    parser.add_argument("-y", "--translate-y", type=float, default=0.0)
    parser.add_argument("-z", "--translate-z", type=float, default=0.0)
    parser.add_argument("-a", "--rotate-alpha", type=float, default=0.0)
    parser.add_argument("-b", "--rotate-beta", type=float, default=0.0)
    parser.add_argument("-c", "--rotate-gamma", type=float, default=0.0)
    parser.add_argument("-m", "--monomers", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    translate = [args.translate_x, args.translate_y, args.translate_z]
    rotate = np.radians([args.rotate_alpha, args.rotate_beta, args.rotate_gamma])

    replicas = args.replicas

    if args.verbose:
        print("translation in radians:", translate)
        print("rotate in radians:", rotate)

    build_trunk(
        args.input_xyz,
        translate,
        rotate,
        replicas,
        monomers=args.monomers,
        align=args.align,
        center=args.center,
        quiet=args.quiet,
        verbose=args.verbose,
        out_xyz=args.out_xyz,
    )


if __name__ == "__main__":
    main()
