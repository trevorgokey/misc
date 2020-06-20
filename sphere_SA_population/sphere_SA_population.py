#!/usr/bin/env python3

import numpy as np


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, hxy)
    return np.array([r,theta,phi])

def read_coordinates(filename):
    return np.vstack([np.loadtxt(f) for f in filename])
    
class SphereSAPopulation:
    def __init__(self, crd, **kwargs):
        """
        """
        self.batch_size = 10000
        self.iterations=10000
        self.crd = crd
        self.theta_min = 0
        self.theta_max = np.pi

        self.phi_min = 0.0
        self.phi_max = 2.0*np.pi

        self.point_radius=1.0
        self.shell_radius=1.0

        for k,v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
        print(self.__dict__)

    def run(self):

        polar = cart2sph(*self.crd.T)

        batch=self.batch_size
        iterations=self.iterations
        hit = 0
        i = 0
        N = 0
        while i < iterations:
            theta,phi = np.random.random((batch,2)).T
            theta = self.theta_min + (self.theta_max - self.theta_min)*theta
            phi   = self.phi_min + (self.phi_max - self.phi_min)*phi

            for (t,p) in zip(theta,phi): 
                central_angle = np.arccos(
                    np.sin(polar[1])*np.sin(t) +
                    np.cos(polar[1])*np.cos(t)*np.cos(np.abs(p - polar[2])%(2*np.pi)))
                d = self.shell_radius * central_angle
                if (d < self.point_radius).any():
                    hit += 1
            
            N += batch
            i += 1
            outstr = "r={:8.2f} {:12.8f} N={:d} hit%={:10.6e} iter={:8d}/{:8d}\n"
            print(outstr.format(
                    self.shell_radius,
                    hit/N * 4*np.pi*self.shell_radius,
                    N,
                    hit/N,
                    i,
                    iterations), 
                end='')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MC integration of a spherical shell')
    parser.add_argument(
        'filename',
        metavar='filename',
        type=str,
        nargs='+',
        help='input filename containing coordinates'
    )
    parser.add_argument( '--theta-min', type=float)
    parser.add_argument( '--theta-max', type=float)
    parser.add_argument( '--phi-min', type=float)
    parser.add_argument( '--phi-max', type=float)
    parser.add_argument( '--point-radius', type=float)
    parser.add_argument( '--shell-radius', type=float)
    parser.add_argument( '--iterations', type=int)
    parser.add_argument( '--batch-size', type=int)
    args = parser.parse_args()

    crd = read_coordinates(args.filename)
    obj = SphereSAPopulation( crd, **args.__dict__)
    obj.run()

if __name__ == "__main__":
    main()
