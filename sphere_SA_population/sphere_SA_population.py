#!/usr/bin/env python3

import numpy as np
from matplotlib import animation
from matplotlib.dates import date2num
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib 

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, hxy)
    return np.array([r,theta,phi])

def sph2cart(r,theta,phi):
    rsin_theta = r * np.sin(theta)
    x = rsin_theta * np.cos(phi)
    y = rsin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def read_coordinates(filename):
    if type(filename) == str:
        return np.loadtxt(filename)
    return np.vstack([np.loadtxt(f) for f in filename])

def update_vis(i,self):
    batch=self.batch_size
    iterations=self.iterations
    polar = self.polar
    
    N = self.N
    hit = self.hit

    theta,phi = np.random.random((batch,2)).T
    theta = self.theta_min + (self.theta_max - self.theta_min)*theta
    phi   = self.phi_min + (self.phi_max - self.phi_min)*phi
    
    for (t,p) in zip(theta,phi):
        d = self.__class__.arclen(t,p,polar[1],polar[2], self.shell_radius)
        if (d < self.point_radius).any():
            hit += 1
            XYZ = sph2cart(self.shell_radius,t,p)
            x,y = self.__class__.cart_project_onto_disc(np.atleast_2d(XYZ))
            self.hitx_data.extend(x)
            self.hity_data.extend(y)
        else:
            XYZ = sph2cart(self.shell_radius,t,p)
            x,y = self.__class__.cart_project_onto_disc(np.atleast_2d(XYZ))
            self.missx_data.extend(x)
            self.missy_data.extend(y)
        
    self.vis_hit.set_data(self.hitx_data, self.hity_data)
    self.vis_miss.set_data(self.missx_data, self.missy_data)
    #x,y = self.cart_project_onto_disc(self.crd)
    #self.vis_data.set_data(x,y) 
    self.ax.set_xlim(
        min(self.hitx_data + self.missx_data),
        max(self.hitx_data + self.missx_data)
        )
    self.ax.set_ylim(
        min(self.hity_data + self.missy_data),
        max(self.hity_data + self.missy_data)
        )
    N += batch
    outstr = "r={:8.2f} {:12.8f} N={:d} hit%={:10.6e} iter={:8d}/{:8d}\n"
    print(outstr.format(
            self.shell_radius,
            hit/N * 4*np.pi*self.shell_radius,
            N,
            hit/N,
            i,
            iterations), 
        end='')
    self.N = N
    self.hit = hit
    return [self.vis_hit, self.vis_miss]#,self.vis_data]
    
class SphereSAPopulation:
    def __init__(self, crd, **kwargs):
        """
        """
        self.visual=False
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

    @staticmethod
    def arclen(t, p, data_theta, data_phi, r):
        central_angle = np.arccos(
            np.sin(data_theta)*np.sin(t) +
            np.cos(data_theta)*np.cos(t)*np.cos(p - data_phi))
        d = r * central_angle
        return d

    def run(self):
        
        if self.visual:
            return self.run_visual()

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
                d = __class__.arclen(t,p,polar[1],polar[2], self.shell_radius)
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

    @staticmethod
    def cart_project_onto_disc(crd):
        x = crd[:,0] / (1-crd[:,2])
        y = crd[:,1] / (1-crd[:,2])

        mag = np.sqrt((x**2 + y**2))
        maxmag = 3.0
        mask = mag > maxmag
        x[mask] = x[mask] / mag[mask] * maxmag
        y[mask] = y[mask] / mag[mask] * maxmag
        return x,y
        
    

    def run_visual(self):

        self.fig = plt.figure(figsize=(10, 10), dpi=120)
        rc("font", **{"size": 12})
        self.ax = self.fig.add_subplot(111)
        self.missx_data = []
        self.missy_data = []
        self.hitx_data = []
        self.hity_data = []
        self.vis_miss = self.ax.plot([], [], 'r.', ms=1)[0]
        # self.vis_miss = self.ax.scatter([0], [0], ',', ms=1, c='r')[0]
        self.vis_hit = self.ax.plot([], [], 'g.', ms=5)[0]
        x,y = self.cart_project_onto_disc(self.crd)
        self.vis_data = self.ax.plot(x,y, 'k,', ms=.1,alpha=.5)[0]
        #ax.set_ylim(-20, 20)
        self.polar = cart2sph(*self.crd.T)
        self.hit = 0
        self.N = 0
        update = 1
        ani = animation.FuncAnimation(self.fig, 
                update_vis, fargs=(self,),
            interval=update, blit=False, frames=self.iterations, repeat=False)
        plt.show()
        input()


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
    parser.add_argument('--theta-min', type=float)
    parser.add_argument('--theta-max', type=float)
    parser.add_argument('--phi-min', type=float)
    parser.add_argument('--phi-max', type=float)
    parser.add_argument('--point-radius', type=float)
    parser.add_argument('--shell-radius', type=float)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--visual', action="store_true")
    args = parser.parse_args()

    crd = read_coordinates(args.filename)
    obj = SphereSAPopulation( crd, **args.__dict__)
    obj.run()


if __name__ == "__main__":
    main()
