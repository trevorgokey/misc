import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation

def pot_ene(xyz, charge, rmin, eps):
    E = 0.0
    for i in range(xyz.shape[0]):
        for j in range(i):
            r = tf.norm(xyz[i] - xyz[j]) # distance between atoms

            # The vdW term for the energy
            t = ((rmin[i]+rmin[j])/(r*2.0))**6
            vdw_ene = (eps[i]*eps[j])**.5 * (t**2 - t*2.0)

            # The electrostatic term
            coulomb_ene = charge[i]*charge[j]/r 

            E = E + coulomb_ene + vdw_ene
    return E

# Initial conditions
L = 3
dt = 0.4

Z,Y,X = np.mgrid[0:L:2, 0:L:2, 0:L:2]
xyz = tf.Variable(np.array([xyz for xyz in zip(X.flat,Y.flat,Z.flat)]), name='pos', dtype=tf.float32)
N = xyz.shape[0] 

eps = tf.tile([.6, .8], [N//2], name='eps')
rmin = tf.tile([1.2, 1.0],[N//2], name='rmin')
charge = tf.tile([-1.,1.], [N//2], name='charge')
vel = tf.Variable(tf.zeros((N,3)), name='vel')
m = tf.reshape(tf.tile([10.,8.],   [N//2], name='mass'), (N,1))


# plotting stuff
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ion = xyz[charge < 0.0].numpy()
cation = xyz[charge > 0.0].numpy()
ion_size = rmin[charge < 0.0].numpy()[0]
cation_size = rmin[charge > 0.0].numpy()[0]
ion_pts, = ax.plot(*(ion.T), marker="o", ms=10*ion_size, linestyle="", color='red') 
cation_pts, = ax.plot(*(cation.T), marker="+", ms=10*cation_size, linestyle="", color='blue') 

ax.axes.set_xlim3d(left=-2, right=L+2)
ax.axes.set_ylim3d(bottom=-2, top=L+2)
ax.axes.set_zlim3d(bottom=-2, top=L+2)
a = None
a2 = None

# the function that simulates and plays the simulation
def update_vis(i):
    global pts,fig
    global xyz, vel, rmin, eps, charge, m, a, a2

    # The simulation propagator
    xyz.assign_add(vel * dt + a * dt**2)
    with tf.GradientTape() as tape2:
        U = pot_ene(xyz, charge, rmin, eps)
        a2 = -1.0*tape2.gradient(U, xyz)/m
    vel.assign_add(.5 * (a - a2) * dt)
    a = tf.identity(a2)

    # write out simulation to file to visualize elsewhere
    with open("out.xyz", 'a') as fid:
        fid.write("{}\n\n".format(N))
        for pos,c in zip(xyz.numpy(),charge.numpy()):
            if c == 0.0:
                sym = 'H'
            else:
                sym = 'Na' if c > 0.0 else 'Cl'
            fid.write("{:4s} {:8.5f}  {:8.5f} {:8.5f}\n".format(sym, *pos))
        fid.flush()

    # for visualization
    ion = xyz[charge < 0.0].numpy()
    x,y,z = ion.T
    ion_pts.set_data(x,y)
    ion_pts.set_3d_properties(z)
    
    cation = xyz[charge > 0.0].numpy()
    x,y,z = cation.T

    cation_pts.set_data(x,y)
    cation_pts.set_3d_properties(z)

    # return the plots that need to be redrawn
    return [ion_pts, cation_pts]

# truncate simulation trajectory
open("out.xyz", 'w').close()

# calculate the initial acceleration
with tf.GradientTape() as tape1:
    U = pot_ene(xyz, charge, rmin, eps)
    a = -1.0*tape1.gradient(U, xyz)/m

ani = animation.FuncAnimation(fig, 
    update_vis, interval=1, blit=True, frames=1000, repeat=False)
plt.show()
