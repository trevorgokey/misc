#!/usr/bin/env python3
import numpy as np
from gridData import Grid
import matplotlib.pyplot as plt
import sys
from scipy.spatial.distance import cdist

print("Loading data file...")
# dat = Grid("pot.dx")
names = ["tube", "sheath", "tail"]
run_charge_density = True
run_resp = True
if run_resp:
    run_charge_density = False
dxname = "pot.dx"
writepdb = False
direction='out'
if run_charge_density:
    dxname = "charge.dx"
tube_fnm = "/home/tgokey/data/viral-tubes/apbs/r2-tail/run-apbs/" + names[0] + "/" + dxname
sheath_fnm = (
    "/home/tgokey/data/viral-tubes/apbs/r2-tail/run-apbs/" + names[1] + "/" + dxname
)
tail_fnm = "/home/tgokey/data/viral-tubes/apbs/r2-tail/run-apbs/" + names[2] + "/" + dxname
print("loading", tube_fnm)
tube = Grid(tube_fnm)

print("loaded shape", tube.grid.shape)
print("loading", sheath_fnm)
sheath = Grid(sheath_fnm)
print("loaded shape", sheath.grid.shape)
# print("loading", sheath_fnm)
# tail = Grid(tail_fnm)

# grid info
print("Generating initial points...")
print("Midpoints shape:", len(tube.midpoints), len(tube.midpoints[0]))
origin = np.array([tube.midpoints[i].mean() for i in range(len(tube.midpoints))])
print("origin", origin)
minz = np.min(tube.edges[2])  # - origin[2]
maxz = np.max(tube.edges[2])  # - origin[2]
minz = 75-(38.398/2)
maxz = 75+(38.398/2)
height = maxz - minz
rmax = np.max(tube.edges[0])  # - origin[0]
print(minz, maxz, rmax)

# grid sizes
theta = np.arange(0, np.pi * 2, 10 / 360, dtype=np.float32)
dz = 5.0
z = np.arange(minz, maxz, dz, dtype=np.float32)

if len(sys.argv) < 2:
    rlist = np.arange(0.5, 100, 0.5, dtype=np.float32)
    if direction == 'in':
        rlist = rlist[::-1]
    startidx=1
else:
    rlist = np.array([float(sys.argv[2])])
    startidx = int(sys.argv[1])


fig = plt.figure(figsize=(10, 5),dpi=100)
# fig = plt.figure(0)
ax = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

if writepdb:
    for name in names:
        surface_fname = "cylinder" + name + ".xyz"
        open(surface_fname, "w").close()
        surface_fname_pdb = "cylinder" + name + ".pdb"
        open(surface_fname_pdb, "w").close()
tube_r = []
sheath_r = []

def resp_torch(grid_xyz, grid_v, surface_xyz,cutoff=12.0):

    """
    Following Bayly's RESP 1993 paper
    Current does not actually using any restraints, so it is just an ESP fit
    """

    import torch
    # breakpoint()

    # breakpoint()
    K = len(grid_xyz)
    N = len(surface_xyz)
    # N = 1 
    # K = 2*N
    print("resp shapes: K,N", K, N)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    grid_xyz = torch.as_tensor(grid_xyz, dtype=torch.float32, device=torch.device('cuda'))
    surface_xyz = torch.as_tensor(surface_xyz,dtype=torch.float32, device=torch.device('cuda'))
    grid_v = torch.as_tensor(grid_v,dtype=torch.float32, device=torch.device('cuda'))
    cutoff = torch.as_tensor(cutoff, device=torch.device('cuda'))
    A = torch.zeros((K, N), dtype=torch.float32, device=torch.device('cuda'))
    B = torch.zeros((K, 1), dtype=torch.float32, device=torch.device('cuda'))

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)

    print("mem alloc:", elapsed_time_ms)

    grid_N = grid_xyz.shape[0]
    surface_N = surface_xyz.shape[0]

    for j in range(K):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        rij = torch.squeeze(torch.cdist(grid_xyz, torch.unsqueeze(grid_xyz[j], 0)))

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print("cdist 1:", elapsed_time_ms)

        #mask = torch.logical_and(rij > 0.0, rij < cutoff)

        # mask = rij > 0.0 #, rij < cutoff)
        if j % 100 == 0:
            print(j,grid_N) #torch.sum(mask), torch.numel(mask))
            # print(j,grid_N, torch.sum(mask), torch.numel(mask))
        #rij_ = rij[mask]

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()

        #B[j] = torch.sum(grid_v[mask]/rij_)
        B[j] = torch.sum(grid_v/rij)

        # end_event.record()
        # torch.cuda.synchronize()  # Wait for the events to be recorded!
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print("bj calc:", elapsed_time_ms)
        for k in range(N):
            # print("   ",j,torch.numel(grid_xyz),k,torch.numel(surface_xyz))
            if j == k:
                A[j][j] = torch.sum(1.0/torch.square(rij))
                #A[j][j] = torch.sum(1.0/torch.square(rij_))
            else:
                rik = torch.squeeze(torch.cdist(grid_xyz, torch.unsqueeze(surface_xyz[k], 0)))
                A[j][k] = torch.sum(1.0/(rij* rik))
                #A[j][k] = torch.sum(1.0/(rij_* rik[mask]))
    
    fit_charges = torch.lstsq(B,A)[0]
    if K > N:
        fit_charges = fit_charges[:N]
    

    chi2 = torch.tensor(0.0, device=torch.device('cuda'))
    for i in range(K):
        rij = torch.squeeze(torch.cdist(surface_xyz, torch.unsqueeze(grid_xyz[i],0)))
        vi = torch.sum(torch.div(fit_charges, rij))
        chi2 += torch.square(grid_v[i] - vi)**2
        
    # vhat = grid_v - np.array([fit_charges / cdist(surface_xyz, [grid_xyz[i]]) for i in range(K)])
    # chi2 = np.sum(vhat**2)
    rrms = torch.sqrt(torch.div(chi2, torch.sum(torch.square(grid_v))))
    print("RRMS RESP FIT:", rrms)

    return np.array(fit_charges.cpu())

def resp(grid_xyz, grid_v, surface_xyz,cutoff=12.0):

    """
    Following Bayly's RESP 1993 paper
    Current does not actually using any restraints, so it is just an ESP fit
    """
    from scipy.spatial.distance import cdist

    import time

    # breakpoint()
    K = len(grid_xyz)
    N = len(surface_xyz)
    print("resp shapes: K,N", K, N)

    A = np.zeros((K, N), dtype=np.float32)
    B = np.zeros((K, 1), dtype=np.float32)

    for j, pt_j in enumerate(grid_xyz):

        # now = time.clock()
        rij = cdist(grid_xyz, [pt_j]).flatten()
        # now = time.clock() - now
        # print("elapsed:",now*1000)
        mask = np.logical_and(rij > 0.0, rij < cutoff)
        if j % 1000 == 0:
            print(j,len(grid_xyz), mask.sum(), len(mask))
        rij = rij[mask]
        B[j] = np.sum(grid_v[mask] / rij)
        for k, pt_k in enumerate(surface_xyz):
            # print("   ",j,len(grid_xyz),k,len(surface_xyz))
            if j == k:
                A[j][j] = np.sum(1.0/rij**2.0)
            else:
                rik = cdist(grid_xyz, [pt_k]).flatten()
                A[j][k] = np.sum(1.0/(rij * rik[mask]))
    
    fit_charges = np.linalg.lstsq(A, B, rcond=None)[0]

    chi2 = 0.0
    for i in range(K):
        rij = cdist(surface_xyz, [grid_xyz[i]])
        vi = np.sum(fit_charges / rij)
        chi2 += (grid_v[i] - vi)**2
        
    # vhat = grid_v - np.array([fit_charges / cdist(surface_xyz, [grid_xyz[i]]) for i in range(K)])
    # chi2 = np.sum(vhat**2)
    rrms = (chi2 / (grid_v**2).sum())**.5
    print("RRMS RESP FIT:", rrms)

    return fit_charges

def pdbatom_factory():
    """
    """

    return {
        "type": "ATOM",
        "serial": 1,
        "name": "X",
        "altLoc": "0",
        "resName": "RES",
        "chainID": "X",
        "resSeq": 1,
        "iCode": "1",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "occupancy": 0.0,
        "tempFactor": 0.0,
        "element": "X",
        "charge": "0",
    }


def pdbatom_formated_str(in_str):
    """
    Returns a formatted ATOM record string
    """

    pdb_fmt = {
        "type": "{:6s}",
        "serial": "{:5d}",
        "space": "{:1s}",
        "name": "{:4s}",
        "altLoc": "{:1s}",
        "resName": "{:3s}",
        "chainID": "{:1s}",
        "resSeq": "{:4d}",
        "iCode": "{:1s}",
        "x": "{:8.3f}",
        "y": "{:8.3f}",
        "z": "{:8.3f}",
        "occupancy": "{:6.2f}",
        "tempFactor": "{:6.2f}",
        "element": "{:2s}",
        "charge": "{:2s}",
    }
    out_str = in_str.copy()
    out_str["space"] = " "
    order = [
        "type",
        "serial",
        "space",
        "name",
        "altLoc",
        "resName",
        "space",
        "chainID",
        "resSeq",
        "iCode",
        "space",
        "space",
        "space",
        "x",
        "y",
        "z",
        "occupancy",
        "tempFactor",
        "space",
        "space",
        "space",
        "space",
        "space",
        "space",
        "space",
        "space",
        "space",
        "space",
        "element",
        "charge",
    ]
    pdb_fmt_str = "".join([pdb_fmt[i] for i in order])
    val_str = [out_str[i] for i in order]
    return pdb_fmt_str.format(*val_str)


for i, r in enumerate(rlist,startidx):
    # for i, r in enumerate([42.0]):
    print(i, len(rlist), "generating...", end=" ")
    for k, dat in enumerate([tube, sheath]):
        if writepdb:
            surface_fname = "cylinder" + names[k] + ".xyz"
            surface_fname_pdb = '{:05d}'.format(i) + "cylinder" + names[k] + ".pdb"
            open(surface_fname_pdb, "w").close()

        if run_resp:
            x = r * np.cos(theta) + origin[0]
            y = r * np.sin(theta) + origin[1]
            cyl = None
            for j, step_z in enumerate(z):
                zcrd = np.repeat(step_z, len(x))
                xyz = np.vstack((x, y, zcrd)).T

                if cyl is None:
                    cyl = xyz
                else:
                    cyl = np.vstack((cyl,xyz))

            X,Y,Z= np.meshgrid(*dat.midpoints)
            grid_xyz = np.array([list(XYZ) for XYZ in zip(X.flat,Y.flat,Z.flat)])
            grid_v = np.array(dat.grid.flatten())

            surface = cyl
            cutoff = 4.0

            from scipy.spatial.distance import cdist
            mask = np.full(grid_xyz.shape[0], False)
            for j in range(len(grid_xyz)):
                if np.any(cdist([grid_xyz[j]], surface) < cutoff):
                    mask[j] = True

            # mask = np.any(cdist(grid_xyz, surface) < cutoff, axis=1)
            # print("reduced from/to", len(mask),len(mask) - mask.sum())
            grid_xyz = grid_xyz[mask]
            grid_v = grid_v[mask]
                
            q = resp(grid_xyz, grid_v, surface, cutoff=cutoff)
            print("q fit", q.mean(), q.std(), "shape", q.shape)
            
            # print("X:", x.shape, "Y:", y.shape, "Z:", z.shape)

            # len(x) is one slice along z (z is constant for x)
            q = q.reshape(len(x), -1)
            mean = q.mean(axis=0)
            stddev = q.std(axis=0)

                
        else:
            x = r * np.cos(theta) + origin[0]
            y = r * np.sin(theta) + origin[1]
            print("interpolating...", end=" ")
            mean = np.zeros(z.shape[0])
            stddev = np.zeros(z.shape[0])
            # cylinder_grid = np.arange(-.1,.11,.1)
            # cylinder_grid = np.array([0.0])
            N = z.shape[0]

            if writepdb:
                with open(surface_fname, "a") as fid:
                    fid.write(f"{len(x)*len(z)}\n\n")

            # for step_z in cylinder_grid:

            for j, step_z in enumerate(z):
                # layer[j] = np.array(
                #     [
                #         dat.interpolated(x, y, np.repeat(step_z, len(x)))
                #     ],
                #     dtype=np.float32,
                # )
                zcrd = np.repeat(step_z, len(x))
                xyz = np.vstack((x, y, zcrd))
                vals = dat.interpolated(*xyz)
                if run_charge_density:
                    mean[j] += np.sum(vals)
                    stddev[j] += np.var(vals)
                else:
                    mean[j] = vals.mean()
                    stddev[j] = vals.std()
                # print(x.min(), x.max(), y.min(), y.max(), step_z, mean.shape)
                # else:
                #     this_layer = np.array(
                #         [
                #             dat.interpolated(x, y, np.repeat(step_z, len(x))) / N
                #         ],
                #         dtype=np.float32,
                #     )
                #     layer.append( this_layer

                if writepdb:
                    with open(surface_fname, "a") as fid:
                        # print(j, len(z), xyz.shape, mean.shape)
                        [fid.write(f"H    {x:.4f} {y:.4f} {z:.4f}\n") for (x, y, z) in xyz.T]
                    with open(surface_fname_pdb, "a") as fid:
                        pdb_str = pdbatom_factory()
                        pdb_str["name"] = "H"
                        pdb_str["occupancy"] = r
                        for (ii, ((xi, yi, zi), v)) in enumerate(zip(xyz.T, vals), 1):
                            pdb_str["x"], pdb_str["y"], pdb_str["z"] = xi,yi,zi
                            pdb_str["tempFactor"] = v/10
                            pdb_str["serial"] = ii
                            fid.write(pdbatom_formated_str(pdb_str) + "\n")
                            fid.write("TER\n")
        if writepdb:
            with open(surface_fname_pdb, "a") as fid:
                fid.write("END\n")

        # mean = layer.mean(axis=0)
        # stddev = layer.std(axis=0)
        # print(mean)
        # print(stddev)
        # stddev =  np.array([
        #     dat.interpolated(x, y, np.repeat(z[i], len(x))).std() for i in range(len(z))
        # ], dtype=np.float32)

        print("plotting...")
        # ax.errorbar(z,mean,yerr=stddev)
        if run_charge_density:
            M2num = 0.0006  # conver molar to ptl per A^3
            # per z, plot the surface charge
            # make sure to use dz as the surface area element

            # mean is length z, with each val the sum of charges on a ring
            # also summed over r

            mean_per_ring = mean 

            # divide by dz to get the charge per cylindrical slice
            # mean_per_ring /= (2.0 * np.pi * r)

            # print(mean_per_ring.mean())
            ax.plot(z, mean_per_ring, label=names[k])
            ax.axhline(mean_per_ring.mean(), xmin=minz, xmax=maxz, label=names[k] + ' avg', color='r' if names[k] == 'tube' else 'g')
            if names[k] == 'tube':
                tube_r.append([mean_per_ring.mean(), mean_per_ring.std()])
                print("tube_r len is ", len(tube_r), len(rlist[:i]))
            else:
                sheath_r.append([mean_per_ring.mean(), mean_per_ring.std()])
        else:
            ax.plot(z, mean, label=names[k])
            print(names[k], "r=", r, "mean mean", mean.mean(), mean.std())
            ax.axhline(mean.mean(), xmin=minz, xmax=maxz, label=names[k] + ' avg', color='r' if names[k] == 'tube' else 'g')
            if names[k] == 'tube':
                tube_r.append([mean.mean(), mean.std()])
                print("tube_r len is ", len(tube_r), len(rlist[:i]))
            else:
                sheath_r.append([mean.mean(), mean.std()])
    ax.text(0, 0.95, "r={:6.2f}".format(r), transform=ax.transAxes)
    ax.legend(loc=1)
    if run_charge_density:
        pass
        # ax.set_ylim(-1e-7, 1e-7)
    else:
        pass
        # ax.set_ylim(-80, 60)
    ax.set_xlim(minz - 5, maxz + 5)
    ax.axhline(0.0, ls='--', color='k', alpha=.5)
    # ax2.plot(rlist[:i],np.array([x[0] for x in tube_r])/(height*2*np.pi*rlist[:i]), label='tube', color='red')
    # ax2.plot(rlist[:i],np.array([x[0] for x in sheath_r])/(height*2*np.pi*rlist[:i]), label='sheath', color='green')
    # ax2.plot(rlist[:i], np.array([x[0] for x in tube_r]), label='tube', color='red')
    # ax2.plot(rlist[:i], np.array([x[0] for x in sheath_r]), label='sheath', color='green')
    # ax2.legend(loc=1)
    # ax2.axhline(0.0, ls='--', color='k', alpha=.5)
    if run_charge_density:
        fig.savefig("charge.mean_r."+direction+".{:05d}.png".format(i))
    elif run_resp:
        fig.savefig("resp.mean_r."+direction+".{:05d}.png".format(i))
    else: 
        fig.savefig("pot.mean_r."+direction+".{:05d}.png".format(i))
    ax.cla()
    # ax2.cla()
print("====DONE====")
