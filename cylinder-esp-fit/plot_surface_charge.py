#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
combined = {}
for i in range(100):
    fig = plt.figure(0)
    for part in ["sheath","tube"]:
        if part not in combined:
            combined[part] = []
        fnm=part+'/dat/resp.{:05d}.dat'.format(i)
        dat = np.loadtxt(fnm)
        s = np.argsort(dat[:,1])
        #plt.plot(dat[:,1], dat[:,2])
        plt.axhline(0.0,color='black')
        # plt.xlim(10, 110)
        plt.ylim(-.25, .25)
        plt.plot(dat[:,1], dat[:,2] / ((99.7977 - 60.331)), label=part)
        fig.suptitle("Surface charge at r = {:6.1f}".format(dat[0][0]))
        mask = np.logical_and(dat[:,1] > 60.331, dat[:,1] < 99.7977)
        q = dat[mask].sum(axis=0)[2] / ((99.7977 - 60.331))
        combined[part].append([dat[0][0], q])
        print(part, "For this R , surface charge should be per disc should be ", dat[0][0], q)
        plt.legend(loc=1)
    plt.savefig('resp.{:05d}.png'.format(i))
    plt.clf()

xax = [x[0] for x in combined["sheath"]]
yax = [x[1] for x in combined["sheath"]]
fig = plt.figure()
plt.plot(xax, yax, label="Sheath")
xax = [x[0] for x in combined["tube"]]
yax = [x[1] for x in combined["tube"]]
plt.plot(xax, yax, label="Tube")
plt.legend()
fig.suptitle("T4-Phage surface charge")
plt.ylabel("Surface charge per disc (q/disc)")
plt.xlabel("Radius (Angstrom)")
plt.ylim(-.5, 1.4)
plt.savefig('charge.png')
