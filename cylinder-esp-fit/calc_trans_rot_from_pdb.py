import json
import numpy as np
x = None
with open('out.json','rb') as fid:
    x = json.load(fid)
com = np.array(x)
print("Loaded COMs:", com.shape)

print("COMs are")
print(com)

print("Mean of each disc")
print(com[:6].mean(axis=0))
print(com[6:].mean(axis=0))

print("Translation is")
print((com[:6] - com[6:]).mean(axis=0))
xy = com[:,:2]

r = np.linalg.norm(xy, axis=1)
r.max() - r.min()

# this way did not work (cosine law)
if False:
    np.arccos((xy[:6] * xy[6:]).sum(axis=1)/(r[:6] * r[6:]))
    rads = np.arccos((xy[:6] * xy[6:]).sum(axis=1)/(r[:6] * r[6:]))
    np.rad2deg(rads)

# just measure from vmd
angles = [18.286, 18.260, 18.314, 18.294, 18.297, 18.333]

np.mean(angles)*2
