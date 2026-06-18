import numpy as np
import awkward as ak



def reco_to_sim(hit_energies, rmask, cmask, purities):
    num = 0
    denom = 0
    for i in range(len(hit_energies)):
        if rmask[i]:
            if cmask[i]:
                num += min((1-purities[i])**2, 1) * (hit_energies[i])**2
                denom += hit_energies[i]**2
            else:
                num +=(hit_energies[i])**2
                denom += hit_energies[i]**2
    RtS = num/denom
    return RtS

