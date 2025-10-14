import glob

input_dir = "/vols/cms/mm1221/geant4sim/simulations/build/output/"
output_dir = "/vols/cms/mm1221/Independent/LC_scripts/LC_out/photons_2"

inputs = sorted(glob.glob(f"{input_dir}/photons_3487738_*.root"))

with open("photons_2_pairs.txt", "w") as f:
    for inp in inputs:
        base = inp.split("/")[-1].replace(".root", "")
        outp = f"{output_dir}/{base}_LC.root"
        f.write(f"{inp} {outp}\n")

