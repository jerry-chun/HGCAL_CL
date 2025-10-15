This folder holds all scripts I have used to produce the final Layer Cluster data files and then train a simple GNN on.

The General pipeline to follow is labeled below with each subfolder containing a README.md with more information:

1, Produce hit level data:
    - In simulations folder build the package and produce hit level data. 
    - More information on this in the simulations folder

2, Apply CLUE2D to output LC level data:
    - Within LC_scripts folder
    - outputs same named events as hit level but with *_LC.root at end

3, Merge LC and hit level data into a single file:
    - within Files_merge folder
    - outputs same named events but with *_Full.root

4, Split data into train - validation - test:
    - Within Data Folder
    - Moves desried amounts into seperate folders, only able to split at the level of granularity of individual files

5, Train GNN:
    - Within training
    