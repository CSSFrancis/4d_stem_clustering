Clustering Diffraction Vectors from Overlapping Structures in 4D STEM Data Sets
===============================================================================

Carter Francis| csfrancis@wisc.edu

This repository contains all of the code to reproduce the results in the paper

> Insert citation here

This github repository also has a Zenodo DOI for completeness sake an

The code is organized into three main folders:

1. toy_models:
    - Data generation for the toy models using a Al.cif and the diffsims (version 0.6.0) package
    - Clustering of the toy models using the pyxem package (version 0.17.0 or greater)
2. al_nano_crystals:
    - Clustering of the (real) Al nano-crystals using the pyxem package (version 0.17.0 or greater)
3. pdcusi_glass_structure:
    - Clustering of the (real) PdCuSi glass structure using the pyxem package (version 0.17.0 or greater)
    - Discussion about how the structure was determined as well as some potential drawbacks of the method

The data for the toy models is self contained and generated using the data_generation notebook. That data for the
al_nano_crystals and pdcusi_glass_structure is too large to be stored on github and must be downloaded separately.

The data for the al_nano_crystals can be downloaded from:

> https://acdc.alcf.anl.gov/mdf/detail/4dclustering_v1.1/

The data for the pdcusi_glass_structure can be downloaded from:

> https://acdc.alcf.anl.gov/mdf/detail/4dclustering_v1.1/

Smaller (perhaps more useful) versions of this data processing can be found in the pyxem documentation and
will easily run on a laptop.

> Small example for Glass Symmetry Analysis

> Small example for Nano-Crystal Segmentation Analysis
