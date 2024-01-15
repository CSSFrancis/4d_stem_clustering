PdCuSi Glass Structure Analysis
-------------------------------

Here we show how to use the pyxem package to perform a full structure analysis of a PdCuSi metallic glass from
a 4-D STEM dataset.

This can be largely broken down into the following steps:

1. Preprocess/filter the data
2. Find the positions of the diffraction peaks
3. Indentify dominate structures and symmetries in the diffraction peaks data
4. Analyze the data and extract the relevant information

If you want to run this example on your own machine, you will need to download the data from the following link:

> Insert link here


There are two different files linked.  The first is the raw 4-D STEM data, and the second
is the list of the diffraction peaks. The second file is much smaller and can be used to run the
`analysis.ipynb` and reproduce the results without having to download the full dataset.

The `find_peaks.ipynb` notebook shows how to find the positions of the diffraction peaks in the data and requires
the larger full dataset to run.  In this example we include some code for using dask-jobqueue to run the analysis
on a slurm cluster.  This is not required but can be useful for speeding up the analysis or in cases where the
ram usage is high.