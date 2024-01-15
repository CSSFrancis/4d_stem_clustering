Toy Models Analysis
-------------------

The toy models are used to test the performance of the different methods explored.

From an analysis stand point the PdCuSi Glass Structure or Al Nano-crystal experiments are much more
interesting/informative.  We've included the toy models here so that if you want to repeat/extend the
analysis you can do so without having to rewrite this code.  It also provides a good starting point
for understanding diffsims and how to create "toy" datasets from simple kinematic models.

There are a couple of notebooks that go with this data:

Notebooks:
----------

1. TestOverlappingDiffractionVectors
    This Notebook goes over how we tested for recall of diffraction vectors
2. TestOverlappingCrystalst
    This Notebook goes over how we tested for recall of nano crystals and making the figure
3. TestOverlappingSymetricCrystals
    This Notebook goes over how we tested for recall of nano crystals using a custom symmetry metric and making the
    figure
4. TestNoise
    This Notebook goes over how we tested for recall vs noise.


Extra Notebooks:
----------------

1. GraphicalMethod
    This just shows how the graphical method figure was made
2. AllProcesses
    This is an older version of a lot of the work but I kept it for completeness sake.  I wouldn't recommend using it "Here be Dragons"


Results:
--------

This folder contains the direct data that is used in the paper. Because of the Random nature of the analysis
the results will be slightly different each time you run it so we have included these results in case you
want to compare your results to ours. (If you are trying to compare please let us know/ Raise an Issue! We'd love to hear
about it!)

python scripts:
---------------

utils.py
    This contains a bunch of helper functions for plotting and analysis that I didn't want
    to clutter the notebooks with
