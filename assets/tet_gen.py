import numpy as np
import pyvista as pv
import tetgen 
import pyacvd

# read 1.obj file
mesh = pv.read('assets/rectangular_prism.obj')
# print the number of faces
print(mesh)
clus = pyacvd.Clustering(mesh)
clus.cluster(800)

"""tet = tetgen.TetGen(mesh)
tet.tetrahedralize()
grid = tet.grid

# get cell centroids
cell_center = grid.cell_centers().points

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, "lightgrey", lighting=True, show_edges=True)
plotter.add_mesh(mesh, "r", "wireframe")
plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
plotter.show()"""
