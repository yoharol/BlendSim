import numpy as np
import pyvista as pv
import meshio

mesh_name = 'assets/stretch3D/1.mesh'

mesh = pv.read(mesh_name)
plotter = pv.Plotter()
plotter.add_mesh(mesh,color='yellow', show_edges=True)
plotter.show()