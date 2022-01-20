import base
from scipy.io import loadmat
import numpy as np

# PATHS TO NECESSARY DATA
rho_dict={'electrode':0.02,
          'scalp':2.7778,
          'skull':71.4286,
          'brain':5} # Initial resistivity

mesh_path='mesh_head_01_SEMSURFACE' # Mesh to solve inverse problem

mesh = base.Mesh(mesh_path,rho_dict)
mesh.calculate_centroid()

c_brain=mesh.tetra['centroid'][:,mesh.tetra['type']==38]
r_brain=mesh.tetra['rho'][mesh.tetra['type']==38]
centro = np.array([0.130,0.1,0.160])
raio = 0.02;

for i in range(len(r_brain)):
    if(np.sum((c_brain[:,i]-centro)**2/raio**2) < 1):
        r_brain[i]=2.5

noise = np.random.normal(0,0.25,len(r_brain))
mesh.rho_brain(r_brain+noise)

mesh.export('testando')
