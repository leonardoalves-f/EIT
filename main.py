import base
from scipy.io import loadmat

# PARAMETERS
num_iterations=40
att_step=0.08
prior_lambda=[3.2e-14, 2e-4] #3.2e-14 8e-8
output_name='Hem_l2_40it_PrimD_2e-4'

# PATHS TO NECESSARY DATA
rho_brain=loadmat('prior_atlas.mat')['rho'][:,0]
rho_dict={'electrode':0.02,
          'scalp':2.7778,
          'skull':71.4286,
          'brain':rho_brain} # Initial resistivity

mesh_path='mesh_head_03_SEMSURFACE' # Mesh to solve inverse problem

#prior_path=['prior_atlas','prior_highPass']
prior_path=['prior_atlas','prior_primD'] # Prior information
                                            # Must contain keys 'W' and 'rho'

measure_path='measure_Hem' # Simulated measurements
                           # Must contain key 'U'

approxError_path='approximation_error' # Approximation error
                                       # Must contain key 'mean_error'
prior_norm = 'l1'

if(prior_norm == 'l2'):
    format='W'
elif(prior_norm == 'l1'):
    format='L'

# IMAGE RECONSTRUCTION
g=base.InverseProblem(mesh_path,rho_dict)
g.direct.calc_local_stiffness()
g.load_prior(prior_path,format)
g.load_measure(measure_path)
#g.add_noise(p=1)
g.load_approximation_error(approxError_path)
g.solve(num_iterations,att_step,prior_lambda,prior_norm)
g.mesh.export('rec_'+output_name)

# Export Difference between reconstruction and atlas
solution = base.Mesh('rec_'+output_name,have_rho=True)
atlas_mesh = base.Mesh('atlas_resistivity',have_rho=True)
solution.export_diff('diff_'+output_name, atlas_mesh)

"""
import matplotlib.pyplot as plt
g.load_measure(measure_path)
U = g.measure.copy()
g.add_noise()

electrode_plot = 8
plt.plot(g.measure[:,electrode_plot].T,linewidth=5)
plt.plot(U[:,electrode_plot].T,'--',linewidth=2)
#plt.plot(g.noise[:,electrode_plot].T,linewidth=5)
plt.show()
"""