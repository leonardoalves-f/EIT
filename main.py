import base
from scipy.io import loadmat

# PARAMETERS
num_iterations=30
att_step=0.08
prior_lambda=[3.2e-14, 8e-8]
output_name='solution_isq_l2'

# PATHS TO NECESSARY DATA
rho_brain=loadmat('prior_atlas.mat')['rho'][:,0]
rho_dict={'electrode':0.02,
          'scalp':2.7778,
          'skull':71.4286,
          'brain':rho_brain} # Initial resistivity

mesh_path='mesh_head_03_SEMSURFACE' # Mesh to solve inverse problem

prior_path=['prior_atlas','prior_highPass'] # Prior information
                                            # Must contain keys 'W' and 'rho'

measure_path='measure_Isq' # Simulated measurements
                           # Must contain key 'U'

approxError_path='approximation_error' # Approximation error
                                       # Must contain key 'mean_error'

# IMAGE RECONSTRUCTION
g=base.InverseProblem(mesh_path,rho_dict)
g.direct.calc_local_stiffness()
g.load_prior(prior_path)
g.load_measure(measure_path)
g.load_approximation_error(approxError_path)
g.solve(num_iterations,att_step,prior_lambda)
g.mesh.export(output_name)
