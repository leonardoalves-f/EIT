import meshio
import numpy as np
import numpy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.io import loadmat

class Mesh:
    def __init__(self,path,rho_dict=None):
        self.path=path+'.msh'
        mesh_info=meshio.read(path+'.msh')

        self.point=mesh_info.points/1000 #from millimeters to meters
        self.triangle={'node':mesh_info.cells_dict['triangle'],
                       'type':mesh_info.cell_data['gmsh:geometrical'][0],
                       'n':len(mesh_info.cells_dict['triangle'])}
        self.tetra={'node':mesh_info.cells_dict['tetra'],
                    'type':mesh_info.cell_data['gmsh:geometrical'][1],
                    'n':len(mesh_info.cells_dict['tetra'])}

        if rho_dict is not None:
            self.define_rho(rho_dict)

        self.insert_virtual()
        self.calc_central_node()

    def insert_virtual(self):
        triangle=np.zeros((self.triangle['n'],4),dtype=int)

        for i in range(self.triangle['n']):
            triangle[i,0:3]=self.triangle['node'][i]
            triangle[i,3]=self.triangle['type'][i]+len(self.point)-1

        self.triangle['node']=triangle

    def calculate_centroid(self):
        triangle_centroid=np.zeros((3,self.triangle['n']))

        for i in range(self.triangle['n']):
            node_matrix=self.point[self.triangle['node'][i,0:3]]
            triangle_centroid[:,i]=np.sum(node_matrix,0)/3

        tetra_centroid=np.zeros((3,self.tetra['n']))

        for i in range(self.tetra['n']):
            node_matrix=self.point[self.tetra['node'][i]]
            tetra_centroid[:,i]=np.sum(node_matrix,0)/4

        self.triangle['centroid']=triangle_centroid
        self.tetra['centroid']=tetra_centroid

    def define_rho(self,rho_dict):
        rho_triangle=rho_dict['electrode']

        rho_tetra=np.zeros(self.tetra['n'])

        for i, tissue in enumerate(['scalp','skull','brain']):
            rho_tetra[self.tetra['type']==36+i]=rho_dict[tissue]

        self.triangle['rho']=rho_triangle
        self.tetra['rho']=rho_tetra

    def rho_brain(self,rho):
        elm_brain=self.tetra['type']==38
        self.tetra['rho'][elm_brain]=rho

    def calc_central_node(self):
        brain_node=np.unique(self.tetra['node'][self.tetra['type']==38])
        coord=self.point[brain_node]

        coord_max=np.max(coord,0)
        coord_min=np.min(coord,0)

        center=(coord_max+coord_min)/2
        distance=np.sum((coord-center)**2,1)

        central_index=np.argmin(distance)

        self.central_node=brain_node[central_index]

    def export(self,name_out):
        mesh_info=meshio.read(self.path)
        data=mesh_info.cell_data
        data['Resistivity']=[np.array(self.triangle['rho'])\
                            *np.ones(self.triangle['n']),
                             np.array(self.tetra['rho'])]

        mesh_out=meshio.Mesh(
            mesh_info.points,
            mesh_info.cells,
            cell_data=data)

        mesh_out.write(name_out+'.msh', file_format="gmsh22")

class DirectProblem:
    def __init__(self,mesh_path,rho_dict,val_current=0.001,skip_m=8):
        self.mesh=Mesh(mesh_path,rho_dict)
        self.rho_dict=rho_dict
        self.n_node=len(self.mesh.point)
        self.n_electrode=max(self.mesh.triangle['type'])
        self.n_elem=self.mesh.triangle['n']+self.mesh.tetra['n']
        self.voltage=np.zeros((self.n_node+self.n_electrode,32))
        self.val_current=val_current
        self.skip_m=skip_m
        self.current=self.current_matrix(val_current,skip_m)

    def current_matrix(self,val_current,skip_m):
        current=np.zeros((self.n_node+self.n_electrode,self.n_electrode))
        current[self.n_node,0]=1
        current[self.n_node+skip_m+1,0]=-1

        for i in range(1,self.n_electrode):
            current[self.n_node:self.n_node+self.n_electrode,i]=np.roll(
                        current[self.n_node:self.n_node+self.n_electrode,0],i)

        return val_current*current

    def calc_domain_stiffness(self,coord):
        det=lambda x: np.linalg.det(x)
        ones=lambda x: np.concatenate((np.ones((3,1)),x),1)

        v_coord=np.concatenate((coord,np.ones((4,1))),1)
        volume=np.abs(np.linalg.det(v_coord)/6)

        alpha1=det(coord[[1,2,3]])
        alpha2=-det(coord[[0,2,3]])
        alpha3=det(coord[[0,1,3]])
        alpha4=-det(coord[[0,1,2]])

        beta1=-det(ones(coord[[1,2,3]][:,[1,2]]))
        beta2=det(ones(coord[[0,2,3]][:,[1,2]]))
        beta3=-det(ones(coord[[0,1,3]][:,[1,2]]))
        beta4=det(ones(coord[[0,1,2]][:,[1,2]]))

        gamma1=det(ones(coord[[1,2,3]][:,[0,2]]))
        gamma2=-det(ones(coord[[0,2,3]][:,[0,2]]))
        gamma3=det(ones(coord[[0,1,3]][:,[0,2]]))
        gamma4=-det(ones(coord[[0,1,2]][:,[0,2]]))

        delta1=-det(ones(coord[[1,2,3]][:,[0,1]]))
        delta2=det(ones(coord[[0,2,3]][:,[0,1]]))
        delta3=-det(ones(coord[[0,1,3]][:,[0,1]]))
        delta4=det(ones(coord[[0,1,2]][:,[0,1]]))

        Y = [[beta1**2+gamma1**2+delta1**2,
              beta1*beta2 + gamma1*gamma2+ delta1*delta2,
              beta1*beta3 + gamma1*gamma3+ delta1*delta3,
              beta1*beta4 + gamma1*gamma4+ delta1*delta4],
             [beta1*beta2 + gamma1*gamma2+ delta1*delta2,
              beta2**2+gamma2**2+delta2**2,
              beta2*beta3 + gamma2*gamma3+ delta2*delta3,
              beta2*beta4 + gamma2*gamma4+ delta2*delta4],
             [beta1*beta3 + gamma1*gamma3+ delta1*delta3,
              beta2*beta3 + gamma2*gamma3+ delta2*delta3,
              beta3**2+gamma3**2+delta3**2,
              beta3*beta4 + gamma3*gamma4+ delta3*delta4],
             [beta1*beta4 + gamma1*gamma4+ delta1*delta4,
              beta2*beta4 + gamma2*gamma4+ delta2*delta4,
              beta3*beta4 + gamma3*gamma4+ delta3*delta4,
              beta4**2+gamma4**2+delta4**2]
             ]

        return Y/(36*volume)

    def calc_electrode_stiffness(self, coord):
        AB=[coord[1,0]-coord[0,0],coord[1,1]-coord[0,1],coord[1,2]-coord[0,2]]
        AC=[coord[2,0]-coord[0,0],coord[2,1]-coord[0,1],coord[2,2]-coord[0,2]]

        area=0.5*np.linalg.norm(np.cross(AB,AC))

        Y = area/3*np.array([[1,0,0,-1],
                             [0,1,0,-1],
                             [0,0,1,-1],
                             [-1,-1,-1,3]
                            ])

        return Y/self.mesh.triangle['rho']

    def calc_local_stiffness(self):
        domain_stiffness=np.zeros((4,4,self.mesh.tetra['n']))

        for i in range(self.mesh.tetra['n']):
            domain_stiffness[:,:,i]=self.calc_domain_stiffness(
                                self.mesh.point[self.mesh.tetra['node'][i]]
                                )

        electrode_stiffness=np.zeros((4,4,self.mesh.triangle['n']))

        for i in range(self.mesh.triangle['n']):
            electrode_stiffness[:,:,i]=self.calc_electrode_stiffness(
                            self.mesh.point[self.mesh.triangle['node'][i,0:3]]
                            )

        self.local_stiffness=np.concatenate((electrode_stiffness,
                                             domain_stiffness),2)

    def calc_global_stiffness(self):
        domain_rho = lambda rho, stiffness: (1/rho)*stiffness

        data=self.local_stiffness.copy()
        data[:,:,self.mesh.triangle['n']:]=domain_rho(self.mesh.tetra['rho'],
                                           data[:,:,self.mesh.triangle['n']:])
        data=data.transpose((2,0,1)).reshape(-1)

        ind = lambda n: np.concatenate((n,n,n,n),1).reshape(-1)

        i=ind(np.concatenate((self.mesh.triangle['node'],
                              self.mesh.tetra['node']),
                              0))
        j=ind(np.concatenate((self.mesh.triangle['node'],
                              self.mesh.tetra['node']),
                              0).reshape((-1,1)))

        data[i==self.mesh.central_node]=0
        data[j==self.mesh.central_node]=0

        diag=np.where((i==self.mesh.central_node)&
                      (j==self.mesh.central_node))[0]
        data[diag[0]]=1

        matrix_dim=self.n_electrode+self.n_node

        global_stiffness=sparse.coo_matrix((data,(i,j)),shape=(matrix_dim,
                                                               matrix_dim))

        self.global_stiffness=global_stiffness.tocsc()

    def solve(self):
        voltage=linalg.spsolve(self.global_stiffness,self.current)

        self.voltage=voltage

    def get_electrode_voltage(self):
        return self.voltage[len(self.mesh.point):,:]

class InverseProblem:
    def __init__(self,mesh_path,rho_dict,val_current=0.001,skip_m=8):
        self.direct=DirectProblem(mesh_path,rho_dict,val_current=0.001,skip_m=8)
        self.mesh=self.direct.mesh
        self.n_node=self.direct.n_node
        self.n_electrode=self.direct.n_electrode

    def calc_jacobian(self,memory_div=4):
        K_inv=linalg.spsolve(self.direct.global_stiffness,
                np.concatenate((np.zeros((self.n_node,self.n_electrode)),
                                np.eye(self.n_electrode)),0)).transpose()

        K=self.direct.local_stiffness[:,:,self.mesh.triangle['n']:]
        U=self.direct.voltage
        all_elm_brain=np.where(self.mesh.tetra['type']==38)[0]
        self.J=np.zeros((self.n_electrode**2,len(all_elm_brain)))

        div=int(len(all_elm_brain)/memory_div)

        for i in range(memory_div):
            elm_brain=all_elm_brain[i*div:i*div+div]
            coord=self.mesh.tetra['node'][elm_brain]

            add=np.arange(len(elm_brain)).reshape((-1,1))
            add=add*(self.n_node+self.n_electrode)
            add=np.repeat(add,4,1)
            coord_T=(coord+add).reshape((-1))

            T=np.zeros((len(elm_brain)*(self.n_node+self.n_electrode),
                        self.n_electrode))

            T[coord_T,:]=(-(1/(self.mesh.tetra['rho'][elm_brain]**2))\
                                .reshape((-1,1,1))\
                                *np.matmul(K[:,:,elm_brain].transpose((2,0,1)),
                                U[coord,:])).reshape((-1,32))

            T=T.reshape((len(elm_brain),self.n_node+self.n_electrode,
                                                    self.n_electrode))

            self.J[:,i*div:i*div+div]=-np.matmul(K_inv,T).transpose((0,2,1))\
                                      .reshape(len(elm_brain),-1).transpose()

    def load_priori(self,path):
        rho_priori=[]
        W_priori=[]

        for priori in path:
            parameters=loadmat(priori)
            rho_priori.append(parameters['rho'])
            W_priori.append(parameters['W'])

        self.rho_priori=np.array(rho_priori)
        self.W_priori=np.array(W_priori)

    def solve(self,iter,step,l):
        self.solve_voltage=[]
        self.solve_rho=[]

        self.solve_rho.append(self.direct.rho_dict['brain'].reshape(-1,1))
        measure=loadmat('PD_single_avcHem_refinado.mat')['U_final'][:,:,1]
        measure=measure.transpose().reshape((-1,1))
        approx_error=loadmat('approximation_error.mat')['mean_error']

        mult = lambda X,Y: np.matmul(X,Y)

        l=np.array(l).reshape((self.W_priori.shape[0],1,1))

        for i in range(iter):
            rho=self.solve_rho[i]

            self.direct.calc_global_stiffness()
            self.direct.solve()
            voltage=self.direct.get_electrode_voltage().transpose()\
                                                       .reshape((-1,1))
            self.solve_voltage.append(voltage)

            self.calc_jacobian()

            A_priori=np.sum(l*mult(self.W_priori,rho-self.rho_priori),0)
            B_priori=np.sum(l*self.W_priori,0)

            error=measure-voltage+approx_error

            A=mult(self.J.transpose(),error) - A_priori
            B=numpy.linalg.solve(mult(self.J.transpose(),self.J)+B_priori,A)

            self.solve_rho.append(rho+step*B)
            self.mesh.rho_brain(self.solve_rho[-1][:,0])
