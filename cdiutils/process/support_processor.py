import os
import numpy as np
from numpy.linalg import norm as npnorm
import math
from collections import deque
from scipy.ndimage import binary_fill_holes as fill_holes
from scipy.ndimage import binary_erosion as erosion
from collections import defaultdict
import shutil
import json

from cdiutils.utils import make_support

class SupportProcessor:
    """
    A class to bundle all functions needed to determine 
    the surface, the support, and analyze the facets.
    """
    
    def __init__(
            self,
            parameters: dict,
            data: np.ndarray=[],
            isosurface: float=0.5,
            nan_values: bool=False
    ) -> None:
        
        #Parameters
        self.params = parameters
        print(f"teset: {parameters}")
        self.isosurface = None
        
        self.amplitude_threshold = None 
        self.order_of_derivative = None
        self.derivative_threshold = None
        self.raw_process = None
        self.input_parameters = None
        
        if self.params["support_path"] is None:
            self.isosurface = isosurface
            self.order_of_derivative = self.params["order_of_derivative"]
            self.raw_process = self.params["raw_process"]

        #Global variables
        
        self.scan = None
        self.sample_name = None

        self.amplitude = None
        self.orthgonolized_data = None

        self.X, self.Y, self.Z = None, None, None
        
        if self.params["support_path"] is None:
            self.scan = self.params["metadata"]["scan"]
            self.sample_name = self.params["metadata"]["sample_name"]

            self.amplitude = data
            self.orthgonolized_data = None

            self.X, self.Y, self.Z = np.shape(self.amplitude)

        #Path
        self.dump_dir = self.params["metadata"]["dump_dir"]
        self.path_surface = None
        self.path_order = None

        if self.params["support_path"] is None:
            if self.params["method_det_support"] == "Amplitude_variation":
                self.path_surface = (f'{self.dump_dir}surface_calculation/'
                                     f'{self.params["method_det_support"]}/'
                )
            elif self.params["method_det_support"] == "Isosurface":
                self.path_surface = (f'{self.dump_dir}surface_calculation/'
                                     f'{self.params["method_det_support"]}'
                                     f'={self.params["isosurface"]}/'
                )

            else:
                raise ValueError("Unknown method_det_support. "
                                 "Use method_det_support='Amplitude_variation'"
                                 " or method_det_support='Isosurface'"
                )
            if self.params["method_det_support"] == "Amplitude_variation":
                if (self.order_of_derivative == "Gradient" 
                    or self.order_of_derivative == "Laplacian"
                ):
                    self.path_order = (f'{self.path_surface}'
                                           f'{self.order_of_derivative}/'
                    )
                else:
                    raise ValueError("Unknown order_of_derivative. "
                                     "Use order_of_derivative='Gradient'"
                                     " or order_of_derivative='Laplacian'"
                    )
            elif self.params["method_det_support"] == "Isosurface":
                self.path_order = self.path_surface

        #Intermediate data
        self.gradient = None
        self.xgrad, self.ygrad, self.zgrad = None, None, None 
        self.laplacian = None
        
        self.grad_val = None 
        self.grad_dir = None 
        
        self.surface_voxel_candidat = None
        self.pre_surface = None 
        self.surface_direction = None 
        
        self.smooth_dir = None
        self.f_presurface = None

        #Results
        
        self.support = None
        self.p_support = None
        self.surface = None
        self.p_surface = None
        
        
  
    def check_previous_data(self) -> None:
        """
        If one of these parameters (derivative_threshold, amplitude_threshold,
        flip) has changed since the previous run, 
        this method deletes the files affected.
        """

        if self.params["derivative_threshold"]==None:
            self.derivative_threshold = 0.2
        else: 
            self.derivative_threshold = self.params["derivative_threshold"]
            
        if self.params["amplitude_threshold"]==None:
            self.amplitude_threshold = 0.2
        else: 
            self.amplitude_threshold = self.params["amplitude_threshold"]

        self.input_parameters=[self.derivative_threshold, 
                               self.amplitude_threshold,
                               self.params["flip"]
        ]

        try: 
            with open(f'{self.path_surface}input_parameters.json', 'r') as f:
                self.previous_input_parameters = json.load(f)
                
            if self.previous_input_parameters[2] != self.input_parameters[2]:
                try:
                    shutil.rmtree(self.path_surface)
                    print(f'The folder at {self.path_surface}'
                          f'has been successfully removed.'
                    )
                except:
                    print(f'The folder {self.path_surface} doesn\'t exist.')
                
            elif self.params["method_det_support"] == "Amplitude_variation":
                if not np.array_equal(self.input_parameters[:2], 
                                      self.previous_input_parameters[:2]
                ):
                    print('amplitude_threshold or derivative_threshold'
                          'has been changed'
                    )
                    try:
                        shutil.rmtree(self.path_order)
                        print(f'The folder at {self.path_order}'
                              f'has been successfully removed.'
                        )
                    except:

                        print(f'The folder {self.path_order} doesn\'t exist.')
        except:
            print("No previous input_parameters found")

        os.makedirs(self.path_surface, exist_ok=True)
        
        with open(f'{self.path_surface}input_parameters.json', 'w') as f:
                json.dump(list(self.input_parameters), f, indent=2)
                
        os.makedirs(self.path_order, exist_ok=True)
        
        
### Determining the surface and the support
            
  
    def def_derivative_value_direction(self) -> None:
        """
        Calculate the gradient of the amplitude, its value and its direction
        in each voxel, and calculate the laplacian of the amplitude 
        if order_of_derivative="Laplacian".
        """

        try:
            self.gradient=np.load(f'{self.path_surface}gradient.npy')
            self.grad_dir=np.load(f'{self.path_surface}grad_dir.npy')
            self.grad_val=np.load(f'{self.path_surface}'
                                        f'grad_val.npy'
            )
            self.xgrad, self.ygrad, self.zgrad = self.gradient
            
        except:
            print('No previous grad_val and grad_dir found')

            self.gradient = np.gradient(self.amplitude)
            self.xgrad, self.ygrad, self.zgrad = self.gradient
            
            gradient_int = np.zeros((3, self.X, self.Y, self.Z))
            self.grad_dir = np.zeros((self.X, self.Y, self.Z, 3))
            self.grad_val = np.zeros((self.X, self.Y, self.Z))

            amplitude_max = np.max(self.amplitude)
            mask = self.amplitude > (self.amplitude_threshold 
                                     * amplitude_max)

            grad = np.array(self.gradient)
            norm = npnorm(grad, axis=0)
            
            gradient_int[:, mask] = grad[:, mask]
            gradient_int = np.moveaxis(gradient_int, 0, 3)

            self.grad_dir = np.ndarray.copy(gradient_int)
            
            self.grad_dir[mask, :] = (-self.grad_dir[mask, :] 
                                      / norm[mask, None]
            )

            self.grad_val[mask] = np.sum((gradient_int[mask] 
                                                * self.grad_dir[mask]),
                                                axis=-1
            )

            np.save(f'{self.path_surface}gradient.npy',
                    self.gradient
            )
            np.save(f'{self.path_surface}grad_dir.npy',
                    self.grad_dir
            )
            np.save(f'{self.path_surface}grad_val.npy',
                    self.grad_val
            )

        if self.order_of_derivative == "Laplacian":
            try:
                self.laplacian = np.load(f'{self.path_surface}/laplacian.npy')
            except FileNotFoundError:
                print("No previous laplacian found")
        
                def calculate_laplacian(vector_field):
                    F_x, F_y, F_z = vector_field
                    dF_x_dx = np.gradient(F_x, axis=0)
                    dF_y_dy = np.gradient(F_y, axis=1)
                    dF_z_dz = np.gradient(F_z, axis=2)
                    return dF_x_dx + dF_y_dy + dF_z_dz

                self.laplacian = calculate_laplacian(self.gradient)
                np.save(f'{self.path_surface}/laplacian.npy', 
                        self.laplacian
                )


    def def_surface_voxel_candidat(self) -> None:
        """
        Determines the voxels with a sufficiently large absolute value 
        of the gradient or laplacian, taking into account derivative_threshold,
        which could potentially belong to the surface.
        """

        try:
            self.surface_voxel_candidat = np.load(f'{self.path_order}'
                                                  f'surface_voxel_candidat.npy'
            )
        except FileNotFoundError:
            print('No previous surface_voxel_candidat found')

            if self.order_of_derivative == "Gradient":
                grad_min = np.min(self.grad_val)
                
                voxel_indices = np.argwhere(self.grad_val 
                                            < (self.derivative_threshold 
                                               * grad_min)
                )

                self.surface_voxel_candidat = voxel_indices.tolist()

            if self.order_of_derivative == "Laplacian":
                lapl_min = np.min(self.laplacian)
                voxel_indices = np.argwhere(self.laplacian 
                                            < (self.derivative_threshold 
                                               * lapl_min)
                )
                self.surface_voxel_candidat = voxel_indices.tolist()

            # Saving as numpy array
            np.save(f'{self.path_order}surface_voxel_candidat.npy', 
                    self.surface_voxel_candidat
            )


    def interpolate_voisins(self, i, j, k, dir, s):
        """
        Args:
            i,j,k (int) : position of a voxel
            dir (np.array): direction associated with this voxel
            s (int): The direction (forward or backward) in 
            which you wish to interpolate
        Returns:
            [float]: interpolation of the value of the derivative 
            (gradient or laplacian) at a unit distance 
            from the voxel in the given direction
        """
        coeff_voisins = np.zeros((2, 2, 2))
        voisins = np.zeros((2, 2, 2))
        dx, dy, dz = dir
        ux = s * np.sign(dx)
        uy = s * np.sign(dy)
        uz = s * np.sign(dz)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    if a==0:
                        coeff_x=1-abs(dx)
                    else:
                        coeff_x=abs(dx)
                    if b==0:
                        coeff_y=1-abs(dy)
                    else: 
                        coeff_y=abs(dy)
                    if c==0:
                        coeff_z=1-abs(dz)
                    else: 
                        coeff_z=abs(dz)

                    coeff_voisins[a][b][c] = coeff_x * coeff_y * coeff_z
                    
                    if self.order_of_derivative == "Gradient": 

                        gradxyz = (
                            self.xgrad[int(i + a * ux), 
                                       int(j + b * uy), 
                                       int(k + c * uz)] 
                            * dx
                            + self.ygrad[int(i + a * ux), 
                                        int(j + b * uy), 
                                        int(k + c * uz)] 
                            * dy
                            + self.zgrad[int(i + a * ux), 
                                         int(j + b * uy), 
                                         int(k + c * uz)] 
                            * dz
                        )
                        voisins[a][b][c] = (gradxyz 
                                            / math.sqrt(dx*dx + dy*dy + dz*dz)
                        )

                    elif self.order_of_derivative == "Laplacian": 
                        
                        laplacianxyz = self.laplacian[int(i + a * ux), 
                                                      int(j + b * uy), 
                                                      int(k + c * uz)
                        ]
                        
                        voisins[a][b][c] = laplacianxyz
                        
                    else: 
                        raise ValueError("Unknown order_of_derivative. " 
                                         "Use order_of_derivative='Gradient'"
                                         " or order_of_derivative='Laplacian'"
                        )

        interpolated = np.sum(coeff_voisins * voisins)
        facteur_normalisation = np.sum(coeff_voisins)

        if facteur_normalisation == 0:
            raise ValueError('facteur de normalisation=0')

        return interpolated / facteur_normalisation

    def interpolation(self, i, j, k, dir) -> [float, float]:
        """
        Args:
            i,j,k (int) : position of a voxel
            dir (np.array): direction associated with this voxel
        Returns:
            [float, float]: interpolation of the value of the derivative 
            (gradient or laplacian) forwards and backwards at a unit distance 
            from the voxel in the given direction
        """

        avant = self.interpolate_voisins(i, j, k, dir, 1)

        arriere = self.interpolate_voisins(i, j, k, dir, -1)

        return avant, arriere
    

    def def_preprocess_surface(self) -> None:
        """
        Defines the preprocess surface from the minimum of the derivative 
        (gradient or Laplacian) of the amplitude.
        """

        try:
            self.pre_surface = np.load(f'{self.path_order}/'
                                       f'preprocess_surface.npy')
            self.surface_direction = np.load(f'{self.path_order}/'
                                             f'surface_direction.npy')
        except FileNotFoundError:
            print('No previous preprocess_surface or surface_direction found')
            
            self.pre_surface = np.zeros((self.X, self.Y, self.Z))
            self.surface_direction = np.zeros((self.X, self.Y, self.Z, 3))

            for elem in self.surface_voxel_candidat:
                i, j, k = elem
                if 0 < i < self.X-1 and 0 < j < self.Y-1 and 0 < k < self.Z-1:
                    dir = self.grad_dir[i][j][k]
                    if self.order_of_derivative == "Gradient":
                        fw_grad, bw_grad = self.interpolation(i, j, k, dir)
                        condition_met = (self.grad_val[i][j][k] < fw_grad 
                                         and self.grad_val[i][j][k] < bw_grad)
                    elif self.order_of_derivative == "Laplacian":
                        fw_lapl, bw_lapl = self.interpolation(i, j, k, dir)
                        condition_met = (self.laplacian[i][j][k] < fw_lapl 
                                         and self.laplacian[i][j][k] < bw_lapl
                        )
                    else:
                        raise ValueError("Unknown order_of_derivative. "
                                         "Use order_of_derivative='Gradient' "
                                         "or order_of_derivative='Laplacian'")

                    if condition_met:
                        self.pre_surface[i][j][k] = 1
                        self.surface_direction[i][j][k] = dir
            

            np.save(f'{self.path_order}/preprocess_surface.npy', 
                    self.pre_surface
            )
            np.save(f'{self.path_order}/surface_direction.npy', 
                    self.surface_direction
            )


    def def_support(self, surface) -> None:
        
        """
        Fill the surface to obtain the support.
        """ 
        support=np.copy(surface)
            
        list_structure=[np.array([[[False,  False, False],
                            [ False,  True,  False],
                            [False,  False, False]], 
                            [[False,  True, False],
                            [ True,  True,  True],
                            [False,  True, False]],
                            [[False,  False, False],
                            [ False,  True,  False],
                            [False,  False, False]]], dtype=bool),

                        np.array([[[False,  True, False],
                            [ True, False,  True],
                            [False,  True, False]],
                            [[True,  False, True],
                            [ False,  True,  False],
                            [True,  False, True]],
                            [[False,  True, False],
                            [ True, False,  True],
                            [False,  True, False]]], dtype=bool)]               
       
        for struct in list_structure:
            support = fill_holes(support, structure=struct).astype(int)
            
        return support
    

    def def_smooth_dir(self) -> None:
        """
        For each voxel, average the directions of all adjacent voxels. 
        This defines a smooth direction for each voxel on the surface.
        """
  
        try:
            self.smooth_dir = np.load(f'{self.path_order}/smooth_grad_dir.npy')
            
        except:
            print('No previous smooth_grad_dir found')
            
            self.smooth_dir = np.zeros((self.X, self.Y, self.Z, 3))
            for x, y, z in zip(*np.nonzero(self.pre_surface)):
                voxel=np.array([x,y,z])
                smooth_dir=np.array([0,0,0])
                nb_voisins = 0
                # Define 27 possible offsets
                offsets = [(dx, dy, dz) for dx in [-1, 0, 1] 
                                        for dy in [-1, 0, 1] 
                                        for dz in [-1, 0, 1]
                ]

                for offset in offsets:
                    adj_voxel = tuple(x + y for x, y in zip(voxel, offset))
                    if self.pre_surface[adj_voxel]>0:
                        dir_adj_voxel=self.grad_dir[adj_voxel]
                        smooth_dir = smooth_dir + dir_adj_voxel
                        nb_voisins += 1
                smooth_dir /= nb_voisins
                self.smooth_dir[x,y,z] = smooth_dir/(npnorm(smooth_dir) 
                                                     if npnorm(smooth_dir) !=0 
                                                     else 1
                                                    ) 
                
            np.save(f'{self.path_order}/smooth_grad_dir.npy', self.smooth_dir)
            

    def distance_filled_surface(self,a,b,u) -> float:
        """
        Args:
            a,b : two voxels
            u : vector normal to the surface of the voxel under consideration
        Returns:
            float : Specific distance between voxel a and voxel b. 
            For this distance, two voxels are closer the more they 
            are aligned on the axis orthogonal to the surface (u).
        """

        
        u_unitaire = u / npnorm(u) #unit vector orthogonal to the facet

        # Calculate the vector product with the vector [0, 0, 1]
        # to obtain a vector collinear with the facet
        v = np.cross(np.append(u_unitaire, 0), [0, 0, 1])[:2] 

        # Normalize the resulting vector to make it unitary
        v_unitaire = v / npnorm(v)  #unit vector collinear with the facet
        
        N=max(self.X,self.Y,self.Z)   #larger data size
        
        return (abs(np.dot(b-a,v_unitaire)) 
                + (N+1)*npnorm( (b-a) - np.dot(b-a,v_unitaire)*v_unitaire )
        )
        #return npnorm(b-a)

    def permutation(self, vect, mark):
        a, b, c = vect
        if mark==0:
            return (a,b,c)
        elif mark==1:
            return (b,a,c)
        elif mark==2:
            return (c,a,b)
        

    def def_filled_slice(self, slice_surface, 
                         list_coord_contour, slice_direction, mark
    ) -> np.ndarray:
        """
        Args:
            slice_surface : slice of the surface to be filled
            list_coord_contour : list of surface voxel coordinates in the slice
            slice_direction : list of the normal direction of the surface voxel 
            in the slice
            mark : indicator that shows whether the slice is according 
            to x, y or z.
        Returns:
            np.ndarray : the filled slice
        """

        f_slice = np.copy(slice_surface)
        N2,N3=np.shape(slice_surface)
        for pixel0 in list_coord_contour:
            b0,c0=pixel0
            m1, m2, m3= self.permutation(slice_direction[b0][c0], mark)
            
            if max(abs(m2),abs(m3))>0:

                def test(b,c):
                    v = np.cross(np.append([m2, m3], 0), [0, 0, 1])[:2] 
                    dot_product = np.dot(v, [b-b0, c-c0])
                    if dot_product>0:
                        return 1
                    else:
                        return 0
                
                for t in range(2):

                    plus_proche_voisins=[]
                    min_dist=-1

                    # Start at the origin
                    origin = (b0, c0)
                    visited = set()
                    visited.add(origin)

                    # Define 9 possible offsets
                    offsets = [(db, dc) 
                               for db in [-1, 0, 1] 
                               for dc in [-1, 0, 1]
                    ]

                    # Use a queue for a breadth-first search
                    queue = deque()
                    queue.append(origin)

                    while len(queue)>0 and (min_dist==-1 or min_dist>1):
                        c_pixel = queue.popleft()
                        b,c=c_pixel
                        if 0<=b<N2 and 0<=c<N3:
                            if slice_surface[b][c]!=0:
                                current_distance=self.distance_filled_surface(
                                                 np.array(pixel0),
                                                 np.array(c_pixel),
                                                 np.array([m2,m3])
                                )
                                if min_dist==-1 and current_distance>0:
                                    min_dist=current_distance
                                    plus_proche_voisins.append(c_pixel)

                                elif min_dist==current_distance:
                                    plus_proche_voisins.append(c_pixel)

                            if min_dist == -1:
                                for offset in offsets:
                                    adj_pixel = tuple(x + y 
                                                      for x, y 
                                                      in zip(c_pixel, offset))
                                    if (adj_pixel not in visited 
                                        and 0<=adj_pixel[0]<N2 
                                        and 0<=adj_pixel[1]<N3
                                    ):
                                        if test(adj_pixel[0],adj_pixel[1])==t:
                                            visited.add(adj_pixel)
                                            queue.append(adj_pixel)

                    if min_dist>1:
                        for elem in plus_proche_voisins:
                            ub=((elem[0]-b0)
                                /math.sqrt((b0-elem[0])**2+(c0-elem[1])**2)
                            )
                            uc=((elem[1]-c0)
                                /math.sqrt(+(b0-elem[0])**2+(c0-elem[1])**2)
                            )                     

                            sorted_dir=sorted([(abs(ub),ub,0),(abs(uc),uc,1)], 
                                              reverse=True
                            )
                            u1=sorted_dir[0][1]
                            u2=sorted_dir[1][1]

                            e1=sorted_dir[0][2]

                            b_trajet, c_trajet=b0, c0
                            trajet=[b_trajet, c_trajet]

                            p=1
               
                            while elem[e1]-trajet[e1]!=0:
                                t=p/abs(u1)     
                                if e1==0:
                                    f_slice[round(b0+t*u1)][round(c0+t*u2)]=1
                                            
                                    b_trajet, c_trajet=b0+t*u1,c0+t*u2
                                            
                                else: 
                                    f_slice[round(b0+t*u2)][round(c0+t*u1)]=1
                                            
                                    b_trajet, c_trajet=b0+t*u2,c0+t*u1

                                trajet=[b_trajet,c_trajet]
                                p=p+1
                                        
                                        
        return f_slice  
        

    def def_filled_surface(self, 
                           surface_to_be_filled
    ) -> (np.ndarray, np.ndarray) :
        """
        Go through all the slices of the surface_to_be_filled according 
        to x, y and z and fill each slice to completely fill the surface.
        """

        f_surface=np.copy(surface_to_be_filled)

        for mark in range(3):
            if mark==0:    
                nonzero_indices = np.nonzero(surface_to_be_filled)
                list_coord_surface = {}
                for i, j, k in zip(*nonzero_indices):
                    list_coord_surface.setdefault(i, []).append([j, k])
                    list_coord_surface = dict(list_coord_surface)
            elif mark==1:
                nonzero_indices = np.nonzero(surface_to_be_filled)
                list_coord_surface = defaultdict(list)
                for i, j, k in zip(*nonzero_indices):
                    list_coord_surface.setdefault(j, []).append([i, k])
                    list_coord_surface = dict(list_coord_surface)
            elif mark==2:
                nonzero_indices = np.nonzero(surface_to_be_filled)
                list_coord_surface = defaultdict(list)
                for i, j, k in zip(*nonzero_indices):
                    list_coord_surface.setdefault(k, []).append([i, j])
                    list_coord_surface = dict(list_coord_surface)
                        
            for a in list_coord_surface:
                if mark==0:
                    slice_surface=f_surface[a,:,:]
                    slice_direction=self.smooth_dir[a,:,:]
                elif mark==1:
                    slice_surface=f_surface[:,a,:]
                    slice_direction=self.smooth_dir[:,a,:]
                elif mark==2:
                    slice_surface=f_surface[:,:,a]
                    slice_direction=self.smooth_dir[:,:,a]
                        
                list_coord_contour=list_coord_surface[a]
                    
                f_slice =self.def_filled_slice(slice_surface, 
                                                    list_coord_contour, 
                                                    slice_direction, mark
                )

                if mark==0:
                    f_surface[a,:,:]=f_slice
                elif mark==1:
                    f_surface[:,a,:]=f_slice
                elif mark==2:
                    f_surface[:,:,a]=f_slice
                    

        return f_surface
  
    
    def def_filled_presurface(self) -> None:
        """
       Fill the pre_surface.
        """
        try:
            self.f_presurface=np.load(f'{self.path_order}/'
                                      f'filled_presurface.npy')
        except:
            print('No previous filled_presurface found')

            f_presurface = self.def_filled_surface(self.pre_surface)
            self.f_presurface = f_presurface 
                    
            np.save(f'{self.path_order}/filled_presurface.npy', f_presurface)

### Pipeline 

    def support_calculation(self) -> np.ndarray :
        if self.params["support_path"] is None:
            self.check_previous_data()

            if self.params["method_det_support"]=='Isosurface':
                try:
                    self.support = np.load(f'{self.path_order}/support.npy')
                except:
                    print('No previous support found')
                    self.support = make_support(
                        self.amplitude,
                        isosurface=self.isosurface,
                        nan_values=False
                    )
                    np.save(f'{self.path_order}/support.npy', self.support)
                try:
                    self.surface=np.load(f'{self.path_order}/surface.npy')
                except:
                    print('No previous surface found')
                    self.surface = (self.support 
                                    - erosion(self.support).astype(int)
                    )
                    np.save(f'{self.path_order}/surface.npy', self.surface)

            elif self.params["method_det_support"]=='Amplitude_variation' :
                self.def_derivative_value_direction()
                self.def_surface_voxel_candidat()
                self.def_preprocess_surface()
                if self.raw_process:
                    try:
                        self.support=np.load(f'{self.path_order}/support.npy')  
                    except:
                        print('No previous support found')

                        self.support = self.def_support(self.pre_surface)
                        np.save(f'{self.path_order}/support.npy',
                                self.support
                        )
                    try:
                        self.surface=np.load(f'{self.path_order}/surface.npy')
                    except:
                        print('No previous surface found')
                        self.surface = (self.support 
                                        - erosion(self.support).astype(int)
                        )
                        np.save(f'{self.path_order}/surface.npy', 
                                self.surface
                        )

                else:
                    self.def_smooth_dir()
                    self.def_filled_presurface()
                    try:

                        self.p_support=np.load(f'{self.path_order}'
                                               f'/processed_support.npy'
                        )
                    except:
                        print('No previous processed_support found')
                        self.p_support = self.def_support(self.f_presurface)
                        np.save(f'{self.path_order}/processed_support.npy',
                                self.p_support
                        )
                    try:
                        self.p_surface=np.load(f'{self.path_order}/'
                                               f'processed_surface.npy')
                    except:
                        print('No previous processed_surface found')
                        self.p_surface = (self.p_support 
                                          - erosion(self.p_support).astype(int)
                        )
                        np.save(f'{self.path_order}/processed_surface.npy',
                                self.p_surface
                        )
                    self.support = self.p_support 
                    self.surface = self.p_surface
 
            else:
                raise ValueError("Unknown method_det_support. "
                                 "Use method_det_support='Isosurface' "
                                  "or method_det_support='Amplitude_variation'"
                )

        else :
            self.support = np.load(self.params["support_path"])
            self.surface = (self.support 
                            - erosion(self.support).astype(int)
            )
            os.makedirs(f'{self.dump_dir}/surface_calculation/', exist_ok=True)
            np.save(f'{self.dump_dir}/surface_calculation/support.npy',
                    self.support
            )
            np.save(f'{self.dump_dir}/surface_calculation/surface.npy', 
                    self.surface
            )
            
        return self.support, self.surface
