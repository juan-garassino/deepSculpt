import numpy as np
import matplotlib.pyplot as plt
import random
import time

class Sculptor():
    
    def __init__(self, void_dim,
                 n_edge_elements,
                 n_plane_elements,
                 n_volume_elements,
                 element_edge_min,
                 element_edge_max,
                 element_plane_min,
                 element_plane_max,
                 element_volume_min,
                 element_volume_max,
                 step,
                 verbose):
        
        self.void = np.zeros((void_dim, void_dim, void_dim))
        self.n_edge_elements = n_edge_elements
        self.n_plane_elements = n_plane_elements
        self.n_volume_elements = n_volume_elements
        self.style = "#ffffff"
        
        self.element_edge_min= element_edge_min
        self.element_edge_max = element_edge_max
        self.element_plane_min = element_plane_min
        self.element_plane_max = element_plane_max
        self.element_volume_min = element_volume_min
        self.element_volume_max = element_volume_max
        self.step = step
        
        self.verbose = verbose
        
        
    def return_axis(self):
        
        self.section = np.random.randint(low=0-1, high=self.void[0].shape[0])
        self.axis_selection = np.random.randint(low=0, high=3)
        
        if self.axis_selection == 0:
            self.working_plane = self.void[self.section,:,:]
        elif self.axis_selection == 1:
            self.working_plane = self.void[:,self.section,:]  
        elif self.axis_selection == 2:
            self.working_plane = self.void[:,:,self.section]
        else:
            print("error")
        return self.working_plane
    
    ### MAIN FUNCTIONS ###
    
    def add_edge(self): # element sizes
        
        self.working_plane = self.return_axis()
        # selection of the axis to work on
        if self.verbose == True:
            print(working_plane)
            print("###############################################################")

        #Variables
        self.edge_length = random.randrange(self.element_edge_min, self.element_edge_max, self.step) # estas variables quizas no necesiten ser self!!
        self.edge_plane = np.random.randint(low=0, high=2)

        if self.edge_plane == 0:
            self.element = np.ones(self.edge_length).reshape(self.edge_length,1)
        else:
            self.element = np.ones(self.edge_length).reshape(self.edge_length,1).T

        # creates the element to be inserted
        self.delta = np.array(self.working_plane.shape) - np.array(self.element.shape) 
        # finds the delta between the size of the void and the size of the element
        top_left_corner = (coor_i, coor_j) = (np.random.randint(low=0, high=self.delta[0]) , np.random.randint(low=0, high=self.delta[1]))
        # finds the coordinates of the top left corner
        top_left_corner = np.array(top_left_corner)
        # converts the result in an array
        bottom_right_corner = np.array(top_left_corner) + np.array(self.element.shape) #- np.array([1,1]))
        # finds the coordinates of the bottom right corner
        self.working_plane[top_left_corner[0]:bottom_right_corner[0] , top_left_corner[1]:bottom_right_corner[1]] = self.element
        # makes the slides using the coordinates equal to the element

        if self.verbose == True:
            print(self.working_plane)
            print("###############################################################")
    
    def add_plane(self): # element sizes
        
        self.element = None
        self.section = None
        self.delta = None
        self.top_left_corner = None
        self.bottom_right_corner = None
        self.working_plane = self.return_axis()
        if self.verbose == True:
            print(self.working_plane)
            print("###############################################################")

        #Variables
        self.element = np.ones((random.randrange(self.element_plane_min, self.element_plane_max, self.step), random.randrange(self.element_plane_min, self.element_plane_max, self.step)))
        # creates the element to be inserted
        self.delta = np.array(self.working_plane.shape) - np.array(self.element.shape) 
        # finds the delta between the size of the void and the size of the element
        self.top_left_corner = (coor_i, coor_j) = (np.random.randint(low=0, high=self.delta[0]) , np.random.randint(low=0, high=self.delta[1]))
        # finds the coordinates of the top left corner
        self.top_left_corner = np.array(self.top_left_corner)
        # converts the result in an array
        self.bottom_right_corner = np.array(self.top_left_corner) + np.array(self.element.shape) #- np.array([1,1]))
        # finds the coordinates of the bottom right corner
        self.working_plane[self.top_left_corner[0]:self.bottom_right_corner[0] , self.top_left_corner[1]:self.bottom_right_corner[1]] = self.element
        # makes the slides using the coordinates equal to the element

        if self.verbose == True:
            self.print_information()
            print("###############################################################")
            
        return self.void
    
    def add_pipe_cantilever(self):
        
        self.element = None
        self.working_plane = None
        self.delta = None
        self.top_left_corner = None
        self.bottom_right_corner = None
        self.axis_selection = np.random.randint(low=0, high=2)
        self.shape_selection = np.random.randint(low=0, high=2)
        self.depth = random.randrange(self.element_volume_min, self.element_volume_max, self.step)
        
        if self.verbose == True:
            print(self.working_plane)
            print("###############################################################")
            
        self.element = np.ones((random.randrange(self.element_volume_min, self.element_volume_max, self.step), random.randrange(self.element_volume_min, self.element_volume_max, self.step)))
        self.element = np.repeat(self.element, repeats=self.depth, axis=0).reshape((self.element.shape[0],self.element.shape[1],self.depth))

        self.element_void = np.zeros((self.element.shape[0]-2, self.element.shape[1]-2))
        self.element_void = np.repeat(self.element_void, repeats=self.depth).reshape((self.element_void.shape[0],self.element_void.shape[1],self.depth))

        # element[1:-1,1:-1,:] = element_void # elegir pasar el vacio o no como parte del volumen
        
        self.delta = np.array(self.void.shape) - np.array(self.element.shape) # ENCONTRAR LOS NUEVOS DELTAS

        corner_1 = np.array((np.random.randint(low=0, high=self.delta[0]) , np.random.randint(low=0, high=self.delta[1]), np.random.randint(low=0, high=self.delta[2])))
        corner_2 = np.array((corner_1[0] + self.element.shape[0], corner_1[1], corner_1[2]))
        corner_3 = np.array((corner_1[0], corner_1[1], corner_1[2] + self.element.shape[2]))
        corner_4 = np.array((corner_1[0] + self.element.shape[0], corner_1[1], corner_1[2] + self.element.shape[2]))
        
        corner_5 = np.array((corner_1[0], corner_1[1] + self.element.shape[1], corner_1[2]))
        corner_6 = np.array((corner_2[0], corner_2[1] + self.element.shape[1], corner_2[2]))
        corner_7 = np.array((corner_3[0], corner_3[1] + self.element.shape[1], corner_3[2]))
        corner_8 = np.array((corner_4[0], corner_4[1] + self.element.shape[1], corner_4[2]))
        
        # creates the floor and ceiling
        self.void[corner_3[0]:corner_8[0], corner_3[1]:corner_8[1], corner_3[2]-1] = self.element[:,:,0]
        self.void[corner_1[0]:corner_6[0], corner_1[1]:corner_6[1], corner_1[2]] = self.element[:,:,1]
        
        # creates de walls
        if self.shape_selection ==0:
            if self.axis_selection == 0:
                self.void[corner_1[0], corner_1[1]:corner_7[1], corner_1[2]:corner_7[2]] = self.element[0,:,:]
                self.void[corner_2[0]-1, corner_2[1]:corner_8[1], corner_2[2]:corner_8[2]] = self.element[1,:,:]
            else:
                self.void[corner_5[0]:corner_8[0], corner_5[1], corner_5[2]:corner_8[2]] = self.element[:,0,:]
                self.void[corner_1[0]:corner_4[0], corner_1[1], corner_1[2]:corner_4[2]] = self.element[:,0,:]
                
        else:
            if self.axis_selection == 0:
                self.void[corner_1[0], corner_1[1]:corner_7[1], corner_1[2]:corner_7[2]] = self.element[0,:,:]
                self.void[corner_5[0]:corner_8[0], corner_5[1], corner_5[2]:corner_8[2]] = self.element[:,0,:]
            else:
                self.void[corner_2[0]-1, corner_2[1]:corner_8[1], corner_2[2]:corner_8[2]] = self.element[1,:,:]
                self.void[corner_1[0]:corner_4[0], corner_1[1], corner_1[2]:corner_4[2]] = self.element[:,0,:]
        
        if self.verbose == True:
            self.print_information()
            print("###############################################################")
        
        return self.void
    
    def add_grill(self):
        pass
    
    ### ULTILS ###
    
    def print_information(self):
        print(f"void shape is: {np.array(self.void[0].shape)}")
        print(f"element shape is : {np.array(self.element.shape)}")
        print(f"the axis selection is: {self.axis_selection}")
        print(f"delta is: {self.delta}")
        print(f"section is: {self.section}")
        print(f"top left corner is: {self.top_left_corner}")
        print(f"bottom right corner is: {self.bottom_right_corner}")
        print(f"slices are: {self.top_left_corner[0]}:{self.bottom_right_corner[0]} and {self.top_left_corner[1]}:{self.bottom_right_corner[1]}")
        print("###############################################################")
    
    def plot_sections(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=6, nrows=int(np.ceil(self.void.shape[0]/6)), figsize=(25, 25), facecolor = (self.style))
        axes = axes.ravel() # flats
        for index in range(self.void.shape[0]):
            axes[index].imshow(sculpture[index,:,:], cmap = "gray")
            
    def plot_sculpture(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 25), facecolor = (self.style), subplot_kw=dict(projection="3d"))
        axes = axes.ravel() # flats
        for index in range(1):
            axes[index].voxels(sculpture, facecolors="orange", edgecolors="k", linewidth=0.05)
    
    ### GENERATOR ###
    
    def generative_sculpt(self):
        start = time.time()
        for edge in range(self.n_edge_elements):
            self.axis_selection = np.random.randint(low=0, high=3)
            self.add_edge()
            
        for plane in range(self.n_plane_elements):
            self.axis_selection = np.random.randint(low=0, high=3)
            self.add_plane()
            
        for volume in range(self.n_volume_elements):
            self.add_pipe_cantilever()
            
        print ('Time for sculptures is {} sec'.format(time.time()-start))
        return self.void