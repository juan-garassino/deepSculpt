def add_edge(self):  # WHAT TO DO WITH THE WORKING PLANE PARAMETER

    self.working_plane = self.return_axis()[0]
    self.color_parameters = self.return_axis()[1]
    # selection of the axis to work on

    if self.verbose == True:
        print(working_plane)
        print("###############################################################")

    # Variables
    self.edge_length = random.randrange(
        self.element_edge_min, self.element_edge_max, self.step
    )  # estas variables quizas no necesiten ser self!!
    self.edge_plane = np.random.randint(low=0, high=2)

    if self.edge_plane == 0:
        self.element = np.ones(self.edge_length).reshape(self.edge_length, 1)
    else:
        self.element = np.ones(self.edge_length).reshape(self.edge_length, 1).T

    # creates the element to be inserted
    self.delta = np.array(self.working_plane.shape) - np.array(self.element.shape)
    # finds the delta between the size of the void and the size of the element
    self.top_left_corner = np.array(
        (
            np.random.randint(low=0, high=self.delta[0]),
            np.random.randint(low=0, high=self.delta[1]),
        )
    )
    # finds the coordinates of the top left corner
    self.bottom_right_corner = np.array(self.top_left_corner) + np.array(
        self.element.shape
    )  # - np.array([1,1]))
    # finds the coordinates of the bottom right corner

    self.working_plane[
        self.top_left_corner[0] : self.bottom_right_corner[0],
        self.top_left_corner[1] : self.bottom_right_corner[1],
    ] = self.element
    # makes the slides using the coordinates equal to the element

    self.color_parameters[
        self.top_left_corner[0] : self.bottom_right_corner[0],
        self.top_left_corner[1] : self.bottom_right_corner[1],
    ] = self.color_edges

    if self.verbose == True:
        print(self.working_plane)
        print("###############################################################")

    return self.void, self.color_void
