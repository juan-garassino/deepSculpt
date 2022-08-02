def add_plane(self):  # WHAT TO DO WITH THE WORKING PLANE PARAMETER

    self.element = None
    self.section = None
    self.delta = None
    self.top_left_corner = None
    self.bottom_right_corner = None
    self.working_plane = self.return_axis()[0]

    if self.verbose == True:
        print(self.working_plane)
        print("###############################################################")

    # Variables
    self.element = np.ones(
        (
            random.randrange(
                self.element_plane_min, self.element_plane_max, self.step
            ),
            random.randrange(
                self.element_plane_min, self.element_plane_max, self.step
            ),
        )
    )
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
    ] = self.color_planes

    if self.verbose == True:
        self.print_information()
        print("###############################################################")

    return self.void, self.color_void
