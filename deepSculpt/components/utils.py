def return_axis(self):

    self.section = np.random.randint(low=0 - 1, high=self.void[0].shape[0])

    self.axis_selection = np.random.randint(low=0, high=3)

    if self.axis_selection == 0:
        self.working_plane = self.void[self.section, :, :]
        self.color_parameters = self.color_void[self.section, :, :]
    elif self.axis_selection == 1:
        self.working_plane = self.void[:, self.section, :]
        self.color_parameters = self.color_void[:, self.section, :]
    elif self.axis_selection == 2:
        self.working_plane = self.void[:, :, self.section]
        self.color_parameters = self.color_void[:, :, self.section]
    else:
        print("error")

    return (
        self.working_plane,
        self.color_parameters,
    )


def print_information(self):
    print(f"void shape is: {np.array(self.void[0].shape)}")
    print(f"element shape is : {np.array(self.element.shape)}")
    print(f"the axis selection is: {self.axis_selection}")
    print(f"delta is: {self.delta}")
    print(f"section is: {self.section}")
    print(f"top left corner is: {self.top_left_corner}")
    print(f"bottom right corner is: {self.bottom_right_corner}")
    print(
        f"slices are: {self.top_left_corner[0]}:{self.bottom_right_corner[0]} and {self.top_left_corner[1]}:{self.bottom_right_corner[1]}"
    )
    print("###############################################################")
