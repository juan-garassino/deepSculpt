def add_pipe_cantilever(self):  # THIS IS GOOD!!

    self.element = None
    self.working_plane = None
    self.delta = None
    self.top_left_corner = None
    self.bottom_right_corner = None
    self.axis_selection = np.random.randint(low=0, high=2)
    self.shape_selection = np.random.randint(low=0, high=2)
    self.depth = random.randrange(
        self.element_volume_min, self.element_volume_max, self.step
    )

    if self.verbose == True:
        print(self.working_plane)
        print("###############################################################")

    self.element = np.ones(
        (
            random.randrange(
                self.element_volume_min, self.element_volume_max, self.step
            ),
            random.randrange(
                self.element_volume_min, self.element_volume_max, self.step
            ),
        )
    )
    self.element = np.repeat(self.element, repeats=self.depth, axis=0).reshape(
        (self.element.shape[0], self.element.shape[1], self.depth)
    )

    self.element_void = np.zeros((self.element.shape[0] - 2, self.element.shape[1] - 2))
    self.element_void = np.repeat(self.element_void, repeats=self.depth).reshape(
        (self.element_void.shape[0], self.element_void.shape[1], self.depth)
    )

    # element[1:-1,1:-1,:] = element_void # elegir pasar el vacio o no como parte del volumen

    self.delta = np.array(self.void.shape) - np.array(
        self.element.shape
    )  # ENCONTRAR LOS NUEVOS DELTAS

    corner_1 = np.array(
        (
            np.random.randint(low=0, high=self.delta[0]),
            np.random.randint(low=0, high=self.delta[1]),
            np.random.randint(low=0, high=self.delta[2]),
        )
    )
    corner_2 = np.array((corner_1[0] + self.element.shape[0], corner_1[1], corner_1[2]))
    corner_3 = np.array((corner_1[0], corner_1[1], corner_1[2] + self.element.shape[2]))
    corner_4 = np.array(
        (
            corner_1[0] + self.element.shape[0],
            corner_1[1],
            corner_1[2] + self.element.shape[2],
        )
    )

    corner_5 = np.array((corner_1[0], corner_1[1] + self.element.shape[1], corner_1[2]))
    corner_6 = np.array((corner_2[0], corner_2[1] + self.element.shape[1], corner_2[2]))
    corner_7 = np.array((corner_3[0], corner_3[1] + self.element.shape[1], corner_3[2]))
    corner_8 = np.array((corner_4[0], corner_4[1] + self.element.shape[1], corner_4[2]))

    self.color_volume = np.random.randint(0, len(self.color_volumes))

    # creates the floor and ceiling
    self.void[
        corner_3[0] : corner_8[0], corner_3[1] : corner_8[1], corner_3[2] - 1
    ] = self.element[:, :, 0]
    self.color_void[
        corner_3[0] : corner_8[0], corner_3[1] : corner_8[1], corner_3[2] - 1
    ] = self.color_volumes[self.color_volume]

    self.void[
        corner_1[0] : corner_6[0], corner_1[1] : corner_6[1], corner_1[2]
    ] = self.element[:, :, 1]
    self.color_void[
        corner_1[0] : corner_6[0], corner_1[1] : corner_6[1], corner_1[2]
    ] = self.color_volumes[self.color_volume]

    # creates de walls
    if self.shape_selection == 0:
        if self.axis_selection == 0:
            self.void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = self.element[0, :, :]
            self.color_void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = self.color_volumes[self.color_volume]

            self.void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = self.element[1, :, :]
            self.color_void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = self.color_volumes[self.color_volume]
        else:
            self.void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = self.element[:, 0, :]
            self.color_void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = self.color_volumes[self.color_volume]

            self.void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = self.element[:, 0, :]
            self.color_void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = self.color_volumes[self.color_volume]

    else:
        if self.axis_selection == 0:
            self.void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = self.element[0, :, :]
            self.color_void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = self.color_volumes[self.color_volume]

            self.void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = self.element[:, 0, :]
            self.color_void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = self.color_volumes[self.color_volume]
        else:
            self.void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = self.element[1, :, :]
            self.color_void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = self.color_volumes[self.color_volume]

            self.void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = self.element[:, 0, :]
            self.color_void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = self.color_volumes[self.color_volume]

    if self.verbose == True:
        self.print_information()
        print("###############################################################")

    return self.void
