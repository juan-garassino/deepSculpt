def add_grid(self):
    self.working_plane = self.void[:, :, self.section]
    self.color_parameters = self.return_axis()[1]
    # selection of the axis to work on

    if self.verbose == True:
        print(working_plane)
        print("###############################################################")

    column_height = random.randrange(
        self.element_grid_min, self.element_grid_max, self.step
    )
    element = np.ones(column_height).reshape(column_height, 1)

    x = np.arange(start=1, stop=49, step=9)
    y = np.arange(start=1, stop=49, step=9)

    z_base = np.zeros((6,))
    z_top = z_base + column_height

    grid_coor_base = np.array(np.meshgrid(x, y, z_base)).T[0]
    grid_coor_top = np.array(np.meshgrid(x, y, z_top)).T[0]
    grid_coor = np.concatenate((grid_coor_base, grid_coor_top), axis=2).astype(int)

    for column_row in grid_coor:
        for column in column_row:
            self.void[
                column[0] : column[3] + 1,
                column[1] : column[4] + 1,
                column[2] : column[5],
            ] = element.reshape((column_height,))

    for column_row in grid_coor:
        for column in column_row:
            self.color_void[
                column[0] : column[3] + 1,
                column[1] : column[4] + 1,
                column[2] : column[5],
            ] = self.color_edges

    if self.verbose == True:
        print(self.working_plane)
        print("###############################################################")

    return self.void, self.color_void
