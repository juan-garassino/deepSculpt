import numpy as np


def return_axis(void, color_void):

    section = np.random.randint(low=0 - 1, high=void[0].shape[0])

    axis_selection = np.random.randint(low=0, high=3)

    if axis_selection == 0:
        working_plane = void[section, :, :]
        color_parameters = color_void[section, :, :]
    elif axis_selection == 1:
        working_plane = void[:, section, :]
        color_parameters = color_void[:, section, :]
    elif axis_selection == 2:
        working_plane = void[:, :, section]
        color_parameters = color_void[:, :, section]
    else:
        print("error")

    return (
        working_plane,
        color_parameters,
        section,
    )


def print_information(
    void=None,
    element=None,
    axis_selection=None,
    delta=None,
    section=None,
    top_left_corner=None,
    bottom_right_corner=None,
):
    print(void)
    if void != None:
        print(f"void shape is: {np.array(void[0].shape)}")
    if element:
        print(f"element shape is : {np.array(element.shape)}")
    if axis_selection:
        print(f"the axis selection is: {axis_selection}")
    if delta:
        print(f"delta is: {delta}")
    if section:
        print(f"section is: {section}")
    if top_left_corner:
        print(f"top left corner is: {top_left_corner}")
    if bottom_right_corner:
        print(f"bottom right corner is: {bottom_right_corner}")
    if bottom_right_corner:
        print(
            f"slices are: {top_left_corner[0]}:{bottom_right_corner[0]} and {top_left_corner[1]}:{bottom_right_corner[1]}"
        )
    print("###############################################################")
