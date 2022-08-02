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
    )


def print_information(void, element, axis_selection, delta, section, top_left_corner, bottom_right_corner):
    print(f"void shape is: {np.array(void[0].shape)}")
    print(f"element shape is : {np.array(element.shape)}")
    print(f"the axis selection is: {axis_selection}")
    print(f"delta is: {delta}")
    print(f"section is: {section}")
    print(f"top left corner is: {top_left_corner}")
    print(f"bottom right corner is: {bottom_right_corner}")
    print(
        f"slices are: {top_left_corner[0]}:{bottom_right_corner[0]} and {top_left_corner[1]}:{bottom_right_corner[1]}"
    )
    print("###############################################################")
