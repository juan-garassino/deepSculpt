import numpy as np


def return_axis(void: np.ndarray, color_void: np.ndarray):  # -> tuple:
    """
    Selects a random plane from a 3D numpy array along a random axis.

    Args:
        void (np.ndarray): The 3D numpy array to select a plane from.
        color_void (np.ndarray): The 3D numpy array that holds the color information.

    Returns:
        tuple: A tuple containing:
            - working_plane (np.ndarray): The randomly selected plane.
            - color_parameters (np.ndarray): The color information of the selected plane.
            - section (int): The index of the selected plane.
    """
    section = np.random.randint(low=0, high=void.shape[0])
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
        print("Error: axis_selection value out of range.")

    return working_plane, color_parameters, section


def print_information(*args, **kwargs):
    """
    Print input arguments and keyword arguments in a formatted way.

    Args:
        *args: Positional arguments to be printed.
        **kwargs: Keyword arguments to be printed.

    Returns:
        None.
    """
    # Print separator and header for verbose output
    print("=" * 50)
    print("Verbose output:")
    print("-" * 50)

    # Print positional arguments, if any
    if args:
        print("Arguments:")
        for arg in args:
            print(f"  {arg}")

    # Print keyword arguments, if any
    if kwargs:
        print("Keyword arguments:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    # Print separator at end of verbose output
    print("=" * 50)
