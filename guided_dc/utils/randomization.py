import logging

import numpy as np

log = logging.getLogger(__name__)


def randomize(value_range, operation, distribution, base=None):
    assert distribution in ["uniform", "normal"]
    if operation == "additive":
        return randomize_additive(base, value_range, distribution)
    elif operation == "absolute":
        return randomize_absolute(value_range, distribution)
    elif operation == "scaling":
        return randomize_scaling(base, value_range, distribution)
    else:
        raise ValueError("Invalid operation: {}".format(operation))


def randomize_array(value_range, operation, distribution, base=None):
    # assert isinstance(value_range, (list, list)), "value_range must be a list of lists."
    if base is not None:
        return [
            randomize(r, operation, distribution, b)
            for r, b in zip(value_range, base, strict=False)
        ]
    else:
        return [randomize(r, operation, distribution, base=None) for r in value_range]


def randomize_additive(base, value_range, distribution="uniform"):
    """
    Randomizes a value by adding a random value within the specified value_range.

    Parameters:
    - value_range (tuple): The range of the randomization.
    - distribution (str): The distribution of the randomization.

    Returns:
    - float: The randomized value.
    """
    if base is None:
        base = 0.0
        # log.warning("Base value is not provided. Defaulting to 0.")

    if distribution == "uniform":
        return base + np.random.uniform(value_range[0], value_range[1])
    elif distribution == "normal":
        return base + np.random.normal(value_range[0], value_range[1])


def randomize_absolute(value_range, distribution="uniform"):
    """
    Randomizes a value within the specified value_range.

    Parameters:
    - value_range (tuple): The range of the randomization.
    - distribution (str): The distribution of the randomization.

    Returns:
    - float: The randomized value.
    """

    if distribution == "uniform":
        return np.random.uniform(value_range[0], value_range[1])
    elif distribution == "normal":
        return np.random.normal(value_range[0], value_range[1])


def randomize_scaling(base, value_range, distribution="uniform"):
    """
    Randomizes a value by scaling it by a random value within the specified value_range.

    Parameters:
    - base (float): The base value to be randomized.
    - value_range (tuple): The range of the randomization.
    - distribution (str): The distribution of the randomization.

    Returns:
    - float: The randomized value.
    """
    if base is None:
        base = 1.0
        # log.warning("Base value is not provided. Defaulting to 1.")

    if distribution == "uniform":
        return base * np.random.uniform(value_range[0], value_range[1])
    elif distribution == "normal":
        return base * np.random.normal(value_range[0], value_range[1])


def randomize_uv(top_uv, aspect_ratio, shift_range=0.1):
    """
    Randomizes the UV coordinates for a periodic texture, scaling for non-square images.

    Parameters:
    - top_uv (np.array): The original UV coordinates to be randomized.
    - image_width (int): The width of the texture image.
    - image_height (int): The height of the texture image.
    - shift_range (float): The maximum shift value for randomization.

    Returns:
    - np.array: The randomized UV coordinates.
    """

    # Determine the range of UV coordinates
    uv_min, uv_max = top_uv.min(axis=0), top_uv.max(axis=0)
    step_size = uv_max - uv_min

    # Adjust step size for aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        step_size[1] *= aspect_ratio
    else:  # Taller than wide
        step_size[0] /= aspect_ratio

    # Randomize the starting point
    random_shift = np.random.uniform(0, shift_range, size=2)

    # Shift the UV coordinates
    randomized_uv = top_uv - uv_min
    randomized_uv = randomized_uv / step_size
    randomized_uv = (
        randomized_uv + random_shift
    ) % 1.0  # Wrap around the texture space

    return randomized_uv
