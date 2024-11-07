from typing import List


def point_to_xml(point: List[float], description: str = "") -> str:
    """Converts a point coordinate and description into XML format.

    Args:
        point (List[float]): A list containing x,y coordinates as floats.
        description (str, optional): Description text for the point. Defaults to "".

    Returns:
        str: XML string representing the point with coordinates and description.
    """
    x, y = point
    return f' <point x="{x:.1f}" y="{y:.1f}" alt="{description}">{description}</point>'


def normalize_point(point: List[float], resolution: List[int]) -> List[float]:
    """Normalizes point coordinates by dividing by the corresponding resolution values.

    Args:
        point (List[float]): A list containing x,y coordinates to normalize.
        resolution (List[int]): A list containing width,height values to normalize against.

    Returns:
        List[float]: Normalized coordinates as a list of floats.
    """
    return [p / r for p, r in zip(point, resolution)]
