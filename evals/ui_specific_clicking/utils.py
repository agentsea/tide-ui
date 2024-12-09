from typing import List


def calculate_normalized_distance(
    pred: List[int], target: List[int], resolution: List[int]
) -> float:
    """
    Calculate the distance between a prediction and target point, normalized by the diagonal of the image.
    """
    pred_x, pred_y = pred
    target_x, target_y = target
    res_x, res_y = resolution

    return (
        ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
        / ((res_x**2) + (res_y**2))
    ) ** 0.5