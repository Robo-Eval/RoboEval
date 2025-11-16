"""Environment related utils."""

import numpy as np
from mojo.elements import Site


def get_random_sites(
    sites: np.ndarray[Site], amount: int, step: int = 1, segment: int = 0
) -> list[Site]:
    """Get random subset of Sites.

    Example:
        Get 2 random sites from a subset of basket sites with a step interval of 3.

        ```python
        sites = self.dishwasher.basket.site_sets[0]
        sites = get_random_sites(sites=sites, amount=2, step=3)
        ```

    Args:
        sites: Array of `Site` objects to choose from.
        amount: Number of sites to randomly select.
        step: Step size for slicing the sites array. Defaults to 1.
        segment: Segment of the sites array to work with:
                    - If positive, consider only the first `segment` sites.
                    - If negative, consider only the last `segment` sites.
                    - If 0, consider all sites. Defaults to 0.
    """
    sites = sites[::step]
    if segment:
        if segment > 0:
            sites = sites[:segment]
        else:
            sites = sites[segment:]
    return np.random.choice(sites, size=amount, replace=False).tolist()


def get_random_points_on_plane(
    amount: int,
    origin: np.ndarray,
    extents: np.ndarray,
    spacing: float,
    random_offset_bounds: np.ndarray = np.zeros(3),
) -> list[np.ndarray]:
    """Get a random set of points on a plane within 2D extents.

    Example:
        Get 2 random points within a 0.2 x 1.0 rectangular space.
        Points will be at least 0.05 apart from each other.
        Picked positions are randomized in a range of +-0.01 along all axes.

        ```python
        spawn_points = get_random_points_on_plane(
        amount=2, origin=[0, 0, 0,],
        extents=[0.1, 0.5], spacing=0.05,
        random_offset_bounds=[0.01, 0.01, 0.01])
        ```

    Args:
        amount: Number of points to randomly select.
        origin: Origin point of the plane (3D).
        extents: Extents of the plane in x and y directions.
        spacing: Spacing between points on the plane.
        random_offset_bounds: Randomization bounds for offset along each axis.
    """
    assert len(extents) >= 2
    x_count = int(extents[0] * 2 // spacing)
    y_count = int(extents[1] * 2 // spacing)
    assert x_count * y_count >= amount
    center = np.array([(x_count - 1) * spacing / 2, (y_count - 1) * spacing / 2, 0])
    ref_points = []
    for x in range(x_count):
        for y in range(y_count):
            ref_points.append((x, y))
    ids = np.random.choice(len(ref_points), size=amount, replace=False).tolist()
    ref_points = [ref_points[i] for i in ids]
    points = []
    for point in ref_points:
        random_offset = np.random.uniform(-random_offset_bounds, random_offset_bounds)
        points.append(
            np.array([point[0], point[1], 0]) * spacing
            + origin
            + random_offset
            - center
        )
    return points
