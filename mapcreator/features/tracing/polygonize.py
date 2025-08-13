from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from typing import List, Tuple
import numpy as np

def contours_to_polygons(contours, *, min_area=5.0, min_points=3, verbose=False):
    polys = []
    for i, c in enumerate(contours):
        if len(c) < min_points: 
            continue
        coords = [(int(x), int(y)) for x, y in c[:,0,:]]
        poly = Polygon(coords)
        cand = []
        if not poly.is_valid or poly.area < min_area or len(poly.exterior.coords) < min_points:
            fixed = poly.buffer(0)
            cand = [fixed] if isinstance(fixed, Polygon) else list(getattr(fixed, "geoms", []))
        else:
            cand = [poly]
        for p in cand:
            ok = p.is_valid and p.area >= min_area and len(p.exterior.coords) >= min_points
            if ok:
                polys.append(p)
            elif verbose:
                print(f"skip {i}: {explain_validity(cand)}")
    return polys

# mapcreator/tracing/polygonize_tree.py

def _depths_and_children(hierarchy: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    """
    hierarchy: shape (1, N, 4) from cv2.findContours(..., RETR_TREE, ...)
    Returns:
      depth[i]: depth of contour i (0=root)
      children[i]: list of direct child indices of i
    """
    h = hierarchy[0]
    N = h.shape[0]
    parent = [h[i, 3] for i in range(N)]
    children = [[] for _ in range(N)]
    for i, p in enumerate(parent):
        if p != -1:
            children[p].append(i)
    depth = [0]*N
    # compute depths via parent chain
    for i in range(N):
        d, p = 0, parent[i]
        while p != -1:
            d += 1
            p = parent[p]
        depth[i] = d
    return depth, children

def _coords(cnt) -> List[tuple]:
    # cnt: (M,1,2)
    return [(int(x), int(y)) for x, y in cnt[:,0,:]]

def contours_to_polygons_with_holes(contours, hierarchy, *, min_area=5.0, min_points=3, verbose=False):
    """
    Build polygons honoring nested rings (even-depth = shell, odd-depth = hole).
    Returns list[shapely.Polygon].
    """
    depth, children = _depths_and_children(hierarchy)
    polys = []

    for i, cnt in enumerate(contours):
        if depth[i] % 2 != 0:  # only build from even-depth shells
            continue
        if len(cnt) < min_points:
            continue

        shell = _coords(cnt)
        holes = []
        for j in children[i]:
            if depth[j] % 2 == 1 and len(contours[j]) >= min_points:
                holes.append(_coords(contours[j]))

        poly = Polygon(shell, holes)
        if (not poly.is_valid) or (poly.area < min_area):
            fixed = poly.buffer(0)
            if fixed.is_empty:
                continue
            if fixed.geom_type == "Polygon":
                cands = [fixed]
            else:
                cands = [g for g in fixed.geoms if g.area >= min_area]
        else:
            cands = [poly]

        for g in cands:
            if g.is_valid and g.area >= min_area:
                polys.append(g)
            elif verbose:
                print("Skipped invalid piece:", explain_validity(g))

    return polys