from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity

def contours_to_polygons(contours, *, min_area=5.0, min_points=3, verbose=False):
    polys = []
    for i, c in enumerate(contours):
        if len(c) < min_points: 
            continue
        coords = [(int(x), int(y)) for x, y in c[:,0,:]]
        poly = Polygon(coords)
        if not poly.is_valid or poly.area < min_area or len(poly.exterior.coords) < min_points:
            fixed = poly.buffer(0)
            candidates = [fixed] if isinstance(fixed, Polygon) else list(getattr(fixed, "geoms", []))
        else:
            candidates = [poly]
        for cand in candidates:
            ok = cand.is_valid and cand.area >= min_area and len(cand.exterior.coords) >= min_points
            if ok:
                polys.append(cand)
            elif verbose:
                print(f"skip {i}: {explain_validity(cand)}")
    return polys