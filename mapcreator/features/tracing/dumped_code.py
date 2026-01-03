def extract_vectors(
    image: Path,
    out_dir: Path,
    meta: dict,
    *,
    even_defs: dict = configs.LAND_DEFS,
    odd_defs: dict = configs.WATERBODY_DEFS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, str]]:
    """Run vector extraction only and write GeoDataFrames to disk.

    Returns ``(even_gdf, odd_gdf, merged_gdf, paths)`` where ``paths`` is a
    dict with ``{"even", "odd", "merged"}`` keys mapped to output file paths.
    """
    process_step(f"Starting vector extraction for {image.name}...")
    meta["width"], meta["height"] = detect_dimensions(image)

    even_gdf, odd_gdf, merged_gdf = vectorize_image_to_gdfs(
        image,
        meta,
        even_defs=even_defs,
        odd_defs=odd_defs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Default to land/waterbody filenames (base case) if not specified in defs
    even_path = export_gdf(
        even_gdf,
        out_dir / even_defs.get("file_name", configs.LAND_TRACING_EXTRACT_NAME),
    )
    odd_path = export_gdf(
        odd_gdf,
        out_dir / odd_defs.get("file_name", configs.WATER_TRACING_EXTRACT_NAME),
    )
    merged_path = export_gdf(
        merged_gdf,
        out_dir / configs.MERGED_TRACING_EXTRACT_NAME,
    )

    paths = {
        "even": str(even_path),
        "odd": str(odd_path),
        "merged": str(merged_path),
    }

    return even_gdf, odd_gdf, merged_gdf, paths

def vectorize_image_to_gdfs(
        image: Path, 
        meta: dict,
        even_defs: dict = configs.LAND_DEFS,
        odd_defs: dict = configs.WATERBODY_DEFS,
        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """High-level extraction driver.

    Steps:
      1. Preprocess -> binary land mask (1=land)
      2. Land polygons from binary
      3. Ocean + inland water polygons from binary
      4. Merge + dissolve
    Returns (land_gdf, waterbody_gdf, ocean_gdf, merged_gdf, bin_img)
    """
    process_step(f"Extracting even and odd features from {image.name}...")
    
    #indicates if verbose logging is on
    verbose = meta.get("verbose", False)
    
    #Step 0: Ensure we have image dimensions in metadata
    if 'width' not in meta or 'height' not in meta:
        w, h = detect_dimensions(image)
        meta["width"] = w
        meta["height"] = h
  

    # Step 1: Process image to centerline outline and filled land mask
    process_step("Step 1: Processing image to outline + filled land mask...")
    land_mask = extract_image(image, meta=meta, out_dir=meta.get("out_dir"))
    bin_img = land_mask
    
    # Step 2: Extract the even and odd contours from the binary image
    process_step("Step 2: Extracting even and odd contours from binary image...")
    even_polys, odd_polys = extract_polygons_from_binary(bin_img=bin_img, meta=meta, verbose=verbose)
    
    if not even_polys or not odd_polys:
        raise ValueError("No even (land) or odd (water) polygons extracted; check input image and preprocessing settings.")
    if verbose:
        info(f"Even (land-view) Polygons: {len(even_polys)}")
        info(f"Odd (water-view) Polygons: {len(odd_polys)}")

    # Step 3: Classify polygons into land / waterbody GDFs and transform
    process_step("Step 3: Building land and waterbody GeoDataFrames...")
    even_gdf = classify_and_transform(even_polys, even_defs, meta)
    odd_gdf = classify_and_transform(odd_polys, odd_defs, meta)
    
    if verbose:
        info(f"\nLand GDF CRS: {even_gdf.crs}, shape {even_gdf.shape}")
        info(f"Water GDF CRS: {odd_gdf.crs}, shape {odd_gdf.shape}")

    process_step("Building merged base...")
    merged_gdf = build_merged_base(even_gdf, odd_gdf)
    info(f"Merged GDF: {len(merged_gdf)} features, CRS: {merged_gdf.crs}, shape {merged_gdf.shape}\n")
    merged_gdf.plot(column="class", legend=True)
    if verbose == 'debug':
        plt.show()
    
    return even_gdf, odd_gdf, merged_gdf



def classify_polygons(polygons_with_depths, class_defs: dict, meta: dict) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame for a set of polygons with depth using provided class definitions.

    - polygons_with_depths: expected as [(geom, depth), ...], but will accept [geom, ...] and set depth=None
    - class_defs: base attributes to apply (e.g., {"class": "land"}); any provided 'depth' will be overridden per feature
    - meta: should contain 'crs'
    """
    process_step(f"Classifying {len(polygons_with_depths)} polygons with defs {class_defs}")

    columns = list(class_defs.keys()) + ["geometry"]
    if not polygons_with_depths:
        return gpd.GeoDataFrame(columns=columns, crs=meta.get("crs"))

    # Support both tuples and plain geometries
    if isinstance(polygons_with_depths[0], tuple) and len(polygons_with_depths[0]) == 2:
        geoms = [g for (g, _d) in polygons_with_depths]
        depths = [_d for (_g, _d) in polygons_with_depths]
    else:
        geoms = list(polygons_with_depths)
        depths = [None] * len(geoms)

    # Merge class defs and inject depth list
    class_metadata = dict(class_defs or {})
    class_metadata["depth"] = depths
    if "class" not in class_metadata:
        class_metadata["class"] = "unknown"

    gdf = to_gdf(geoms, class_metadata, crs=meta.get("crs"))
    # Ensure expected column order when possible
    existing = [c for c in columns if c in gdf.columns]
    gdf = gdf[existing + [c for c in gdf.columns if c not in existing]]
    return gdf

def build_merged_base(land_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parts = [df for df in [land_gdf, water_gdf] if not df.empty]
    if not parts:
        return gpd.GeoDataFrame(columns=["class", "geometry"], crs=land_gdf.crs)
    all_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=land_gdf.crs)
    return all_gdf #dissolve_class(all_gdf, class_col="class")


def build_and_transform_gdf(
    polygons_with_depths: list,
    meta: dict,
    parity: str = None, #default not provided (indifferent)
) -> gpd.GeoDataFrame:
    
    """Build merged GeoDataFrame from classified polygons and apply pixel->map affine."""

    land_polys = [pd for pd in polygons_with_depths if pd[0].geom_type != 'Polygon' or land_defs["class"] == 'land']
    water_polys = [pd for pd in polygons_with_depths if pd[0].geom_type != 'Polygon' or waterbody_defs["class"] == 'waterbody']

    land_gdf = classify_and_transform(land_polys, land_defs, meta)
    water_gdf = classify_and_transform(water_polys, waterbody_defs, meta)

    merged_gdf = build_merged_base(land_gdf, water_gdf)
    return merged_gdf