tracing submodule:
    High:
    - break out write all --> divide extract-all cli functionality:
        extract-image 
            returns and/or outputs image result (land/plug mask)
        extract-vector --layer all|hole|plug|...
            hole = (even, water, black), 
            plug = (odd, land, white) 
            returns and/or outputs vector files (defualt shp)
        extract-raster --layer all|climate|terrain|...|political 
            returns and/or outputs raster (tif) files of tile classifications
    
    Medium:
    - pull ALL functionality defaults notable extract-image (process image) function defaults to a defaults module to set this will help develop a UI
    - In some cases level may not matter so even and odd can be treated as just shapes (boundaries) (e.g in the case of political borders (depth insignificant (in some cases))) In this case their isn't hole/plug just shape

    Low:
    -build functionality to read other colors as an option, default grayscale (maintain either white or black to study either 'holes' or 'plugs')
    - develop a UI to set and view inputs and set parameters to process a drawn image and output all/raster/vector/mask. Select parameters, input file, output file locaiton, view changes of the selected input image (render)
        - historically I'd think tkinter but surely we can do better nowadays.
        -serves as base for world building creation. New world/map -> select image -> tune parameters/set meanings to levels/hole-plug/metadata ---> easily let's overaly build atop clearly.

overlay submodule:
    - utilize tracing to get outlines and/or levels of additional drawing aligned to particular files (default plug/hole vector or classification raster) This is used for topology: contours → elevation levels, climate → zones, political → borders/regions, mountains → ridge masks / ranges, for example.
    



12/28:
Plan:
- break out 'vectorize_image_to_gdfs'
- 'image_to_binary' a wrapper on process_image-- doesn't seem necessary.
- create extract-image clean functionality and cli function wrapper

