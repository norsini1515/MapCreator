tracing submodule:
    - break out write all --> divide extract-all cli functionality:
        extract-image #outputs image result (land/plug mask)
        extract-vector --layer all|hole(even, water, black), plug(odd, land, white) #outptus vector files (defualt shp)
        extract-raster --layer all|climate|terrain|...|political #outputs raster (tif) files of tile classifications
                -build functionality to read other colors (maintain either white or black to study either 'holes' or 'plugs')
    - pull ALL functionality defaults notable extract-image (process image) function defaults to a defaults module to set this will help develop a UI
    - develop a UI to set and view inputs and set parameters to process a drawn image and output all/raster/vector/mask. Select parameters, input file, output file locaiton, view changes of the selected input image (render)
        - historically I'd think tkinter but surely we can do better nowadays.
        -serves as base for world building creation. New world/map -> select image -> tune parameters/set meanings to levels/hole-plug/metadata ---> easily let's overaly build atop clearly.

overlay submodule:
    - utilize tracing to get outlines and/or levels of additional drawing aligned to particular files (default plug/hole vector or classification raster) This is used for topology: 0contours → elevation levels, climate → zones, political → borders/regions, mountains → ridge masks / ranges, for example.



