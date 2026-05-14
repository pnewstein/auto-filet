# Auto-filet

## Instalation

### Prerequisits

1. ensure you have
   [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
   installed

### Installing the package

    micromamba install -c conda-forge napari pyqt python=3.13 scipy git pip
    install git+https://www.github.com/pnewstein/auto-filet
    git+https://www.github.com/pnewstein/napari-czifile2

### Troubleshooting install

if the above fails, see [napari installation
troubleshooting](https://napari.org/stable/getting_started/installation.html).
Once napari is succesfuly installed, use the following command

    pip install git+https://www.github.com/pnewstein/auto-filet
    git+https://www.github.com/pnewstein/napari-czifile2

## Usage

1. Load a microscopy image of an embryo into napari
    1. You can drag and drop a .czi file and it will load
    1. .tif files created by imagej will require the following code:
       ```viewer.layers[0].data = viewer.layers[0].data.swapaxes(0, 1)``` then
       split stack to load properly

1. Create a new [points
   layer](https://napari.org/stable/howtos/layers/points.html) and add two
   points to define the central axis of the embryo
1. Run the following code to make a low range full resolution auto-filet
   preview 

   ```
   from auto_filet import AutoFilet, View
   af = AutoFilet.create(viewer)
   ```
1. To change where the auto-filet is split, create a new points layer and add a
   point at the x value where you would like the split. (the Z and Y
   coordinates are ignored)
1. run the following code to perform the split

    ``` af.shift() ```

1. run the following code to perform the split

    ``` af.shift() ```

1. To create a high resolution rendering of a portion of the image add two points defining the bounding box in autofilet coordinets

    ``` View.create(af) ``` 

### IO

1. Save the instructions for this view

    ```
    from pathlib import Path
    import json

    Path("Auto_filet.json").write_text(json.dumps(af.to_dict()))
    ```
1.  After loading raw image, load the autofilet

    ```AutoFilet.from_dict(json.loads(Path("Auto_filet.json").read_text()), viewer)```
    
### Rendering autofilets

1. if you rendered an autofilet with `af = AutoFilet.from_dict(..., create_out_layer=False)` you can render it later with af.render()
