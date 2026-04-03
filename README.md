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
   from auto_filet import PreviewCylinder, ZoomIn
   pc = PreviewCylinder.create(viewer)
   ```
1. To change where the auto-filet is split, create a new points layer and add a
   point at the x value where you would like the split. (the Z and Y
   coordinates are ignored)
1. run the following code to perform the split

    ``` pc.shift() ```

1. To create a high resolution auto-filet of a subset of an image, first create
   another points layer, and add two points defining the bounding box of the
   ZoomIn

    ``` Zoomin.create(pc) ``` 

1. Save the newly created layers
