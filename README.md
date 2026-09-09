# Auto-Filet
<p align="center">
<img width="396" height="171" alt="Image" src="https://github.com/user-attachments/assets/3a515c3b-ff20-4eff-82f9-ee2b61dafbe9" />
</p>





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

   https://github.com/user-attachments/assets/a1066ef2-975f-4f54-8ddc-f09f01844775


```python
from auto_filet import AutoFilet, ZoomIn
af = AutoFilet.create(viewer)
```
1. To change where the auto-filet is split, create a new points layer and add a
   point at the x value where you would like the split. (the Z and Y
   coordinates are ignored)
1. run the following code to perform the split
   
https://github.com/user-attachments/assets/61a3bb7a-610b-413a-825c-68b1fc506fec


```python
af.shift()
```


1. To create a high resolution rendering of a portion of an image, make a new
   points layer with points that cover the extent the preview that you want to
   render at high resolution
   
   https://github.com/user-attachments/assets/41c4b967-9d84-401c-b49f-598da71580d5

```python
ZoomIn.create(af)
```
   
   https://github.com/user-attachments/assets/183b8653-248e-4232-b533-99692627d4c3


    

