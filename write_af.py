from pathlib import Path
import napari_scripts as ns
import json
from napari.layers import Points, Image
import numpy as np
from auto_filet import AutoFilet, ZoomIn, View
from snapshot import Snapshot2D
import scipy.ndimage as ndi
from skimage.util import img_as_ubyte

path = Path("<path to file>")
i = 2

viewer = ns.get_viewer_from_file(path, i)
raw_layers = viewer.layers.copy()

def next_(arg):
    viewer.layers.select_next()
    viewer.layers.toggle_selected_visibility()
    viewer.layers.selection
    next_layer = next(iter(viewer.layers.selection))
    next_layer.mode = "add"
viewer.bind_key("x", next_, overwrite=True)

# select hrp channel
hrp = viewer.layers[0]
blurred = viewer.add_image(
    ndi.gaussian_filter(img_as_ubyte(hrp.data), sigma=(0, 1, 1)), scale=hrp.scale
)

# Define the central axis
af = AutoFilet.create(
    viewer,
    preview_channel=blurred,
    theta_resolution=700,
    height_resolution=500,
    radius_resolution=100,
)
# select shift point
af.shift()
viewer.add_shapes(ndim=2, name="reference", opacity=0.35)
for number in ("1", "2", "3"):
    for word in ("el", "bl", "nl", "vnc", "nr", "br", "er"):
        viewer.add_points(name=word + number, size=8, out_of_slice_display=True, ndim=3)

# then manualy add all the points
# then check that theyre all good
for number in ("1", "2", "3"):
    points_layers = [
        l for l in viewer.layers if isinstance(l, Points) and l.name.endswith(number)
    ]
    assert not any(len(l.data) == 0 for l in points_layers)

# make sure all of the points layers have points
af.save(Path("autofilet.hdf"), render_layers=raw_layers)
