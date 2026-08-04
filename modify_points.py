from pathlib import Path

from auto_filet import AutoFilet
from napari.viewer import Viewer
import numpy as np
import h5py


def load_lite(path: Path, viewer: Viewer) -> list[str]:
    """
    returns the names of points channels that are modifiable
    """
    layer_names: list[str] = []
    with h5py.File(path, "r") as f:
        for key in f["layer_names"]:
            try:
                dset = f["layers"][key]
            except KeyError:
                continue
            if dset.attrs["type"] == "Points":
                layer_names.append(key.decode())
                viewer.add_points(
                    np.array(dset), scale=dset.attrs["scale"], name=key.decode()
                )
        af_hdf = f["auto_filet"]
        out_layer = af_hdf.attrs["out_layer"]
        image_dset = f["layers"][out_layer.encode()]
        img = viewer.add_image(
            np.array(image_dset), scale=image_dset.attrs["scale"], name=out_layer
        )
    return layer_names


def write_lite(path: Path, layer_names: list[str], viewer: Viewer):
    """
    writes the new data to hdf file
    """
    layer_dict = {n: viewer.layers[n].data for n in layer_names}
    with h5py.File(path, "a") as f:
        for n, data in layer_dict.items():
            f["layers"][n.encode()][...] = data
