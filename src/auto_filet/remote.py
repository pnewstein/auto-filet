from pathlib import Path
import json
from itertools import count
from napari.layers import Points, Image
import numpy as np
from auto_filet import AutoFilet, ZoomIn, View
# from snapshot import Snapshot2D
from aicspylibczi import CziFile
import napari
from scipy import ndimage as ndi
import h5py
import os

path = Path(
    "/mnt/z/lab_member_data/Peter Newstein/elena/autofilet_fd4_best/proc/fd4-lacz_full4/autofilet.hdf"
)
def postproc(path: Path):
    add_render_layers(path)
    os.chdir(path.parent)
    Path("out").mkdir(exist_ok=False)
    viewer = napari.Viewer()
    af = AutoFilet.load(viewer, path)
    # blurring images before subsabmping is wise
    render_layers: list[Image] = [
        l for l in viewer.layers if isinstance(l, Image) and l.metadata["render_layers"]
    ]
    blurred_layers = []
    for layer in render_layers:
        blurred = ndi.gaussian_filter(layer.data, sigma=(0, 1, 1))
        blurred_layers.append(Image(blurred, scale=layer.scale, name=layer.name))

    for number_int in range(1, 10):
        number = str(number_int)
        points_layers = [l for l in viewer.layers if isinstance(l, Points) and l.name.endswith(number)]
        if len(points_layers) == 0:
            break
        assert not any(len(l.data) == 0 for l in points_layers)
        all_points = np.concatenate([l.data for l in points_layers])
        #creates calculation
        for point_layer in points_layers:
            print(point_layer)
            view = View.create(af, image_layers=render_layers, bbox_points=point_layer)
            (Path("out") / point_layer.name).with_suffix(".json").write_text(json.dumps(view.to_dict()))
            view.save_image((Path("out") / point_layer.name).with_suffix(".ome.tiff"))
        zm = ZoomIn.create(af, blurred_layers, Points(all_points, scale=points_layers[0].scale), slice_npixels=500_000, r_resolution=100)
        zm.save_image(Path(f"out/seg{number}.ome.tiff"))
        Path(f"out/seg{number}.json").write_text(json.dumps({"microns": zm.get_microns_per_pixel(), "degrees": zm.get_degrees_per_pixel()}))


def add_render_layers(path: Path):
    with h5py.File(str(path), "a") as f:
        render_layers = [
            l
            for l in f["layers"]
            if l.startswith("raw-")
            and l.endswith("-channel")
        ]
        print(render_layers)
        if "render_layers" in f:
            del f["render_layers"]
        rl_dset = f.create_dataset(
            "render_layers", len(render_layers), dtype=h5py.string_dtype()
        )
        rl_dset[:] = render_layers
