from typing import TYPE_CHECKING
import json

import numpy as np

if TYPE_CHECKING:
    import napari.viewer

from auto_filet import PreviewCylinder, ZoomIn


def add_data(viewer: "napari.viewer.Viewer"):
    disk = np.zeros((100, 50))
    for xi in range(100):
        for yi in range(50):
            d = np.sqrt(((xi - 50) ** 2) + ((2 * yi - 50) ** 2))
            disk[xi, yi] = d
    disk = np.round(disk * 255 / disk.max()).astype(np.uint8)
    disks = [disk] * 40
    for i, disk in enumerate(disks):
        if i % 5 == 0:
            disks[i] = 200 * (disk > 0)
    cylindar = np.stack(disks)
    cylindar[:, :, 30] = 10
    viewer.add_image(cylindar.astype(np.uint8), scale=(1, 1, 2))
    viewer.add_points([[0, 50, 25], [40, 50, 25]], scale=(1, 1, 2))


def test_preview(make_napari_viewer):
    viewer = make_napari_viewer()
    add_data(viewer)
    pc = PreviewCylinder.create(viewer)
    assert np.array_equal([[0, 50, 50], [40, 50, 50]], pc.axis_points)
    assert pc.out_layer.data[31].mean() < 100
    assert pc.out_layer.data[87].mean() > 100
    assert pc.out_layer.data[151].mean() == 0
    viewer.add_points([0, 0, 75])
    pc.shift()
    assert pc.out_layer.data[29, 100, 36] == 10, "stripe has moved"
    viewer.add_points(([[27, 92, 14], [49, 282, 269]]))
    zm = ZoomIn.create(pc, r_resolution=10, slice_npixels=90_000)
    assert 100 > zm.out_layers[0].data.mean() > 80
    viewer.add_points([[2, 24, 34], [4, 53, 68]])
    z2 = ZoomIn.create(zm, r_resolution=10, slice_npixels=90_000)
    z2_dict = z2.to_dict()
    z3 = ZoomIn.from_dict(z2_dict, viewer)
    # pc2 = PreviewCylinder.from_dict(pc.to_dict(), viewer)
    assert json.dumps(z3.to_dict()) == json.dumps(z3.to_dict())
    assert z3.out_layers[0].data[6, 277, 80] == 10
    full = z3.get_full_resolution_dict()
    z4 = ZoomIn.from_dict(full, viewer, image_layers=[viewer.layers[0]])
    assert len(z4.out_layers) == 1
    assert z4.out_layers[0].data.sum() == 368208
    
