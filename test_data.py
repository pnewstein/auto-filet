from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

from napari.layers import Image
import napari_scripts as ns
import napari
import napari.viewer

def add_test_data(viewer: napari.viewer.Viewer):
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
    viewer.add_image(cylindar, scale=(1, 1, 2))



# viewer = napari.Viewer()
# viewer.add_image(ar3d)
# viewer.add_image(ar23d, scale=(1, 1, 2))
viewer = ns.get_viewer_from_file(
    Path("uas-bh1-nkx6-555_ha_647_hrp_whole_embryo.czi"), 0
)

p0, p1 = (
    viewer.layers["Points"]._transforms[1:].simplified(viewer.layers["Points"].data)
)
axis_vector = (p1 - p0) / np.linalg.norm(p1 - p0)

# create the other coordinates
arb = np.array([1.0, 0.0, 0.0])
if abs(np.dot(arb, axis_vector)) > 0.5:
    arb = np.array([0.0, 1.0, 0.0])


x_prime = arb - np.dot(arb, axis_vector) * axis_vector
x_prime /= np.linalg.norm(x_prime)
y_prime = np.cross(axis_vector, x_prime)
y_prime /= np.linalg.norm(y_prime)


def cartesian(theta: float, r: float, h: float):
    return (
        p0 + h * axis_vector + r * np.cos(theta) * x_prime + r * np.sin(theta) * y_prime
    )


def cartesian(theta, r, h):
    theta, r, h = np.broadcast_arrays(theta, r, h)
    return (
        p0
        + h[..., None] * axis_vector
        + r[..., None] * np.cos(theta)[..., None] * x_prime
        + r[..., None] * np.sin(theta)[..., None] * y_prime
    )


def cylindrical_to_map_coordinates(theta, r, h):
    theta = np.asarray(theta)
    r = np.asarray(r)
    h = np.asarray(h)

    # create cylindrical grid (height, radius, angle)
    H, R, T = np.meshgrid(h, r, theta, indexing="ij")

    # convert to Cartesian
    coords = (
        p0
        + H[..., None] * axis_vector
        + R[..., None] * np.cos(T)[..., None] * x_prime
        + R[..., None] * np.sin(T)[..., None] * y_prime
    )
    coords.reshape(-1, 3)
    # convert to data
    coords = (
        viewer.layers["Points"]
        ._transforms[1:]
        .simplified.inverse(coords.reshape(-1, 3))
        .reshape(coords.shape)
    )

    # reorder to (3, H, R, T) for map_coordinates
    return np.moveaxis(coords, -1, 0)


for layer in viewer.layers[:2]:
    if not isinstance(layer, Image):
        continue
    # data = layer.data.copy().clip(*layer.contrast_limits)
    # data *= 255 / data.max()
    # data = data.astype(np.uint8)
    data = layer.data
    theta = np.linspace(0, 2 * np.pi, 512)
    h = np.linspace(0, np.linalg.norm(p1 - p0), 512)
    h = np.linspace(100, 150, 512)
    coordinates = cylindrical_to_map_coordinates(theta, np.linspace(32, 85, 500), h)
    out = map_coordinates(data, coordinates, order=3, cval=0)
    out = out.swapaxes(1, 0)
    viewer.add_image(
        out,
        name=layer.name,
        colormap=layer.colormap,
        blending=layer.blending,
        projection_mode=layer.projection_mode,
    )
