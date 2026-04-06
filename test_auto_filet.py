from typing import TYPE_CHECKING
import json

import numpy as np
import pytest
from napari.layers import Image

if TYPE_CHECKING:
    import napari.viewer

from auto_filet import (
    PreviewCylinder,
    ZoomIn,
    cylindrical_to_map_coordinates,
    get_square_pixels,
)


def layer(scale=None, translate=None):
    kw = {
        k: v for k, v in dict(scale=scale, translate=translate).items() if v is not None
    }
    return Image(np.zeros((10, 10, 10)), **kw)


def run(theta, r, h, axis, **layer_kw):
    return cylindrical_to_map_coordinates(
        np.array(theta), np.array(r), np.array(h), axis, layer(**layer_kw)
    )


Z = ((0, 0, 0), (0, 0, 1))
X = ((0, 0, 0), (1, 0, 0))
OFFSET_Z = ((1, 2, 3), (1, 2, 4))


@pytest.mark.parametrize("nh,nr,nt", [(1, 1, 1), (5, 3, 8), (10, 4, 16)])
def test_shape(nh, nr, nt):
    assert run(
        np.linspace(0, 2 * np.pi, nt, endpoint=False), np.ones(nr), np.ones(nh), Z
    ).shape == (3, nh, nr, nt)


@pytest.mark.parametrize(
    "axis", [Z, X, OFFSET_Z, ((0, 0, 0), (1, 0, 1)), ((0, 0, 0), (0, 1, 0))]
)
def test_finite(axis):
    assert np.all(
        np.isfinite(
            run(np.linspace(0, 2 * np.pi, 8, endpoint=False), [1.0], [0.0, 1.0], axis)
        )
    )


@pytest.mark.parametrize(
    "theta,r,xyz",
    [
        (0, 1, [1, 0, 0]),
        (np.pi / 2, 1, [0, 1, 0]),
        (np.pi, 1, [-1, 0, 0]),
        (0, 2, [2, 0, 0]),
    ],
)
def test_known_xyz(theta, r, xyz):
    np.testing.assert_allclose(run([theta], [r], [0], Z)[:, 0, 0, 0], xyz, atol=1e-12)


@pytest.mark.parametrize("dh,axis", [(1.0, Z), (2.5, Z), (1.0, X), (1.0, OFFSET_Z)])
def test_height_along_axis(dh, axis):
    p0, p1 = np.array(axis[0]), np.array(axis[1])
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    r0 = run([0], [0.5], [0], axis)[:, 0, 0, 0]
    r1 = run([0], [0.5], [dh], axis)[:, 0, 0, 0]
    np.testing.assert_allclose(r1 - r0, dh * ax, atol=1e-12)


@pytest.mark.parametrize("s", [0.5, 2.0, 10.0])
def test_scale(s):
    base = run([0], [1], [0], Z)[:, 0, 0, 0]
    scaled = run([0], [1], [0], Z, scale=(s, s, s))[:, 0, 0, 0]
    np.testing.assert_allclose(scaled, base / s, atol=1e-12)


@pytest.mark.parametrize("t", [(1, 0, 0), (0, 5, -3)])
def test_translate(t):
    base = run([0], [1], [0], Z)[:, 0, 0, 0]
    translated = run([0], [1], [0], Z, translate=t)[:, 0, 0, 0]
    np.testing.assert_allclose(translated, base - np.array(t), atol=1e-12)


@pytest.mark.parametrize("sf", [0.5, 2.0, 100.0])
def test_axis_length_invariant(sf):
    r1 = run(
        np.linspace(0, 2 * np.pi, 6, endpoint=False),
        [0.5, 1],
        [0, 1],
        ((0, 0, 0), (0, 0, 1)),
    )
    r2 = run(
        np.linspace(0, 2 * np.pi, 6, endpoint=False),
        [0.5, 1],
        [0, 1],
        ((0, 0, 0), (0, 0, sf)),
    )
    np.testing.assert_allclose(r1, r2, atol=1e-12)


def test_theta_periodicity():
    t = np.array([0.3, 1.1, 2.5])
    r1 = run(t, [1], [0], Z)
    r2 = run(t + 2 * np.pi, [1], [0], Z)
    np.testing.assert_allclose(r1, r2, atol=1e-12)


@pytest.mark.parametrize("npix", [100, 1000, 4000])
def test_gsp_total_pixels_approx(npix):
    h, t = get_square_pixels(2.0, 0.0, 1.0, 0.0, npix, 1.0)
    assert abs(len(h) * len(t) - npix) / npix < 0.05


@pytest.mark.parametrize(
    "theta_range,h_range,mean_r",
    [
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (np.pi, 5.0, 2.0),
    ],
)
def test_gsp_square_aspect_ratio(theta_range, h_range, mean_r):
    h, t = get_square_pixels(theta_range, 0.0, h_range, 0.0, 1000, mean_r)
    assert (
        abs(len(t) / len(h) - theta_range * mean_r / h_range)
        / (theta_range * mean_r / h_range)
        < 0.05
    )


@pytest.mark.parametrize(
    "mintheta,maxtheta,minh,maxh",
    [
        (0.0, 2.0, 0.0, 1.0),
        (1.0, 3.0, -1.0, 1.0),
        (-np.pi, np.pi, 0.5, 2.5),
    ],
)
def test_gsp_linspace_bounds(mintheta, maxtheta, minh, maxh):
    h, t = get_square_pixels(maxtheta, mintheta, maxh, minh, 1000, 1.0)
    np.testing.assert_allclose(
        [h[0], h[-1], t[0], t[-1]], [minh, maxh, mintheta, maxtheta]
    )


def test_gsp_larger_radius_increases_theta_decreases_height():
    h1, t1 = get_square_pixels(2.0, 0.0, 1.0, 0.0, 1000, 1.0)
    h2, t2 = get_square_pixels(2.0, 0.0, 1.0, 0.0, 1000, 4.0)
    assert len(t2) > len(t1) and len(h2) < len(h1)


def test_gsp_swap_ranges_swaps_resolutions():
    h1, t1 = get_square_pixels(2.0, 0.0, 1.0, 0.0, 1000, 1.0)
    h2, t2 = get_square_pixels(1.0, 0.0, 2.0, 0.0, 1000, 1.0)
    assert len(h1) == len(t2) and len(t1) == len(h2)


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
