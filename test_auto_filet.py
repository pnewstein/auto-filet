from typing import TYPE_CHECKING
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import pytest
import numpy as np
import pytest

if TYPE_CHECKING:
    import napari.viewer

from auto_filet import (
    AutoFilet,
    CylinderFrame,
    ViewFrame,
    view_to_data,
    get_square_pixels,
    View,
    cylindrical_to_view,
    cylindrical_to_data,
)

import numpy as np
import pytest


# --- Fixtures ---


@pytest.fixture
def simple_cyl():
    """Cylinder with axis along world Y."""
    return CylinderFrame.create(((0, 0, 0), (0, 1, 0)))


@pytest.fixture
def diagonal_cyl():
    """Cylinder with a diagonal axis to test non-trivial frames."""
    return CylinderFrame.create(((10, 20, 30), (40, 50, 60)))


# --- CylinderFrame ---


def test_cylinder_frame_axis_is_unit(simple_cyl):
    assert np.isclose(np.linalg.norm(simple_cyl.axis), 1.0)


def test_cylinder_frame_orthonormal(diagonal_cyl):
    """axis, x_prime, y_prime must be mutually orthogonal unit vectors."""
    vecs = [diagonal_cyl.axis, diagonal_cyl.x_prime, diagonal_cyl.y_prime]
    for v in vecs:
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-10)
    assert np.isclose(np.dot(diagonal_cyl.axis, diagonal_cyl.x_prime), 0.0, atol=1e-10)
    assert np.isclose(np.dot(diagonal_cyl.axis, diagonal_cyl.y_prime), 0.0, atol=1e-10)
    assert np.isclose(
        np.dot(diagonal_cyl.x_prime, diagonal_cyl.y_prime), 0.0, atol=1e-10
    )


def test_cylinder_frame_origin(diagonal_cyl):
    assert np.allclose(diagonal_cyl.origin, [10, 20, 30])


def test_cylinder_frame_axis_aligned_with_world_z():
    """Axis parallel to first world axis should not break Gram-Schmidt."""
    cyl = CylinderFrame.create(((0, 0, 0), (1, 0, 0)))
    assert np.isclose(np.linalg.norm(cyl.x_prime), 1.0, atol=1e-10)
    assert np.isclose(np.dot(cyl.axis, cyl.x_prime), 0.0, atol=1e-10)


# --- ViewFrame ---


def test_view_frame_orthonormal(simple_cyl):
    view = ViewFrame.create(simple_cyl, view_angle=0.3)
    vecs = [view.z, view.y, view.x]
    for v in vecs:
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-10)
    assert np.isclose(np.dot(view.z, view.y), 0.0, atol=1e-10)
    assert np.isclose(np.dot(view.z, view.x), 0.0, atol=1e-10)
    assert np.isclose(np.dot(view.y, view.x), 0.0, atol=1e-10)


def test_view_frame_y_is_cylinder_axis(simple_cyl):
    for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
        view = ViewFrame.create(simple_cyl, angle)
        assert np.allclose(view.y, simple_cyl.axis)


def test_view_frame_z_is_radial(simple_cyl):
    """view.z should be parallel to the radial direction at view_angle."""
    for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
        view = ViewFrame.create(simple_cyl, angle)
        radial = np.cos(angle) * simple_cyl.x_prime + np.sin(angle) * simple_cyl.y_prime
        cosine_sim = np.dot(view.z, radial) / (
            np.linalg.norm(view.z) * np.linalg.norm(radial)
        )
        assert np.isclose(abs(cosine_sim), 1.0, atol=1e-10)


def test_view_frame_rotation_invariant_y(diagonal_cyl):
    """Rotating view_angle should never change the y axis."""
    angles = np.linspace(0, 2 * np.pi, 20)
    for angle in angles:
        view = ViewFrame.create(diagonal_cyl, angle)
        assert np.allclose(view.y, diagonal_cyl.axis, atol=1e-10)


# --- cylindrical_to_data ---


def test_cylindrical_to_data_shape(simple_cyl):
    h = np.linspace(0, 1, 5)
    r = np.linspace(0, 1, 4)
    t = np.linspace(0, 2 * np.pi, 6)
    scale = np.array([1.0, 1.0, 1.0])
    coords = cylindrical_to_data(h, r, t, simple_cyl, scale)
    assert coords.shape == (3, 5, 4, 6)


def test_cylindrical_to_data_origin(simple_cyl):
    """r=0, h=0, any theta should give the cylinder origin."""
    scale = np.array([1.0, 1.0, 1.0])
    coords = cylindrical_to_data(
        np.array([0.0]),
        np.array([0.0]),
        np.linspace(0, 2 * np.pi, 10),
        simple_cyl,
        scale,
    )
    for i in range(10):
        assert np.allclose(coords[:, 0, 0, i], simple_cyl.origin / scale, atol=1e-10)


def test_cylindrical_to_data_scale(simple_cyl):
    """Doubling scale should halve the data coords."""
    h = np.linspace(0, 1, 3)
    r = np.linspace(0, 1, 3)
    t = np.linspace(0, np.pi, 3)
    c1 = cylindrical_to_data(h, r, t, simple_cyl, np.array([1.0, 1.0, 1.0]))
    c2 = cylindrical_to_data(h, r, t, simple_cyl, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(c1, c2 * 2, atol=1e-10)


# --- cylindrical_to_view / view_to_data roundtrip ---


def test_cylindrical_view_data_roundtrip(diagonal_cyl):
    """
    Converting cylindrical -> view -> data should give the same result
    as cylindrical -> data directly.
    """
    scale = np.array([2.0, 1.5, 1.0])
    h = np.linspace(0, 10, 5)
    r = np.linspace(0, 3, 4)
    t = np.linspace(0, 2 * np.pi, 6)
    view = ViewFrame.create(diagonal_cyl, view_angle=0.5)

    # direct
    direct = cylindrical_to_data(h, r, t, diagonal_cyl, scale)

    # via view
    view_coords = cylindrical_to_view(h, r, t, diagonal_cyl, view)
    vz = view_coords[0].ravel()
    vy = view_coords[1].ravel()
    vx = view_coords[2].ravel()
    via_view = view_to_data((vz, vy, vx), view, scale)

    # shapes differ so compare a known point: h=0, r=0, t=0
    assert np.allclose(direct[:, 0, 0, 0], via_view[:, 0, 0, 0], atol=1e-10)


def test_view_to_data_origin_maps_correctly(diagonal_cyl):
    """vz=vy=vx=0 in view coords should map to cylinder origin in data."""
    scale = np.array([2.0, 1.5, 1.0])
    view = ViewFrame.create(diagonal_cyl, view_angle=0.5)
    coords = view_to_data(
        (np.array([0.0]), np.array([0.0]), np.array([0.0])), view, scale
    )
    assert np.allclose(coords[:, 0, 0, 0], diagonal_cyl.origin / scale, atol=1e-10)


def test_view_to_data_even_spacing():
    """
    A uniform linspace in view coords should map to evenly spaced,
    axis-aligned coords in a unit-scale volume.
    """
    cyl = CylinderFrame.create(((0, 0, 0), (0, 1, 0)))  # axis along world Y
    view = ViewFrame.create(cyl, view_angle=0.0)
    scale = np.array([1.0, 1.0, 1.0])

    vz = np.linspace(-5, 5, 11)
    vy = np.linspace(0, 10, 11)
    vx = np.linspace(-5, 5, 11)

    coords = view_to_data((vz, vy, vx), view, scale)  # (3, 11, 11, 11)

    # each axis of data coords should vary along exactly one output axis
    # and be uniform (evenly spaced)
    for i in range(3):
        plane = coords[i]
        # differences along each axis
        diffs = [np.diff(plane, axis=ax) for ax in range(3)]
        for ax, diff in enumerate(diffs):
            if diff.std() > 1e-10:
                # this axis varies — check it's uniform
                assert np.allclose(
                    diff, diff.flat[0], atol=1e-10
                ), f"coord[{i}] is not evenly spaced along output axis {ax}"


def test_view_to_data():
    """
    A point expressed in view coords should round-trip back to the
    same world position.
    """
    cyl = CylinderFrame.create(((10, 20, 30), (40, 50, 60)))
    view = ViewFrame.create(cyl, view_angle=0.3)
    scale = np.array([2.0, 1.5, 1.0])

    # single point at cylinder origin should map to origin / scale
    vz = np.array([0.0])
    vy = np.array([0.0])
    vx = np.array([0.0])

    coords = view_to_data((vz, vy, vx), view, scale)  # (3, 1, 1, 1)
    expected = cyl.origin / scale

    assert np.allclose(coords[:, 0, 0, 0], expected, atol=1e-10)


def test_view_to_data_spacing_matches_linspace():
    """
    Step size in data coords should match linspace step / scale
    when view basis vectors are aligned with world axes.
    """
    cyl = CylinderFrame.create(((0, 0, 0), (0, 1, 0)))
    view = ViewFrame.create(cyl, view_angle=0.0)
    scale = np.array([2.0, 3.0, 4.0])

    n = 20
    vz = np.linspace(0, 10, n)
    vy = np.linspace(0, 10, n)
    vx = np.linspace(0, 10, n)

    coords = view_to_data((vz, vy, vx), view, scale)

    # step along vz axis (axis 0 of output) for coord[0] (Z data coord)
    dz = np.diff(coords[0, :, 0, 0])
    expected_dz = (vz[1] - vz[0]) / scale[0]
    assert np.allclose(dz, expected_dz, atol=1e-10)


Z = ((0, 0, 0), (0, 0, 1))
X = ((0, 0, 0), (1, 0, 0))
OFFSET_Z = ((1, 2, 3), (1, 2, 4))


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


def test_90_deg(make_napari_viewer):
    viewer = make_napari_viewer()
    add_data(viewer)
    viewer.layers[0].data = viewer.layers[0].data.T
    viewer.layers[1].data = viewer.layers[1].data[:, ::-1]
    pc = AutoFilet.create(
        viewer, radius_resolution=150, height_resolution=250, theta_resolution=150
    )


def test_preview(make_napari_viewer):
    viewer = make_napari_viewer()
    add_data(viewer)
    pc = AutoFilet.create(
        viewer, radius_resolution=150, height_resolution=250, theta_resolution=150
    )
    assert np.array_equal([0, 50, 50], pc.cyl_frame.origin)
    assert pc.out_layer.data[15].mean() < 100
    assert pc.out_layer.data[43].mean() > 100
    assert pc.out_layer.data[76].mean() == 0
    viewer.add_points([0, 0, 37.5])
    pc.shift()
    assert pc.out_layer.data[14, 50, 18] == 10, "stripe has moved"
    # check file roundtrip
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "out.hd5"
        pc.save(viewer, path)
        v2 = make_napari_viewer()
        pc4 = AutoFilet.load(v2, path)
        assert pc4.to_dict() == pc.to_dict()
    # check round trip
    data = pc.to_dict()
    pc2 = pc.from_dict(data, viewer)
    assert np.mean(pc2.out_layer.data == pc.out_layer.data) > 0.95
    assert np.all(np.array(pc.axis_points) == np.array(pc2.axis_points))
    assert np.all(pc.theta == pc2.theta)
    viewer.add_points(([[14, 42, 7], [12, 141, 134]]))
    view = View.create(pc)
    view_data = view.out_layers[0].data
    assert view_data[8, 12, 20] == 10
    # check round trip
    data = view.to_dict()
    view2 = view.from_dict(json.loads(json.dumps(data)), viewer)
    assert np.all(view_data == view2.out_layers[0].data)
