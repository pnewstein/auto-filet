from dataclasses import dataclass

from napari.layers import Points, Image
import napari.viewer
from scipy.ndimage import map_coordinates
import numpy as np


def cylindrical_to_map_coordinates(
    theta: np.ndarray,
    r: np.ndarray,
    h: np.ndarray,
    axis_points: tuple[tuple[float, float, float], tuple[float, float, float]],
    source_layer: Image,
):
    """
    returns a list of coordinates ready for map_coordinates
    """
    # define the axis
    p0, p1 = np.array(axis_points)
    axis_vector = (p1 - p0) / np.linalg.norm(p1 - p0)
    # create a cubix coordinate system around axis_vector
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arb, axis_vector)) > 0.5:
        arb = np.array([0.0, 1.0, 0.0])
    x_prime = arb - np.dot(arb, axis_vector) * axis_vector
    x_prime /= np.linalg.norm(x_prime)
    y_prime = np.cross(axis_vector, x_prime)
    y_prime /= np.linalg.norm(y_prime)
    # create cylindrical grid (height, radius, angle)
    H, R, T = np.meshgrid(h, r, theta, indexing="ij")
    # convert to Cartesian
    coords = (
        p0
        + H[..., None] * axis_vector
        + R[..., None] * np.cos(T)[..., None] * x_prime
        + R[..., None] * np.sin(T)[..., None] * y_prime
    )
    # convert from world to data
    # world_to_data is not vectorized, so we must get the affine ourselves
    coords = (
        source_layer._transforms[1:]
        .simplified.inverse(coords.reshape(-1, 3))
        .reshape(coords.shape)
    )
    # reorder to (3, H, R, T) for map_coordinates
    return np.moveaxis(coords, -1, 0)


def get_square_pixels(
    maxtheta: float,
    mintheta: float,
    maxh: float,
    minh: float,
    slice_npixels: int,
    mean_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta_range = maxtheta - mintheta
    # theta_range in radians * radius is arc length
    h_range = maxh - minh
    arc_range = theta_range * mean_r
    # arc_range / h_range = theta_resolution / y_length
    # x_lenth * y_length = slice_npixels
    # arc_range / h_range = slice_npixels / y_length ** 2
    # y_length = sqrt(slice_npixels * h_range / arc_range)
    # arc_range / h_range = theta_resolution ** 2 / slice_npixels
    # theta_resolution = sqrt(arc_range * slice_npixels / h_range)
    theta_resolution = np.sqrt(slice_npixels * arc_range / h_range).astype(int)
    height_resolution = np.sqrt(slice_npixels * h_range / arc_range).astype(int)
    height = np.linspace(minh, maxh, height_resolution)
    theta = np.linspace(mintheta, maxtheta, theta_resolution)
    return height, theta


@dataclass
class PreviewCylinder:
    """
    Represents the central axis and the coordinate system in an autofilet

    used for a lower resolution preview of the filet and can be used as an argument to ZoomIn.create
    """

    viewer: napari.viewer.Viewer
    theta: np.ndarray
    height: np.ndarray
    radius: np.ndarray
    source_layer: Image
    axis_points: tuple[tuple[float, float, float], tuple[float, float, float]]
    out_layer: Image | None

    @classmethod
    def create(
        cls,
        viewer: "napari.viewer.Viewer",
        preview_channel: Image | None = None,
        axis_layer: Points | None = None,
        max_radius=150.0,
        height_resolution=500,
        theta_resolution=300,
    ):
        """
        does a preview returning the preview cyclinder object

        viewer is the napari.Viewer to operate on
        preview channel is the Image layer to autofilet. Default is the first image
        axis_layer is the Points layer that defines the central axis. Asserts
            that there are two points. This also defines the height range being
            autofileted default is the first points layer
        max_radius is the furthest radius calcuated. resolution is always 300
        height_resolution is the number of points sampled between the two axis points
        theta_resolution is the number of points sampled between 0 and 2pi
        """
        if axis_layer is None:
            axis_layer = next(l for l in viewer.layers if isinstance(l, Points))
        if len(axis_layer.data) != 2:
            raise ValueError("wrong number of points")
        if preview_channel is None:
            preview_channel = next(l for l in viewer.layers if isinstance(l, Image))
        if axis_layer.ndim == 4 and preview_channel.data.shape[0] == 1:
            # do full load
            for layer in viewer.layers:
                layer.data = np.array(layer.data)
                assert layer.ndim == 4
                if isinstance(layer, Image):
                    assert layer.data.shape[0] == 1
                    layer.data = layer.data[0, ...]
                if isinstance(layer, Points):
                    assert np.all(layer.data[:, 0] == 0)
                    layer.data = layer.data[:, 1:]
        radius = np.linspace(0, max_radius, 300)
        theta = np.linspace(0, np.pi * 2, theta_resolution)
        p0, p1 = axis_layer._transforms[1:].simplified(axis_layer.data)
        height = np.linspace(0, np.linalg.norm(p1 - p0), height_resolution)
        p0z, p0y, p0x = p0.tolist()
        p1z, p1y, p1x = p1.tolist()
        axis_points = ((p0z, p0y, p0x), (p1z, p1y, p1x))
        coordinates = cylindrical_to_map_coordinates(
            theta, radius, height, axis_points, preview_channel
        )
        out = map_coordinates(preview_channel.data, coordinates, order=0, cval=0)
        out = out.swapaxes(1, 0)
        out_layer = viewer.add_image(out, name="preview")
        assert isinstance(out_layer, Image)
        return cls(
            viewer=viewer,
            theta=theta,
            height=height,
            radius=radius,
            source_layer=preview_channel,
            axis_points=axis_points,
            out_layer=out_layer,
        )

    def shift(self, break_points: Points | None = None):
        """
        Uses the the first point in break_points to change the angle at the edges of the x axis

        break_points defaults to the last points layer
        """
        assert self.out_layer is not None
        if break_points is None:
            break_points = next(
                l for l in self.viewer.layers[::-1] if isinstance(l, Points)
            )
        break_index = self.out_layer.world_to_data(
            break_points.data_to_world(break_points.data[0])
        )[2]
        theta_resolution = len(self.theta)
        shifted_indexor = (
            (np.arange(theta_resolution) + break_index) % theta_resolution
        ).astype(int)
        self.out_layer.data = self.out_layer.data[:, :, shifted_indexor]
        self.theta = self.theta + self.theta[int(break_index)]

    def to_dict(self) -> dict:
        """
        Creates a json ready dictionary containing the object
        """
        return {
            "theta": [self.theta[0].tolist(), self.theta[-1].tolist(), self.theta.size],
            "height": [
                self.height[0].tolist(),
                self.height[-1].tolist(),
                self.height.size,
            ],
            "radius": [
                self.radius[0].tolist(),
                self.radius[-1].tolist(),
                self.radius.size,
            ],
            "axis_points": self.axis_points,
        }

    @classmethod
    def from_dict(
        cls,
        src_dict: dict,
        viewer: napari.viewer.Viewer,
        source_layer: Image | None = None,
        create_out_layer=True,
    ):
        """
        creates a PreviewCylinder from a dictionary, perhaps created by
        PreviewCylinder.to_dict Additional arguments include the viewer, and
        source_layer. source_layer defaults to the first image layer in viewer
        """
        if source_layer is None:
            source_layer = next(l for l in viewer.layers if isinstance(l, Image))
        p0, p1 = src_dict["axis_points"]
        p0z, p0y, p0x = p0
        p1z, p1y, p1x = p1
        axis_points = ((p0z, p0y, p0x), (p1z, p1y, p1x))
        theta = np.linspace(*src_dict["theta"])
        radius = np.linspace(*src_dict["radius"])
        height = np.linspace(*src_dict["height"])
        if create_out_layer:
            coordinates = cylindrical_to_map_coordinates(
                theta, radius, height, axis_points, source_layer
            )
            out = map_coordinates(source_layer.data, coordinates, order=0, cval=0)
            out = out.swapaxes(1, 0)
            out_layer = viewer.add_image(out, name="preview")
            assert isinstance(out_layer, Image)
        else:
            out_layer = None
        return cls(
            viewer=viewer,
            theta=theta,
            height=height,
            radius=radius,
            source_layer=source_layer,
            axis_points=axis_points,
            out_layer=out_layer,
        )


@dataclass(frozen=True)
class ZoomIn:
    """
    Represents a full channel high resolution zoom of an autofilet

    can also be the source of further zoomins
    """

    theta: np.ndarray
    height: np.ndarray
    radius: np.ndarray
    preview: PreviewCylinder
    out_layers: list[Image]

    @classmethod
    def create(
        cls,
        source: "PreviewCylinder | ZoomIn",
        image_layers: list[Image] | None = None,
        bbox_points: Points | None = None,
        r_resolution=512,
        slice_npixels=500_000,
    ):
        """
        Zooms in on a region of an autofilet

        source can be a PreviewCylinder or zoomin. This defines what layer
            bbox_points uses to define limits for height, theta, and radius
        bbox_points is a points layer a set of points that are gaurenteed to be
            included in the zoomin. Defaults to the last Points layer r_resolution
            is the number samples to take in the radius axis. Defaults to 512
        slice_npixels is the number of pixels in a 2D angle by height image.
            defaults to 500_000 these pixels are exactly square in micron space in
            the middle of the radius axis
        """
        if isinstance(source, PreviewCylinder):
            preview = source
        else:
            preview = source.preview
            assert np.array_equal(source.out_layers[0].scale, [1, 1, 1])
        if bbox_points is None:
            bbox_points = next(
                l for l in preview.viewer.layers[::-1] if isinstance(l, Points)
            )
        bbox_world = bbox_points._transforms[1:].simplified(bbox_points.data)
        minr_i, minh_i, mintheta_i = bbox_world.min(axis=0).astype(int)
        maxr_i, maxh_i, maxtheta_i = bbox_world.max(axis=0).astype(int)
        minr, maxr = source.radius[[minr_i, maxr_i]]
        minh, maxh = source.height[[minh_i, maxh_i]]
        mintheta, maxtheta = source.theta[[mintheta_i, maxtheta_i]]
        # Calculate height and theta resolution
        mean_r = (maxr - minr) / 2
        height, theta = get_square_pixels(
            maxtheta, mintheta, maxh, minh, slice_npixels, mean_r
        )
        radius = np.linspace(minr, maxr, r_resolution)
        coordinates = cylindrical_to_map_coordinates(
            theta, radius, height, preview.axis_points, preview.source_layer
        )
        if image_layers is None:
            # take the layers before the preview layer
            image_layers = []
            for layer in preview.viewer.layers:
                if isinstance(layer, Image):
                    if preview.out_layer is not None and (
                        layer.name == preview.out_layer.name
                    ):
                        break
                    image_layers.append(layer)
        out_layers: list[Image] = []
        for layer in image_layers:
            out = map_coordinates(layer.data, coordinates, order=3, cval=0)
            out = out.swapaxes(1, 0)
            # fix dtype
            out_layers.append(
                preview.viewer.add_image(
                    out,
                    name=layer.name + "-zoomin",
                    colormap=layer.colormap,
                    blending=layer.blending,
                    projection_mode=layer.projection_mode,
                )
            )
        return cls(
            theta=theta,
            height=height,
            radius=radius,
            preview=preview,
            out_layers=out_layers,
        )

    def get_degrees_per_pixel(self) -> float:
        return 360 * (self.theta[-1] - self.theta[0]) / self.theta.size / 2 / np.pi

    def get_microns_per_pixel(self) -> float:
        return (self.height[-1] - self.height[0]) / self.height.size

    def get_max_scale(self) -> tuple[float, float, float]:
        """
        Returns the scale in microns in all three axes at their lowest resolution
        """
        return (
            (self.radius[-1] - self.radius[0]) / self.radius.size,
            self.get_microns_per_pixel(),
            self.radius[-1] / self.theta.size,
        )

    def to_dict(self) -> dict:
        """
        Creates a json ready dictionary containing the object
        """
        return {
            "theta": [self.theta[0].tolist(), self.theta[-1].tolist(), self.theta.size],
            "height": [
                self.height[0].tolist(),
                self.height[-1].tolist(),
                self.height.size,
            ],
            "radius": [
                self.radius[0].tolist(),
                self.radius[-1].tolist(),
                self.radius.size,
            ],
            "preview": self.preview.to_dict(),
        }

    def get_full_resolution_dict(self) -> dict:
        target_scale = self.preview.source_layer.scale.min()
        r_range = self.radius[-1] - self.radius[0]
        n_radius_samples = int(np.ceil(r_range / target_scale))
        radius = np.linspace(self.radius[0], self.radius[-1], n_radius_samples)
        current_scale = max(self.get_max_scale()[1:])
        sample_frac = target_scale / current_scale
        new_npixels = (self.height.size * self.theta.size) // (sample_frac**2)
        height, theta = get_square_pixels(
            self.theta[-1],
            self.theta[0],
            self.height[-1],
            self.height[0],
            new_npixels,
            (self.radius[-1] - self.radius[0]) / 2,
        )
        return self.__class__(
            theta, height, radius, self.preview, self.out_layers
        ).to_dict()

    @classmethod
    def from_dict(
        cls,
        src_dict: dict,
        viewer: napari.viewer.Viewer,
        source_layer: Image | None = None,
        image_layers: list[Image] | None = None,
    ):
        """
        creates a ZoomIn from a dictionary, perhaps created by ZoomIn.to_dict
        Additional arguments include the viewer, and source_layer for the
        PreviewCylinder. source_layer defaults to the first image layer in
        viewer. image_layers are the layers to be included in the zoomin
        """
        preview = PreviewCylinder.from_dict(
            src_dict["preview"], viewer, source_layer, create_out_layer=False
        )
        theta = np.linspace(*src_dict["theta"])
        radius = np.linspace(*src_dict["radius"])
        height = np.linspace(*src_dict["height"])
        if image_layers is None or len(image_layers) != 0:
            coordinates = cylindrical_to_map_coordinates(
                theta, radius, height, preview.axis_points, preview.source_layer
            )
        if image_layers is None:
            # take the layers before the preview layer
            image_layers = []
            for layer in preview.viewer.layers:
                if isinstance(layer, Image):
                    if (
                        preview.out_layer is not None
                        and (layer.name == preview.out_layer.name)
                    ) or layer.name.endswith("-zoomin"):
                        break
                    image_layers.append(layer)
        out_layers: list[Image] = []
        for layer in image_layers:
            out = map_coordinates(layer.data, coordinates, order=3, cval=0)
            out = out.swapaxes(1, 0)
            out_layers.append(
                preview.viewer.add_image(
                    out,
                    name=layer.name + "-zoomin",
                    colormap=layer.colormap,
                    blending=layer.blending,
                    projection_mode=layer.projection_mode,
                )
            )
        theta_range = theta[-1] - theta[0]
        h_range = height[-1] - height[0]
        return cls(
            theta=theta,
            height=height,
            radius=radius,
            preview=preview,
            out_layers=out_layers,
        )
