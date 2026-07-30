from dataclasses import dataclass
from pathlib import Path

import h5py
from napari.layers import Points, Image
import napari.viewer
from scipy.ndimage import map_coordinates
import numpy as np
import tifffile


def save_image(layers: list[Image], path: Path, scale=True):
    data = np.stack([l.data for l in layers])
    with tifffile.TiffWriter(path, ome=True) as tif:
        metadata = {"axes": "CZYX", "Channel": {"Name": [l.name for l in layers]}}
        if scale:
            metadata = metadata | {
                "PhysicalSizeZ": layers[0].scale[0],
                "PhysicalSizeY": layers[0].scale[1],
                "PhysicalSizeX": layers[0].scale[2],
            }
        tif.write(
            data, metadata=metadata, compression="ZLIB", compressionargs={"level": 8}
        )


@dataclass(frozen=True, slots=True)
class CylinderFrame:
    """
    Encapulates the cartesian axes what include the cylindrical axis and 2
    orthoginal x_prime and y_prime
    """

    origin: np.ndarray  # (3,) world coords, ZYX
    axis: np.ndarray  # (3,) unit vector along cylinder axis
    x_prime: np.ndarray  # (3,) unit vector, radial reference direction
    y_prime: np.ndarray  # (3,) unit vector, completes right-handed frame

    @classmethod
    def create(
        cls, axis_points: tuple[tuple[float, float, float], tuple[float, float, float]]
    ):
        """Construct a cylinder coordinate frame from two points on the axis."""
        p0, p1 = np.array(axis_points)
        axis_vector = (p1 - p0) / np.linalg.norm(p1 - p0)
        # create a cubic coordinate system around axis_vector
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arb, axis_vector)) > 0.5:
            arb = np.array([0.0, 1.0, 0.0])
        x_prime = arb - np.dot(arb, axis_vector) * axis_vector
        x_prime /= np.linalg.norm(x_prime)
        y_prime = np.cross(axis_vector, x_prime)
        y_prime /= np.linalg.norm(y_prime)
        return cls(origin=p0, axis=axis_vector, x_prime=x_prime, y_prime=y_prime)


@dataclass(frozen=True, slots=True)
class ViewFrame:
    """
    Encapulates the cartesian orthoginal axes what include the depth being z, Y
    being the same as the cylinder axis
    """

    origin: np.ndarray  # (3,) world coords, ZYX
    z: np.ndarray  # (3,) depth axis (world)
    y: np.ndarray  # (3,) along cylinder axis (world)
    x: np.ndarray  # (3,) across cylinder (world)

    @classmethod
    def create(cls, cyl: CylinderFrame, view_angle: float):
        view_z = np.cos(view_angle) * cyl.x_prime + np.sin(view_angle) * cyl.y_prime
        view_x = np.cross(cyl.axis, view_z)
        return cls(origin=cyl.origin, z=view_z, y=cyl.axis, x=view_x)


def cylindrical_to_data(
    height: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    cyl: CylinderFrame,
    scale: np.ndarray,  # (3,) ZYX
) -> np.ndarray:  # (3, len(height), len(radius), len(theta))
    """Map cylindrical coords (height, radius, theta) to data voxel coords."""
    height_coords, radius_coords, theta_coords = np.meshgrid(
        height, radius, theta, indexing="ij"
    )
    world = (
        cyl.origin
        + height_coords[..., None] * cyl.axis
        + radius_coords[..., None] * np.cos(theta_coords)[..., None] * cyl.x_prime
        + radius_coords[..., None] * np.sin(theta_coords)[..., None] * cyl.y_prime
    )
    data = world / scale
    return np.moveaxis(data, -1, 0)


def cylindrical_to_view(
    height: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    cyl: CylinderFrame,
    view: ViewFrame,
) -> np.ndarray:  # (3, len(h), len(r), len(theta))
    """Map cylindrical coords to view coords (vz, vy, vx)."""
    height_coords, radius_coords, theta_coords = np.meshgrid(
        height, radius, theta, indexing="ij"
    )
    world = (
        cyl.origin
        + height_coords[..., None] * cyl.axis
        + radius_coords[..., None] * np.cos(theta_coords)[..., None] * cyl.x_prime
        + radius_coords[..., None] * np.sin(theta_coords)[..., None] * cyl.y_prime
    )
    d = world - view.origin
    vz = d @ view.z
    vy = d @ view.y
    vx = d @ view.x
    return np.stack([vz, vy, vx], axis=0)


def view_to_data(
    view_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    view: ViewFrame,
    voxel_size: np.ndarray,  # (3,) ZYX
) -> np.ndarray:  # (3, len(vz), len(vy), len(vx))
    """Map view coords (vz, vy, vx) to data voxel coords."""
    VZ, VY, VX = np.meshgrid(*view_coords, indexing="ij")
    world = (
        view.origin
        + VZ[..., None] * view.z
        + VY[..., None] * view.y
        + VX[..., None] * view.x
    )
    data = world / voxel_size
    return np.moveaxis(data, -1, 0)


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
class AutoFilet:
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
    cyl_frame: CylinderFrame
    out_layer: Image | None

    @classmethod
    def create(
        cls,
        viewer: "napari.viewer.Viewer",
        preview_channel: Image | None = None,
        axis_layer: Points | None = None,
        max_radius=150.0,
        radius_resolution=300,
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
        assert np.all(axis_layer.translate == (0, 0, 0))
        assert np.all(axis_layer.rotate == np.eye(3))
        radius = np.linspace(0, max_radius, radius_resolution)
        theta = np.linspace(0, np.pi * 2, theta_resolution)
        scale = axis_layer.scale
        p0, p1 = axis_layer.data * scale
        height = np.linspace(0, np.linalg.norm(p1 - p0), height_resolution)
        p0z, p0y, p0x = p0.tolist()
        p1z, p1y, p1x = p1.tolist()
        axis_points = ((p0z, p0y, p0x), (p1z, p1y, p1x))
        cyl_frame = CylinderFrame.create(axis_points)
        out_layer = None
        out = cls(
            viewer=viewer,
            theta=theta,
            height=height,
            radius=radius,
            source_layer=preview_channel,
            cyl_frame=cyl_frame,
            axis_points=axis_points,
            out_layer=out_layer,
        )
        out.render()
        return out

    def render(self):
        coords = cylindrical_to_data(
            self.height,
            self.radius,
            self.theta,
            self.cyl_frame,
            self.source_layer.scale,
        )
        out_data = map_coordinates(self.source_layer.data, coords, order=0, cval=0)
        out_data = out_data.swapaxes(1, 0)
        out_layer = self.viewer.add_image(
            out_data, name="preview", projection_mode="max"
        )
        assert isinstance(out_layer, Image)
        self.out_layer = out_layer

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
        creates a AutoFilet from a dictionary, perhaps created by
        AutoFilet.to_dict Additional arguments include the viewer, and
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
        cyl_frame = CylinderFrame.create(axis_points)
        out_layer = None
        out = cls(
            viewer=viewer,
            theta=theta,
            height=height,
            radius=radius,
            cyl_frame=cyl_frame,
            source_layer=source_layer,
            axis_points=axis_points,
            out_layer=out_layer,
        )
        if create_out_layer:
            out.render()
        return out

    def save(
        self, path: Path, compression_arg=1, render_layers: list[Image] | None = None
    ):
        """
        saves the viewer with the autofilet
        compression_arg mus be an intger 1 to 9. 9 meaning most compression
        if compression_arg is 0, no compression
        """
        if render_layers is None:
            render_layers = []

        with h5py.File(path, "w") as f:
            layers = f.create_group("layers")
            for layer in self.viewer.layers:
                ltype = layer.__class__.__name__
                if ltype == "Image" and compression_arg > 0:
                    dset = layers.create_dataset(
                        layer.name,
                        data=layer.data,
                        compression="gzip",
                        compression_opts=compression_arg,
                    )
                else:
                    dset = layers.create_dataset(layer.name, data=layer.data)
                dset.attrs["type"] = layer.__class__.__name__
                dset.attrs["scale"] = np.array(layer.scale)
            layer_names = f.create_dataset(
                "layer_names", shape=len(self.viewer.layers), dtype=h5py.string_dtype()
            )
            layer_names[:] = [l.name for l in self.viewer.layers]
            rl_dset = f.create_dataset(
                "render_layers", shape=len(render_layers), dtype=h5py.string_dtype()
            )
            rl_dset[:] = [l.name for l in render_layers]
            af_dset = f.create_group("auto_filet")
            af_dset.attrs["theta"] = np.array(
                [self.theta[0], self.theta[-1], self.theta.size]
            )
            af_dset.attrs["height"] = np.array(
                [self.height[0], self.height[-1], self.height.size]
            )
            af_dset.attrs["radius"] = np.array(
                [self.radius[0], self.radius[-1], self.radius.size]
            )
            af_dset.attrs["source_layer"] = self.source_layer.name
            if self.out_layer is None:
                raise ValueError("Run .render first")
            af_dset.attrs["out_layer"] = self.out_layer.name
            af_dset.attrs["axis_points"] = np.array(self.axis_points)

    @classmethod
    def load(cls, viewer: napari.Viewer, path: Path):
        with h5py.File(path, "r") as f:
            for key in f["layer_names"]:
                print(key in f["render_layers"])
                dset = f["layers"][key]
                if dset.attrs["type"] == "Image":
                    img = viewer.add_image(
                        np.array(dset), scale=dset.attrs["scale"], name=key.decode()
                    )
                    img.metadata["render_layers"] = key in f["render_layers"]
                elif dset.attrs["type"] == "Points":
                    viewer.add_points(
                        np.array(dset), scale=dset.attrs["scale"], name=key.decode()
                    )
                else:
                    raise NotImplementedError()
            af_hdf = f["auto_filet"]
            assert isinstance(af_hdf, h5py.Group)
            p0, p1 = af_hdf.attrs["axis_points"]  # type: ignore
            p0z, p0y, p0x = p0
            p1z, p1y, p1x = p1
            axis_points = ((p0z, p0y, p0x), (p1z, p1y, p1x))

            def args_to_tuple(args):
                return args[0], args[1], int(args[2])

            return cls(
                viewer=viewer,
                theta=np.linspace(*args_to_tuple(af_hdf.attrs["theta"])),
                height=np.linspace(*args_to_tuple(af_hdf.attrs["height"])),
                radius=np.linspace(*args_to_tuple(af_hdf.attrs["radius"])),
                source_layer=viewer.layers[af_hdf.attrs["source_layer"]],
                out_layer=viewer.layers[af_hdf.attrs["out_layer"]],
                cyl_frame=CylinderFrame.create(axis_points),
                axis_points=axis_points,
            )


@dataclass(frozen=True)
class View:
    view_coords: tuple[np.ndarray, np.ndarray, np.ndarray]
    preview: AutoFilet
    view_frame: ViewFrame
    out_layers: list[Image]
    mid_theta: float

    @classmethod
    def create(
        cls,
        preview: AutoFilet,
        image_layers: list[Image] | None = None,
        bbox_points: Points | None = None,
        scale=1,
    ) -> "View":
        if bbox_points is None:
            bbox_points = next(
                l for l in preview.viewer.layers[::-1] if isinstance(l, Points)
            )
        # get the range of cylindrical coords
        bbox_cyl = np.stack(
            [bbox_points.data_to_world(point) for point in bbox_points.data]
        )
        minr_i, minh_i, mintheta_i = np.floor(bbox_cyl.min(axis=0)).astype(int)
        maxr_i, maxh_i, maxtheta_i = np.ceil(bbox_cyl.max(axis=0)).astype(int)
        minr, maxr = preview.radius[[minr_i, maxr_i]]
        minh, maxh = preview.height[[minh_i, maxh_i]]
        mintheta, maxtheta = preview.theta[[mintheta_i, maxtheta_i]]
        # 20 seems fine since we will add 10 percent buffer at the end
        n_samples = 20
        radius = np.linspace(minr, maxr, n_samples)
        height = np.linspace(minh, maxh, n_samples)
        theta = np.linspace(mintheta, maxtheta, n_samples)
        # now covert to view
        mid_theta = (mintheta + maxtheta) / 2
        view_frame = ViewFrame.create(preview.cyl_frame, mid_theta)
        view_limits = cylindrical_to_view(
            height=height,
            radius=radius,
            theta=theta,
            cyl=preview.cyl_frame,
            view=view_frame,
        )
        mins = np.min(view_limits, axis=(1, 2, 3))
        maxs = np.max(view_limits, axis=(1, 2, 3))
        padding = 0.2 * (maxs - mins)
        lower_limit = mins - padding
        upper_limit = maxs + padding
        ranges = upper_limit - lower_limit
        nsamples = np.ceil(scale * ranges / min(preview.source_layer.scale)).astype(int)
        coord_arrays = tuple[np.ndarray, np.ndarray, np.ndarray](
            np.linspace(start, stop, num, endpoint=False)
            for start, stop, num in zip(lower_limit, upper_limit, nsamples)
        )
        out_coords = view_to_data(coord_arrays, view_frame, preview.source_layer.scale)
        # handle layers
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
        out_scale = tuple(c[1] - c[0] for c in coord_arrays)
        out_layers: list[Image] = []
        first = True
        for layer in image_layers:
            out = map_coordinates(layer.data, out_coords, order=3, cval=0)
            out_layers.append(
                preview.viewer.add_image(
                    out,
                    name=layer.name + "-view",
                    scale=out_scale,
                    colormap=layer.colormap,
                    blending="translucent" if first else "additive",
                    projection_mode=layer.projection_mode,
                    visible=False,
                )
            )
            first = False
        return cls(
            view_coords=coord_arrays,
            preview=preview,
            view_frame=view_frame,
            out_layers=out_layers,
            mid_theta=mid_theta,
        )

    def to_dict(self) -> dict:
        """
        Creates a json ready dictionary containing the object
        """
        return {
            "view_coords": [
                [coords[0].tolist(), coords[-1].tolist(), coords.size]
                for coords in self.view_coords
            ],
            "mid_theta": self.mid_theta,
            "preview": self.preview.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        viewer: napari.Viewer,
        image_layers: list[Image] | None = None,
        source_layer: Image | None = None,
    ):
        preview = AutoFilet.from_dict(
            data["preview"],
            viewer=viewer,
            source_layer=source_layer,
            create_out_layer=False,
        )
        preview.cyl_frame
        view_frame = ViewFrame.create(preview.cyl_frame, data["mid_theta"])
        coord_arrays = tuple(np.linspace(*specs) for specs in data["view_coords"])
        out_coords = view_to_data(coord_arrays, view_frame, preview.source_layer.scale)
        # handle layers
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
        view_frame = ViewFrame.create(preview.cyl_frame, data["mid_theta"])
        out_scale = tuple(c[1] - c[0] for c in coord_arrays)
        out_layers: list[Image] = []
        first = True
        for layer in image_layers:
            out = map_coordinates(layer.data, out_coords, order=3, cval=0)
            out_layers.append(
                preview.viewer.add_image(
                    out,
                    name=layer.name + "-view",
                    scale=out_scale,
                    colormap=layer.colormap,
                    blending="translucent" if first else "additive",
                    projection_mode=layer.projection_mode,
                )
            )
            first = False
            return cls(
                view_coords=coord_arrays,
                preview=preview,
                view_frame=view_frame,
                out_layers=out_layers,
                mid_theta=data["mid_theta"],
            )

    def save_image(self, path: Path):
        save_image(self.out_layers, path=path, scale=True)


@dataclass(frozen=True)
class ZoomIn:
    """
    Represents a full channel high resolution zoom of an autofilet

    can also be the source of further zoomins
    """

    theta: np.ndarray
    height: np.ndarray
    radius: np.ndarray
    preview: AutoFilet
    out_layers: list[Image]

    @classmethod
    def create(
        cls,
        source: AutoFilet,
        image_layers: list[Image] | None = None,
        bbox_points: Points | None = None,
        r_resolution=512,
        slice_npixels=500_000,
    ):
        """
        Zooms in on a region of an autofilet

        source can be an AutoFilet. This defines what layer
            bbox_points uses to define limits for height, theta, and radius
        bbox_points is a points layer a set of points that are gaurenteed to be
            included in the zoomin. Defaults to the last Points layer r_resolution
            is the number samples to take in the radius axis. Defaults to 512
        slice_npixels is the number of pixels in a 2D angle by height image.
            defaults to 500_000 these pixels are exactly square in micron space in
            the middle of the radius axis
        """
        if bbox_points is None:
            bbox_points = next(
                l for l in source.viewer.layers[::-1] if isinstance(l, Points)
            )
        bbox_world = bbox_points._transforms[1:].simplified(bbox_points.data)
        minr_i, minh_i, mintheta_i = bbox_world.min(axis=0).astype(int)
        maxr_i, maxh_i, maxtheta_i = bbox_world.max(axis=0).astype(int)
        minr, maxr = source.radius[[minr_i, maxr_i]]
        minh, maxh = source.height[[minh_i, maxh_i]]
        mintheta, maxtheta = source.theta[[mintheta_i, maxtheta_i]]
        mintheta = mintheta - 0.25  # ~ 15 degree buffer
        maxtheta = maxtheta + 0.25  # ~ 15 degree buffer
        maxr = maxr + 10
        minr = minr - 10
        # Calculate height and theta resolution
        mean_r = (maxr - minr) / 2
        height, theta = get_square_pixels(
            maxtheta, mintheta, maxh, minh, slice_npixels, mean_r
        )
        radius = np.linspace(minr, maxr, r_resolution)
        coordinates = cylindrical_to_data(
            height, radius, theta, source.cyl_frame, source.source_layer.scale
        )
        if image_layers is None:
            # take the layers before the preview layer
            image_layers = []
            for layer in source.viewer.layers:
                if isinstance(layer, Image):
                    if source.out_layer is not None and (
                        layer.name == source.out_layer.name
                    ):
                        break
                    image_layers.append(layer)
        out_layers: list[Image] = []
        for layer in image_layers:
            out = map_coordinates(layer.data, coordinates, order=3, cval=0)
            out = out.swapaxes(1, 0)
            # fix dtype
            out_layers.append(
                source.viewer.add_image(
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
            preview=source,
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

    def save_image(self, path: Path):
        save_image(self.out_layers, path=path, scale=False)


def compress_autofilet(path: Path):
    with h5py.File(path, "a") as f:
        layer_names = f["layer_names"]
        assert isinstance(layer_names, h5py.Dataset)
        for key in layer_names:
            layers = f["layers"]
            assert isinstance(layers, h5py.Group)
            dset: h5py.Dataset = layers[key] # type: ignore
            print(dset)
            print(dict(dset.attrs))

            if dset.attrs["type"] == "Image" and dset.compression_opts != 9:
                attrs = dset.attrs
                compressed_key = key + b"_compressed"
                assert compressed_key not in f["layers"]
                layers.create_dataset(
                    compressed_key, data=dset, compression="gzip", compression_opts=9
                )
                for akey, avalue in layers[key].attrs.items():
                    layers["compressed_key"].attrs[akey] = avalue
                del layers[key]
                layers.move(compressed_key, key)
                print(layers[key].nbytes / layers[key].get_storage_size())
