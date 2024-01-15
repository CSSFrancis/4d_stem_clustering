from diffsims.generators.rotation_list_generators import get_beam_directions_grid
from scipy.spatial import ConvexHull
from matplotlib.collections import PolyCollection
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.simulation_generator import SimulationGenerator
from skimage.draw import disk

from orix.quaternion import Rotation
from orix.sampling import sample_S2


class CrystalSTEMSimulation:
    def __init__(
        self,
        phase,
        num_crystals=10,
        k_range=1,
        crystal_size=10,
        real_space_pixels=128,
        recip_space_pixels=64,
        generator=None,
        max_excitation_error=0.01,
        rotations=None,
        **kwargs,
    ):
        # Set all of the parameters
        if generator is None:
            self.generator = SimulationGenerator(200, minimum_intensity=1e-10)
        else:
            self.generator = generator
        self.num_crystals = num_crystals
        self.k_range = k_range
        self.crystal_size = crystal_size
        self.real_space_pixels = real_space_pixels
        self.recip_space_pixels = recip_space_pixels
        self.phase = phase

        # Set up the Simulations for calculating the Diffraction Data
        if rotations is None:
            beam_dir = get_beam_directions_grid("cubic", resolution=1)
            r = Rotation.from_euler(beam_dir, degrees=True, direction="crystal2lab")
        else:
            r = rotations

        self.simulation = self.generator.calculate_ed_data(
            phase=phase,
            reciprocal_radius=1,
            rotation=r,
            max_excitation_error=max_excitation_error,
        )

        # Randomly set up the center and rotations for the crystals
        self.centers = np.random.randint(
            low=crystal_size + 5,
            high=real_space_pixels - (crystal_size + 5),
            size=(num_crystals, 2),
        )
        self.random_rotations = np.random.randint(0, len(r.data), num_crystals)
        self.random_inplane = np.random.random(num_crystals) * np.pi

        # Get the Nano crystal Vectors

        (
            self.vectors,
            self.coordinates,
            self.real_space_vectors,
            self.intensities,
        ) = self.get_nano_crystal_vectors(max_excitation_error=max_excitation_error)

    def get_nano_crystal_vectors(self, max_excitation_error=0.015):
        """
        Get the nano crystal vectors
        """
        real_space_pixels = self.real_space_pixels
        recip_space_pixels = self.recip_space_pixels
        # Build the 4-D STEM dataset
        dataset = np.zeros(
            (
                real_space_pixels,
                real_space_pixels,
                recip_space_pixels,
                recip_space_pixels,
            )
        )

        coordinates = []
        real_space_pos = []
        vectors = []
        intens = []
        # For each crystal randomly rotate a
        for rot, center, inplane in zip(
            self.random_rotations, self.centers, self.random_inplane
        ):
            # Getting the simulation
            cor = self.simulation.irot[rot].coordinates.data[:, :2]
            inten = self.simulation.irot[rot].coordinates.intensity

            vector_pos = np.argwhere(
                create_blob(
                    center=center,
                    size=self.crystal_size,
                    real_space_pixels=real_space_pixels,
                )
            )
            real_space_pos.append(vector_pos)

            # Randomly rotating in plane
            corx = cor[:, 0] * np.cos(inplane) - cor[:, 1] * np.sin(inplane)
            cory = cor[:, 0] * np.sin(inplane) + cor[:, 1] * np.cos(inplane)
            cors = np.stack([corx, cory, inten], axis=1)
            coordinates.append(cors)
            intens.append(inten)
            # Create a list of 4-D Vectors describing the extent of the
            v = [list(v) + list(c) for v in vector_pos for c in cors]
            vectors.append(v)
            intens.append(intens)
        # Unpack the vectors for plotting
        vectors = np.array([v for vector in vectors for v in vector])
        return vectors, coordinates, real_space_pos, intens

    def get_coords(self, low=0.0, high=0.9):
        filtered_coords = []
        for co in self.coordinates:
            norms = np.linalg.norm(co[:, :2], axis=1)
            within_range = (norms > low) * (norms < high)
            filtered_coords.append(co[within_range])
        return filtered_coords

    def make_4d_nano(
        self,
        num_electrons=1,
        electron_gain=1,
        noise_level=0.1,
        radius=5,
    ):
        """
        Make a 4-D Nanocrystal strucuture.
        """
        real_space_pixels = self.real_space_pixels
        recip_space_pixels = self.recip_space_pixels
        k_range = self.k_range

        vectors_by_index = convert_to_markers(
            self.vectors, real_space=self.real_space_pixels
        )

        arr = np.zeros(
            (
                real_space_pixels,
                real_space_pixels,
                recip_space_pixels,
                recip_space_pixels,
            )
        )

        scale = (recip_space_pixels / 2) / k_range
        center = recip_space_pixels / 2
        for i in np.ndindex((real_space_pixels, real_space_pixels)):
            vector = vectors_by_index[i]
            im = np.zeros((self.recip_space_pixels, self.recip_space_pixels))
            for v in vector:
                im = add_disk(
                    im, center=v[:2] * scale + center, radius=radius, intensity=v[2]
                )
            arr[i] = im

        arr = arr * num_electrons
        arr = np.random.poisson(arr) * electron_gain

        noise = np.random.random(arr.shape) * noise_level
        arr = arr + noise
        return arr

    def plot_real_space(
        self,
        ax=None,
        remove_below_n=None,
        remove_non_symmetric=False,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        colors = [
            "blue",
            "green",
            "red",
            "yellow",
            "purple",
            "orange",
            "violet",
            "indigo",
            "black",
            "grey",
        ]
        verts = []
        if remove_below_n is not None:
            vectors = [
                v
                for v, c in zip(self.real_space_vectors, self.coordinates)
                if len(c) > remove_below_n
            ]
        else:
            vectors = self.real_space_vectors

        if remove_non_symmetric:
            new_vectors = []
            for c, v in zip(self.coordinates, self.real_space_vectors):
                un, counts = np.unique(
                    np.round(np.linalg.norm(c[:, :2], axis=1), 2), return_counts=True
                )
                if np.any(counts > 2):
                    new_vectors.append(v)
            vectors = new_vectors

        for v in vectors:
            hull = ConvexHull(v)
            vert = hull.points[hull.vertices]
            verts.append(vert)
        p = PolyCollection(verts, color=colors, **kwargs)
        ax.add_collection(p)
        ax.set_xlim(0, self.real_space_pixels)
        ax.set_ylim(0, self.real_space_pixels)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_example_dp(
        self,
        rotation=None,
        num_electrons=1,
        electron_gain=1,
        noise_level=0.1,
        pixels=64,
        reciprocal_radius=1,
        ax=None,
        threshold=0.8,
        disk_r=5,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        sim = self.generator.calculate_ed_data(
            self.phase,
            rotation=rotation,
            reciprocal_radius=reciprocal_radius,
            max_excitation_error=0.01,
            shape_factor_width=1,
        )

        img = np.zeros((pixels, pixels))
        coors = sim.coordinates[:, :2]  # x,y
        intens = sim.intensities
        scale = (pixels / 2) / reciprocal_radius
        center = pixels / 2

        for c, inten in zip(coors, intens):
            img = add_disk(
                img, center=c * scale + center, radius=disk_r, intensity=inten
            )
        img = img * num_electrons

        img = np.random.poisson(img) * electron_gain
        img = img + np.random.random(img.shape) * noise_level
        if threshold is not None:
            kwargs["vmax"] = np.max(img) * threshold
        ax.imshow(
            img,
            extent=(
                -reciprocal_radius,
                reciprocal_radius,
                -reciprocal_radius,
                reciprocal_radius,
            ),
            **kwargs,
        )
        return ax


#################################
# Extra Methods
################################

from random import random
from math import cos, sin, pi, radians
from matplotlib import Path
from skimage.segmentation import watershed


def create_blob(center, size, real_space_pixels):
    N = 4
    amps = [random() * (1 / (2 * N)) for _ in range(N)]
    phases = [random() * 2 * pi for _ in range(N)]

    points = np.empty((360, 2))
    img = np.zeros((real_space_pixels, real_space_pixels))
    for deg in range(360):
        alpha = radians(deg)
        radius = 1 + sum([amps[i] * cos((i + 1) * alpha + phases[i]) for i in range(N)])
        points[deg, 0] = (cos(alpha) * radius * size) + center[0]
        points[deg, 1] = (sin(alpha) * radius * size) + center[1]
    points = np.array(
        np.round(
            points,
        ),
        dtype=int,
    )
    img[points[:, 0], points[:, 1]] = 1
    img = watershed(img) == 2
    return img


import numpy as np
from copy import deepcopy


def convert_to_markers(peaks, real_space, **kwargs):
    new_peaks = deepcopy(peaks)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    by_ind_peaks = np.empty((real_space, real_space), dtype=object)
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, real_space), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, real_space + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, real_space), side="left")
        high_y_ind = np.searchsorted(
            x_inds[:, 1], range(1, real_space + 1), side="left"
        )
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            by_ind_peaks[i, j] = x_inds[lo_y:hi_y, 2:]
    return by_ind_peaks


def add_disk(image, center, radius, intensity):
    disk_image = np.zeros_like(image)
    rr, cc = disk(center=center, radius=radius, shape=image.shape)
    disk_image[rr, cc] = intensity  # expected 1 electron per pixel
    image = disk_image + image
    return image


import numpy as np
from copy import deepcopy


def convert_flat_to_markers(peaks, signal, **kwargs):
    new_peaks = deepcopy(peaks)
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_peaks[:, 0] = np.round((new_peaks[:, 0] - x_axis.offset) / x_axis.scale)
    new_peaks[:, 1] = np.round((new_peaks[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    x, y = signal.axes_manager.signal_axes
    shape = signal.axes_manager.navigation_shape
    by_ind_peaks = np.empty(shape, dtype=object)
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[0]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[0] + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[1]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[1] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            x_values = x_inds[lo_y:hi_y, 2]
            y_values = x_inds[lo_y:hi_y, 3]
            # print(lo_x,hi_x, lo_y, hi_y)
            by_ind_peaks[i, j] = np.stack((y_values, x_values), axis=1)
    return by_ind_peaks


def unwrap(pks):
    r = np.linalg.norm(pks[:, :2], axis=1)  # ignore the intensity
    theta = np.arctan2(pks[:, 0], pks[:, 1])
    return np.stack((r, theta, pks[:, 2]), axis=1)


def get_angles(angles):
    all_angles = np.abs(np.triu(np.subtract.outer(angles, angles)))
    all_angles = all_angles[all_angles != 0]
    all_angles[all_angles > np.pi] = np.pi - np.abs(
        all_angles[all_angles > np.pi] - np.pi
    )
    return all_angles


import numpy as np
from copy import deepcopy


def convert_to_labeled_markers(
    peaks, signal, polar=True, return_marker=False, colors=None, **kwargs
):
    new_peaks = deepcopy(peaks.data)
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_peaks[:, 0] = np.round((new_peaks[:, 0] - x_axis.offset) / x_axis.scale)
    new_peaks[:, 1] = np.round((new_peaks[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    x, y = signal.axes_manager.signal_axes
    shape = signal.axes_manager.navigation_shape
    by_ind_peaks = np.empty(shape, dtype=object)
    by_ind_colors = np.empty(shape, dtype=object)
    num_labels = np.max(new_peaks[:, -1])
    if colors is None:
        colors_by_index = (
            np.random.random((int(num_labels + 1), 3)) * 0.9
        )  # (Stay away from white)
    else:
        colors_by_index = colors
    colors_by_index = np.vstack((colors_by_index, [1, 1, 1]))
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[0]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[0] + 1), side="left"
    )
    # print(low_x_ind)
    # print(high_x_ind)
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[1]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[1] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            x_values = x_inds[lo_y:hi_y, 2]
            y_values = x_inds[lo_y:hi_y, 3]
            # print(lo_x,hi_x, lo_y, hi_y)
            labels = np.array(x_inds[lo_y:hi_y, -1], dtype=int)
            by_ind_peaks[i, j] = np.stack((y_values, x_values), axis=1)
            by_ind_colors[i, j] = colors_by_index[labels]
    return by_ind_peaks, by_ind_colors, colors_by_index


from scipy.spatial import ConvexHull, convex_hull_plot_2d


def points_to_poly_collection(points, hull_index=(0, 1)):
    try:
        hull = ConvexHull(points[:, hull_index][:, ::-1])
    except:
        return np.array([[0, 0], [0, 0], [0, 0]])
    return hull.points[hull.vertices]


from sklearn.cluster import OPTICS
from pyxem.utils.labeled_vector_utils import column_mean


def cluster_labeled_vectors(self, eps=2, min_samples=2):
    mean_pos = self.map_vectors(
        column_mean, columns=[0, 1], label_index=-1, dtype=float, shape=(2,)
    )
    vectors = self.data
    clustering = OPTICS(min_samples=min_samples, max_eps=eps).fit(mean_pos)
    labels = clustering.labels_
    initial_labels = self.data[:, -1].astype(int)
    new_labels = labels[initial_labels]
    new_labels[initial_labels == -1] = -1
    print(f"{np.max(labels) + 1} : Clusters Found!")
    vectors_and_labels = np.hstack([vectors, new_labels[:, np.newaxis]])
    new_signal = self._deepcopy_with_new_data(data=vectors_and_labels)
    new_signal.axes_manager.signal_axes[0].size = (
        new_signal.axes_manager.signal_axes[0].size + 1
    )
    new_signal.is_clustered = True
    return new_signal


def reduced_subtract(min_vector, v2):
    return np.abs(v2 - np.round(v2 / min_vector) * min_vector)


reduced_subtract(60, 120)


import itertools


def get_filtered_combinations(
    pks,
    num,
    radial_index=0,
    angle_index=1,
    intensity_index=2,
    intensity_threshold=None,
    min_angle=None,
    min_k=None,
):
    """
    Creates combinations of `num` peaks but forces at least one of the combinations to have
    an intensity higher than the `intensity_threshold`.
    This filter is useful for finding high intensity features but not losing lower intensity
    paired features which contribute to symmetry etc.
    """
    angles = pks[:, angle_index]
    k = pks[:, radial_index]

    angle_combos = list(itertools.combinations(angles, num))
    k_combos = list(itertools.combinations(k, num))
    # Filtering out combinations with only diffraction from vectors below the intensity threshold
    if intensity_threshold is not None:
        intensity_combos = itertools.combinations(intensity, num)
        has_min_intensity = np.array(
            [any(np.array(i) > intensity_threshold) for i in intensity_combos]
        )
    else:
        has_min_intensity = True
    # Filtering out combinations where there are two peaks close to each other
    if min_angle is not None:
        above_angle = np.array(
            [
                all(
                    [
                        np.abs(np.subtract(*c)) > min_angle
                        for c in itertools.combinations(a, 2)
                    ]
                )
                for a in angle_combos
            ]
        )
    else:
        above_angle = True
    # Filtering out combinations of diffraction vectors at different values for k.
    if min_k is not None:
        in_k_range = np.array(
            [np.mean(np.abs(np.subtract(np.mean(k), k))) < min_k for k in k_combos]
        )
    else:
        in_k_range = True

    in_combos = above_angle * has_min_intensity * in_k_range
    if np.all(in_combos):
        combos = angle_combos
        combos_k = [np.mean(ks) for ks in k_combos]

    else:
        combos = [c for c, in_c in zip(angle_combos, in_combos) if in_c]
        combos_k = [
            np.mean(ks) for ks, in_range in zip(k_combos, in_combos) if in_range
        ]
    return combos, combos_k


import itertools


def get_three_angles(
    pks,
    k_index=0,
    angle_index=1,
    intensity_index=2,
    intensity_threshold=None,
    accept_threshold=0.05,
    min_k=0.1,
    include_multi=False,
    return_min=True,
    multi=True,
    min_angle=None,
):
    """
    This function takes the angle between three points and determines the angle between them,
    returning the angle if it is repeated using the `accept_threshold` to measure the acceptable
    difference between angle a and angle b
           o
           |
           |_   angle a
           | |
           x--------o
           |_|
           |    angle b
           |
           o
    """
    three_angles = []
    min_angles = []
    combos, combo_k = get_filtered_combinations(
        pks,
        3,
        angle_index=angle_index,
        intensity_index=intensity_index,
        intensity_threshold=intensity_threshold,
        min_angle=min_angle,
        min_k=min_k,
    )
    for c, k in zip(combos, combo_k):
        angular_seperations = get_angles(c)
        min_ind = np.argmin(angular_seperations)
        min_sep = angular_seperations[min_ind]
        angular_seperations = np.delete(angular_seperations, min_ind)
        if multi:  # test to see if any of the angles are multiples of each other
            remain = [reduced_subtract(min_sep, a) for a in angular_seperations]
            remain = [np.abs(f) < accept_threshold for f in remain]
            is_symetric = np.all(remain)
        else:
            is_symetric = np.any(
                np.abs((angular_seperations - min_sep)) < accept_threshold
            )
        if is_symetric:
            if not return_min:
                for a in angular_seperations:
                    three_angles.append(a)
                three_angles.append(min_sep)
            else:
                min_angle = np.min(c)
                num_times = np.round(min_angle / min_sep)
                three_angles.append(
                    [min_angle, min_sep, np.abs(min_angle - (num_times * min_sep)), k]
                )
    if len(three_angles) == 0:
        three_angles = np.empty((0, 4))
    return np.array(three_angles)


def unwrap(pks):
    if len(pks) == 0:
        return np.empty((0, 3))
    r = np.linalg.norm(pks[:, :2], axis=1)  # ignore the intensity
    theta = np.arctan2(pks[:, 0], pks[:, 1])
    return np.stack((r, theta, pks[:, 2]), axis=1)


def filter_mag(z, min_mag, max_mag):
    norm = np.linalg.norm(z[:, :2], axis=1)
    in_range = norm < max_mag * (norm > min_mag)
    return z[in_range]


import numpy as np
from copy import deepcopy


def angles_to_markers(angles, signal, polar=True, return_marker=False, **kwargs):
    """Convert a set of angles to markers.

    Angles are set as [Initial Position, Angular Seperation, Reduced Position, k]

    """
    new_angles = deepcopy(angles.data)

    # Sorting based on navigation position
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_angles[:, 0] = np.round((new_angles[:, 0] - x_axis.offset) / x_axis.scale)
    new_angles[:, 1] = np.round((new_angles[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_angles[:, 1], new_angles[:, 0]))
    sorted_peaks = new_angles[ind]
    x, y = signal.axes_manager.signal_axes
    # Create Ragged arrays
    shape = signal.axes_manager.navigation_shape
    by_ind_peaks = np.empty(shape, dtype=object)
    by_ind_colors = np.empty(shape, dtype=object)
    num_labels = np.max(new_angles[:, -1])
    # Random Colors
    colors_by_index = (
        np.random.random((int(num_labels + 1), 3)) * 0.9
    )  # (Stay away from white)
    colors_by_index = np.vstack((colors_by_index, [1, 1, 1]))
    # Serach sorted for speed
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[0]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[0] + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[1]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[1] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            # Get positions
            initial_theta = x_inds[lo_y:hi_y, 2]
            angle_seperation = x_inds[lo_y:hi_y, 3]
            k = x_inds[lo_y:hi_y, 5]
            # make optional for unlabeled...
            labels = np.array(x_inds[lo_y:hi_y, -1], dtype=int)
            # Compute angles based on 3 angle seperation
            angles = [initial_theta + angle_seperation * i for i in [0, 1, 2]]
            y_values = np.hstack([np.cos(a) * k for a in angles])
            x_values = np.hstack([np.sin(a) * k for a in angles])
            labels = np.hstack((labels, labels, labels))
            by_ind_peaks[i, j] = np.stack((y_values, x_values), axis=1)
            by_ind_colors[i, j] = colors_by_index[labels]
    return by_ind_peaks, by_ind_colors, colors_by_index


def get_vector_centers(vect, label_ind=-2, columns=[2, 3]):
    sub_labels = np.unique(vect[:, label_ind])
    centers = np.array(
        [np.mean(vect[vect[:, label_ind] == l], axis=0) for l in sub_labels]
    )
    return centers[:, columns]
