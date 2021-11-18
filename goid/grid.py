import os
import luigi
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import find_contours
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf, UnivariateSpline
from skimage.io import imread, imsave

from goid.skeleton import find_longest_skeleton_path

from goid.luigi_utils import BaseGoidTask, ExternalInputFile
from goid.skeleton import SkeletonTask, clear_border


def smooth_path(path, n_points=100, extent=(0, 0), smoothing_factor=None):
    '''spline smoothing interpolation'''

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)  #/distance[-1]

    new_dists = np.linspace(-extent[0], distance.max() + extent[1], n_points)

    if smoothing_factor is None:
        smoothing_factor = len(distance)
    splines = [
        UnivariateSpline(distance, coords, k=1, s=smoothing_factor)
        for coords in path.T
    ]
    new_pts = np.vstack([spl(new_dists) for spl in splines]).T

    return new_pts


def extrapolate_path_to_boundary(path, mask, n_points=100):
    '''smooth and extrapolate path till the boundary of the object'''

    dist = distance_transform_edt(mask)
    extent_backward = dist[tuple(path[0])]
    extent_forward = dist[tuple(path[-1])]

    return smooth_path(path,
                       n_points=n_points,
                       extent=(extent_backward, extent_forward))


def closest_point_on_path(pt, path):
    '''return the index of the point in path that is the closes to pt'''

    if pt.ndim < path.ndim:
        pt = pt[None]
    return np.argmin(cdist(pt, path))


def equidistant_point_on_path(pt1, pt2, path):
    '''return the index of the point in path that is the equidistant to p1, p2'''

    if pt1.ndim < path.ndim:
        pt1 = pt1[None]

    if pt2.ndim < path.ndim:
        pt2 = pt2[None]

    return np.argmin(np.abs(cdist(pt1, path) - cdist(pt2, path)))


def split_path(path, idxs):
    '''Returns a list of subpath split at selected indices. assumes path is a loop'''

    reverse_sort_lut = np.argsort(np.argsort(idxs))
    idxs = sorted(idxs)

    segments = []
    for start, end in zip(idxs[:-1], idxs[1:]):
        segments.append(path[start:end])

    # add looping segment, skip last point because first and last are the same
    segments.append(np.concatenate([path[idxs[-1]:-1], path[:idxs[0]]],
                                   axis=0))

    # reorder according to original idxs order
    segments = [segments[i] for i in reverse_sort_lut]

    return segments


def normal_unit_vector(path, idx, scale=1.):
    '''Return 2 points defining a normal vector at position idx of path'''

    diff = np.diff(path, axis=0)[idx]
    normal_offset = np.array([-diff[1], diff[0]])
    normal_offset = normal_offset / np.linalg.norm(normal_offset)

    p = path[idx]
    return np.stack([p, p + scale * normal_offset], axis=0)


def dist_to_line(p, line_points):
    '''Return the distance between a point p and a line defined by 2 points'''
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    if p.ndim == 1:
        p = p[None]

    diff = line_points[1] - line_points[0]

    numerator = diff[1] * p[:, 0] - diff[0] * p[:, 1] + line_points[
        1, 0] * line_points[0, 1] - line_points[1, 1] * line_points[0, 0]
    numerator = np.abs(numerator)

    return numerator / np.linalg.norm(diff)


def find_template_corners(mask, skel_path, debug_plot=False):
    '''Find organoid "corners" needed to align a grid'''

    contour = find_contours(mask, 0.5)[0]

    # find points at both ends of the organoid
    endA, endB = extrapolate_path_to_boundary(skel_path, mask,
                                              n_points=2)  # include smoothing
    endA_id = closest_point_on_path(endA, contour)
    endB_id = closest_point_on_path(endB, contour)
    if endA_id > endB_id:
        # clockwise contour, if needed flip skel path
        endA_id, endB_id = sorted([endA_id, endB_id])
        skel_path = skel_path[::-1]
    endA = contour[endA_id]
    endB = contour[endB_id]

    # split in quarters
    l1 = (endB_id - endA_id)
    mid1_id = endA_id + l1 // 2
    l2 = endA_id + (len(contour) - endB_id)
    mid2_id = (endB_id + l2 // 2) % len(contour)
    q1A, q1B, q2B, q2A = split_path(contour,
                                    [endA_id, mid1_id, endB_id, mid2_id])

    # remove endpoints so that "corners" falling on segment ends are separated by > 1 pt
    q1A = q1A[1:-1]
    q1B = q1B[1:-1]
    q2B = q2B[1:-1]
    q2A = q2A[1:-1]

    # find corners: points on contour that are equidistant from skeleton ends and extended skeleton ends
    a1 = equidistant_point_on_path(endA, skel_path[0], q1A)
    a1 = q1A[a1]
    a2 = equidistant_point_on_path(endA, skel_path[0], q2A)
    a2 = q2A[a2]

    b1 = equidistant_point_on_path(endB, skel_path[-1], q1B)
    b1 = q1B[b1]
    b2 = equidistant_point_on_path(endB, skel_path[-1], q2B)
    b2 = q2B[b2]

    corners = np.array([a1, b1, b2, a2])
    corner_idxs = [closest_point_on_path(c, contour) for c in corners]

    if debug_plot:
        plt.figure(figsize=(12, 12))
        plt.imshow(mask, cmap='Greys_r')
        plt.plot(skel_path[:, 1], skel_path[:, 0], linewidth=5, color='r')

        plt.plot(q1A[:, 1], q1A[:, 0], linewidth=3, color='purple')
        plt.scatter(a1[1], a1[0], color='purple', s=100)
        plt.plot(q1B[:, 1],
                 q1B[:, 0],
                 linewidth=3,
                 color='purple',
                 dashes=[1, 1])
        plt.scatter(b1[1], b1[0], color='purple', s=100, facecolors='none')

        plt.plot(q2A[:, 1], q2A[:, 0], linewidth=3, color='orange')
        plt.scatter(a2[1], a2[0], color='orange', s=100)
        plt.plot(q2B[:, 1],
                 q2B[:, 0],
                 linewidth=3,
                 color='orange',
                 dashes=[1, 1])
        plt.scatter(b2[1], b2[0], color='orange', s=100, facecolors='none')

        plt.scatter(endA[1], endA[0], color='r', s=100)
        plt.scatter(endB[1], endB[0], color='r', facecolors='none', s=100)

        plt.scatter(skel_path[0, 1], skel_path[0, 0], color='red', s=100)
        plt.scatter(skel_path[-1, 1],
                    skel_path[-1, 0],
                    color='red',
                    s=100,
                    facecolors='none')

        plt.show()

    return corner_idxs, contour


def map_grid(corner_idxs,
             contour,
             n_points_min=100,
             n_points_max=3000,
             step_size=2,
             fit_n_points=50):
    '''Maps a rectangular grid to a surface defined by contour and 
    corner indices with thin-plate interpolation.
    
    Notes:
        - opposite edges are split uniformly in n_segments_max
        - interpolator is build with compute_n_segments_max and n_segments_max with step_size
    '''

    # TODO if fix n_points instead of fix step size works --> clean up

    # split contour using corners
    segments = split_path(contour, corner_idxs)

    sl = [len(s) for s in segments]

    # max of pair of segments on opposit sides
    n_points_interp = [
        np.max(sl[::2]).astype(int) // step_size,
        np.max(sl[1::2]).astype(int) // step_size
    ]
    n_points_interp = np.array(n_points_interp).clip(n_points_min,
                                                     n_points_max)

    n_points = [fit_n_points, fit_n_points, fit_n_points, fit_n_points]

    segments = [
        smooth_path(s, n_p + 2)[1:-1] for s, n_p in zip(segments, n_points)
    ]
    n_steps1 = len(segments[0])  # TODO clean up (same as n_points[0])
    n_steps2 = len(segments[1])

    # build regular (r) grid coords and
    # TODO find cleaner way to get coords along rectangle
    rg1 = list(range(n_points[0])) + [
        n_points[0] - 1 for _ in range(n_points[1])
    ] + list(range(n_points[0]))[::-1] + [0 for _ in range(n_points[1])]
    rg2 = [n_steps2 - 1 for _ in range(n_points[0])] + list(range(
        n_points[1]))[::-1] + [0 for _ in range(n_points[0])] + list(
            range(n_points[1]))
    rg1 = np.array(rg1)
    rg2 = np.array(rg2)

    # corresponding deformed (d) pair
    dg1 = np.concatenate([s[:, 0] for s in segments])
    dg2 = np.concatenate([s[:, 1] for s in segments])

    irg1, irg2 = np.meshgrid(
        np.arange(0, n_points[0], n_points[0] / n_points_interp[0]),
        np.arange(0, n_points[1], n_points[1] / n_points_interp[1]))

    x_interpolator = Rbf(rg1, rg2, dg1, function='thin_plate', smooth=0.001)
    y_interpolator = Rbf(rg1, rg2, dg2, function='thin_plate', smooth=0.001)
    idg1 = x_interpolator(irg1, irg2)
    idg2 = y_interpolator(irg1, irg2)

    grid = np.stack([idg1, idg2], axis=-1)

    # clip to image bounds
    max_bounds = contour.max(axis=0)
    grid[..., 0] = grid[..., 0].clip(0, max_bounds[0] - 1)
    grid[..., 1] = grid[..., 1].clip(0, max_bounds[1] - 1)

    return grid.astype(np.float32)


def grid_label(grid, mask):
    '''Splits mask in superpixels centered on grid.
    
    Notes:
    cells ids correspond to their grid coordinates encoded as x: 16 bits high, y: 16 bits low.
    Vertical/horizontal segments can be extracted by taking 16 high/low bits
    '''

    labels = np.zeros_like(mask, dtype=np.uint32)

    if np.all(grid == 0):
        return labels

    grid = np.round(grid).astype(int)
    grid_coord = np.meshgrid(np.arange(0, grid.shape[1], 1),
                             np.arange(0, grid.shape[0], 1))
    grid_labels = grid_coord[0] * 2**16 + grid_coord[1]

    grid = grid.reshape(-1, 2)
    grid_labels = grid_labels.reshape(-1)

    #place label id on interpolated coords
    labels[tuple(grid.T)] = grid_labels

    # nearest inrpolation of the rest
    distance = distance_transform_edt(labels == 0)
    # mask afterwards because some grid points might be slightly outside the mask
    labels = watershed(distance, markers=labels).astype(np.int32)
    labels[~mask] = 0

    return labels.astype(np.uint32)


def grid_to_segments(grid_labels, mask, n_segments=3, axis='l'):
    '''subsample the grid labels generated by grid_label() and return segments labels along the length or width axis'''

    if axis == 'l':
        segments = grid_labels // 2**16
    else:
        segments = grid_labels % 2**16

    if segments.max() < n_segments:
        return np.zeros_like(segments, dtype=np.uint8)

    segments = np.digitize(segments / segments.max(),
                           np.linspace(0, 1., n_segments + 1)[1:-1])
    segments[mask] += 1

    if segments.max() < 256:
        segments = segments.astype(np.uint8)
    else:
        segments = segments.astype(np.uint16)

    return segments


def normalize_grid_direction(grid, flu):
    '''Normalizes grid direction along the longitudinal 
    axis so that the side with the highest mean intensity 
    has higher label ids'''

    grid_shape = grid.shape

    # sample flu along the grid
    flat_grid = np.round(grid).astype(int).reshape(-1, 2)
    grid_values = flu[tuple(flat_grid.T)].reshape(grid_shape[:2])

    if grid_values[:, :grid_shape[1] //
                   2].mean() > grid_values[:, grid_shape[1] // 2:].mean():
        # flip first coordinate
        grid = grid[:, ::-1]

    return grid


def compute_grid(mask, skel, orient_channel):
    '''Combines all steps needed to map a template grid on given gastruloid mask.
    
    Args:
        mask: object mask
        skel: skeleton image
        orient_channel: intensity channel used to normalize grid orientation
    '''

    mask = clear_border(mask)
    skel_path = find_longest_skeleton_path(skel)
    corner_idxs, contour = find_template_corners(mask, skel_path)
    grid = map_grid(corner_idxs, contour)
    grid = normalize_grid_direction(grid, orient_channel)

    return grid


class GridTask(BaseGoidTask):
    ''''''

    orientation_channel = luigi.Parameter(
        1, description='channel id or stain used to orient the grid')
    longitudinal_segments = luigi.ListParameter(
        description='list of number of labelled segments to output')

    def requires(self):
        row = self.filter_dc_mip(ch=self.orientation_channel)[0]

        return {
            'orient_ch': ExternalInputFile(path=self.dcrow_to_path(row)),
            'skel_out': SkeletonTask(dc_mip=self.dc_mip)
        }

    def output(self):
        out = {}
        # segments as label image
        for n_seg in self.longitudinal_segments:
            row = dict(self.dc_mip[0])
            row['subdir'] = 'SEGMENTS_L{}'.format(n_seg)
            out[('l', n_seg)] = luigi.LocalTarget(self.dcrow_to_path(row))

        # raw grid coordinates (e.g. to compute height, width features)
        row_grid = dict(self.dc_mip[0], subdir='GRID', ext='npz')
        return out, luigi.LocalTarget(self.dcrow_to_path(row_grid))

    def run(self):

        orient_channel = imread(self.input()['orient_ch'].path, img_num=0)
        skel = imread(self.input()['skel_out']['skeleton'].path).astype(bool)
        mask = imread(
            self.input()['skel_out']['mask_with_sep'].path).astype(bool)

        if np.any(mask):
            grid = compute_grid(mask, skel, orient_channel)
            grid_labels = grid_label(grid, mask)
        else:
            grid = 0
            grid_labels = np.zeros_like(mask)

        for (axis, n_seg), target in self.output()[0].items():
            segments = grid_to_segments(grid_labels, mask, n_seg, axis)
            out_path = target.path
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imsave(out_path, segments.astype(np.uint16), compress=9)

        self.output()[1].makedirs()
        np.savez_compressed(self.output()[1].path, arr=grid)
