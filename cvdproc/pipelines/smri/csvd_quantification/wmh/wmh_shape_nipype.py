import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import marching_cubes, mesh_surface_area
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import label, sum
import math
from .BC3D import BoxCountingMethod_Solid, FD_plot

from .....bids_data.rename_bids_file import rename_bids_file

class WMHShapeInputSpec(BaseInterfaceInputSpec):
    wmh_mask = File(exists=True, desc="A WMH mask file (in standard MNI space)", mandatory=True)
    #vent_mask = File(exists=True, desc="A ventricle mask file (in standard MNI space)", mandatory=True) # Needed for plots
    threshold = Int(desc="The threshold for the voxels to be considered", default_value=5)
    save_plots = Bool(desc="Whether to generate plots", default_value=False)

    output_dir = Directory(desc="The output directory to save the shape features", mandatory=True)
    wmh_labeled_filename = Str(desc="The name of the WMH labeled file", mandatory=True)
    shape_csv_filename = Str(desc="The name of the CSV file to save the shape features", mandatory=True, default_value='wmh_shape_features.csv')
    shape_csv_avg_filename = Str(desc="The name of the CSV file to save the average shape features", mandatory=True, default_value='average_wmh_shape_features.csv')

class WMHShapeOutputSpec(TraitedSpec):
    wmh_labeled = File(exists=True, desc="The WMH labeled file")
    shape_csv = File(exists=True, desc="The CSV file containing the shape features")
    shape_csv_avg = File(exists=True, desc="The CSV file containing the average shape features")

class WMHShape(BaseInterface):
    input_spec = WMHShapeInputSpec
    output_spec = WMHShapeOutputSpec

    # Function to plot three views in a row without axes 病灶可视化
    # 可视化PWMH、DWMH、和脑室的三维图像
    # 1：PWMH，2：DWMH，3：侧脑室
    def _plot_three_views_in_row(self, vertices1, faces1, vertices2, faces2, vertices3, faces3, angles):
        fig = plt.figure(figsize=(30, 10))
        for i, angle in enumerate(angles):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            mesh1 = Poly3DCollection(vertices1[faces1], alpha=0.2, edgecolor='darkorange', facecolor='orange') # 设置PWMH的颜色
            ax.add_collection3d(mesh1)

            mesh2 = Poly3DCollection(vertices2[faces2], alpha=0.2, edgecolor='olivedrab', facecolor='yellowgreen') # 设置DWMH的颜色
            ax.add_collection3d(mesh2)

            mesh3 = Poly3DCollection(vertices3[faces3], alpha=0.2, edgecolor='dimgray', facecolor='grey') # 设置脑室的颜色
            ax.add_collection3d(mesh3)

            ax.view_init(azim=angle[0], elev=angle[1])
            #ax.grid(False)
            ax.axis('off') # 关闭坐标轴

            # Auto scaling to the mesh size
            scale = np.concatenate([vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], vertices2[:, 0], vertices2[:, 1],
                                    vertices2[:, 2], vertices3[:, 0], vertices3[:, 1], vertices3[:, 2]]).flatten()
            ax.auto_scale_xyz(scale, scale, scale)

        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show()

    # 目前的主要问题：1.算出来solidity和convexity显著负相关，但理论上是正相关（至少在2D图像上是）；
    # 2.concavity index的计算公式和文献中不一致（1-改成了2-），是否是因为3D图像的原因？；3. 分形维数r->0或1有区别吗？
    # Function to calculate the shape features
    # SHAPE FEATURES:
    ## solidity = volume / convex hull volume
    ## convexity = convex hull area / area
    ## concavity index = ((2 - convexity) ** 2 + (1 - solidity) ** 2) ** (1 / 2)
    ## inverse shape index = (area ** 3) ** 0.5 / (6 * volume * (pi) ** 0.5))
    ## fractal dimension (FD): box counting
    ## eccentricity = minor axis / major axis
    def _calculate_shape_features(self, data):
        verts, faces, normals, values = marching_cubes(data)

        mc_surface_area = mesh_surface_area(verts, faces)
        mc_volume = np.count_nonzero(data)

        hull = ConvexHull(verts)
        convex_hull_area = hull.area
        convex_hull_volume = hull.volume

        convexity = convex_hull_area / mc_surface_area
        solidity = mc_volume / convex_hull_volume
        concavity_index = ((2 - convexity) ** 2 + (1 - solidity) ** 2) ** (1 / 2)
        inverse_sphericity_index = (mc_surface_area ** 3) ** 0.5 / (6 * mc_volume * (math.pi) ** 0.5)

        # Eccentricity: minor axis / major axis
        hull_points = hull.points[hull.vertices]
        diameters = np.sqrt(((hull_points[:, None, :] - hull_points) ** 2).sum(axis=2)) # 计算凸包中各点之间的距离
        max_diameter = np.max(diameters) # 凸包中最大的距离（major axis）
        end_points = np.unravel_index(np.argmax(diameters), diameters.shape) # 最大距离的两个点的索引
        direction_long = hull_points[end_points[1]] - hull_points[end_points[0]] # 最大距离的方向向量
        direction_long /= np.linalg.norm(direction_long) # 最大距离的方向向量的单位向量

        max_length = 0 # 初始化最大长度（minor axis）
        short_axis_points = (None, None)
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                line_direction = hull_points[j] - hull_points[i]
                line_direction /= np.linalg.norm(line_direction)
                if np.abs(np.dot(line_direction, direction_long)) < 1e-5: # 如果两个向量的夹角小于1e-5，则认为两个向量正交
                    length = np.linalg.norm(hull_points[i] - hull_points[j])
                    if length > max_length:
                        max_length = length
                        short_axis_points = (hull_points[i], hull_points[j])

        # 如果没有完全正交的两个向量，则选择点乘最小的两个向量（最接近正交）
        min_dot_product = np.inf
        if short_axis_points[0] is None or short_axis_points[1] is None:
            for i in range(len(hull_points)):
                for j in range(i + 1, len(hull_points)):
                    line_direction = hull_points[j] - hull_points[i]
                    line_length = np.linalg.norm(line_direction)
                    if line_length > 0:
                        line_direction /= line_length  # 计算单位方向向量
                        dot_product = np.abs(np.dot(line_direction, direction_long))  # 计算点积的绝对值

                        if dot_product < min_dot_product:
                            min_dot_product = dot_product
                            max_length = line_length
                            #short_axis_points = (hull_points[i], hull_points[j])
        eccentricity = max_length / max_diameter

        # Fractal Dimension (FD): box counting
        _, _, _, coeff, _, _, _ = BoxCountingMethod_Solid.BC_Solid(data)
        fractal_dimension = coeff[0]

        return mc_surface_area, mc_volume, convex_hull_area, convex_hull_volume, convexity, solidity, concavity_index, \
            inverse_sphericity_index, eccentricity, fractal_dimension

    def _shape_features_plot(self, data, name, shape_features_plot_dir):
        ## Fractal Dimension Plot
        para = BoxCountingMethod_Solid.BC_Solid(data)
        path = os.path.join(shape_features_plot_dir, '{0}_FD.png'.format(name))
        FD_plot.Draw_SCI(para, path)

        ## Lesion Plot
        vertices, faces, _, _ = marching_cubes(data)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='red', alpha=0.8)
        ax.axis('off')
        path = os.path.join(shape_features_plot_dir, '{0}_lesion_plot.png'.format(name))
        plt.savefig(path, dpi=300)
        plt.close('all')

        ## Convex Hull Plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()  # Remove the axes
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='blue', alpha=0.3)

        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            ax.plot(vertices[simplex, 0], vertices[simplex, 1], vertices[simplex, 2], color='red', linewidth=2)
        path = os.path.join(shape_features_plot_dir, '{0}_convex_hull_plot.png'.format(name))
        plt.savefig(path, dpi=300)
        plt.close('all')

        ## Eccentricity Plot
        hull_points = hull.points[hull.vertices]
        diameters = np.sqrt(((hull_points[:, None, :] - hull_points) ** 2).sum(axis=2))  # 计算凸包中各点之间的距离
        max_diameter = np.max(diameters)  # 凸包中最大的距离（major axis）
        end_points = np.unravel_index(np.argmax(diameters), diameters.shape)  # 最大距离的两个点的索引
        direction_long = hull_points[end_points[1]] - hull_points[end_points[0]]  # 最大距离的方向向量
        direction_long /= np.linalg.norm(direction_long)  # 最大距离的方向向量的单位向量

        max_length = 0  # 初始化最大长度（minor axis）
        short_axis_points = (None, None)
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                line_direction = hull_points[j] - hull_points[i]
                line_direction /= np.linalg.norm(line_direction)
                if np.abs(np.dot(line_direction, direction_long)) < 1e-5:  # 如果两个向量的夹角小于1e-5，则认为两个向量正交
                    length = np.linalg.norm(hull_points[i] - hull_points[j])
                    if length > max_length:
                        max_length = length
                        short_axis_points = (hull_points[i], hull_points[j])

        # 如果没有完全正交的两个向量，则选择点乘最小的两个向量（最接近正交）
        min_dot_product = np.inf
        if short_axis_points[0] is None or short_axis_points[1] is None:
            for i in range(len(hull_points)):
                for j in range(i + 1, len(hull_points)):
                    line_direction = hull_points[j] - hull_points[i]
                    line_length = np.linalg.norm(line_direction)
                    if line_length > 0:
                        line_direction /= line_length  # 计算单位方向向量
                        dot_product = np.abs(np.dot(line_direction, direction_long))  # 计算点积的绝对值

                        if dot_product < min_dot_product:
                            min_dot_product = dot_product
                            max_length = line_length
                            short_axis_points = (hull_points[i], hull_points[j])

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='blue', alpha=0.3)
        ax.plot(*zip(hull_points[end_points[0]], hull_points[end_points[1]]), color='r', linewidth=3)
        ax.plot(*zip(short_axis_points[0], short_axis_points[1]), color='lime', linewidth=3)
        azimuth, elevation = np.arctan2(hull_points[end_points[1]][1] - hull_points[end_points[0]][1],
                                        hull_points[end_points[1]][0] - hull_points[end_points[0]][
                                            0]) * 180 / np.pi, np.arctan2(
            hull_points[end_points[1]][2] - hull_points[end_points[0]][2], np.sqrt(
                (hull_points[end_points[1]][0] - hull_points[end_points[0]][0]) ** 2 + (
                            hull_points[end_points[1]][1] - hull_points[end_points[0]][1]) ** 2)) * 180 / np.pi
        ax.view_init(elev=elevation, azim=azimuth)
        ax.axis('off')
        path = os.path.join(shape_features_plot_dir, '{0}_eccentricity_plot.png'.format(name))
        plt.savefig(path)
        plt.close('all')

    def _run_interface(self, runtime):
        min_voxels = self.inputs.threshold

        plot = self.inputs.save_plots
        na_placeholder = None

        columns = ['Region', 'Surface Area', 'Voxels', 'Convex Hull Area', 'Convex Hull Volume', 'Convexity', 'Solidity',
                   'Concavity Index', 'Inverse Sphericity Index', 'Eccentricity', 'Fractal Dimension']
        wmh_shape_features_df = pd.DataFrame(columns=columns)

        if plot:
            plot_subdir = os.path.join(self.inputs.output_dir, 'plots')
            os.makedirs(plot_subdir, exist_ok=True)
        
        wmh_data = nib.load(self.inputs.wmh_mask).get_fdata()

        # Remove small regions
        labeled_wmh, num_features_wmh = label(wmh_data)
        voxels = sum(wmh_data, labeled_wmh, range(num_features_wmh + 1))
        remove = voxels < min_voxels
        remove_indices = np.where(remove)[0]
        for idx in remove_indices:
            wmh_data[labeled_wmh == idx] = 0
        labeled_wmh, num_features_wmh = label(wmh_data)
        print('{0} WMH regions detected'.format(num_features_wmh))

        wmh_labeled = np.zeros_like(wmh_data, dtype=np.int32)

        if wmh_data.any():
            for region_num in range(1, num_features_wmh + 1):
                region = (labeled_wmh == region_num).astype(np.int32)
                region[region != 0] = 1

                wmh_labeled[labeled_wmh == region_num] = region_num

                if plot:
                    # Save the shape features plots
                    name = 'WMH_region_{}'.format(region_num)
                    self._shape_features_plot(region, name, plot_subdir)

                shape_features = self._calculate_shape_features(region)
                wmh_shape_features_df.loc[len(wmh_shape_features_df)] = [f'WMH_region_{region_num}'] + list(shape_features)

            wmh_labeled_nii = nib.Nifti1Image(wmh_labeled, nib.load(self.inputs.wmh_mask).affine, nib.load(self.inputs.wmh_mask).header)
            nib.save(wmh_labeled_nii, os.path.join(self.inputs.output_dir, self.inputs.wmh_labeled_filename))
            self._wmh_labeled = os.path.join(self.inputs.output_dir, self.inputs.wmh_labeled_filename)

            subject_avg_data = wmh_shape_features_df[
                ['Convexity', 'Solidity', 'Concavity Index', 'Inverse Sphericity Index', 'Eccentricity', 'Fractal Dimension']
            ].mean()
        else:
            print('no WMH lesion detected')
            wmh_shape_features_df.loc[len(wmh_shape_features_df)] = [na_placeholder] * len(columns)

            subject_avg_data = {
                'Convexity': na_placeholder,
                'Solidity': na_placeholder,
                'Concavity Index': na_placeholder,
                'Inverse Sphericity Index': na_placeholder,
                'Eccentricity': na_placeholder,
                'Fractal Dimension': na_placeholder
            }

            self._wmh_labeled = ''

        self._shape_csv = os.path.join(self.inputs.output_dir, self.inputs.shape_csv_filename)
        wmh_shape_features_df.to_csv(self._shape_csv, index=False)

        subject_avg_df = pd.DataFrame([subject_avg_data])
        subject_avg_path = os.path.join(self.inputs.output_dir, self.inputs.shape_csv_avg_filename)
        subject_avg_df.to_csv(subject_avg_path, index=False)

        self._shape_csv_avg = subject_avg_path

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs['wmh_labeled'] = self._wmh_labeled
        outputs['shape_csv'] = self._shape_csv
        outputs['shape_csv_avg'] = self._shape_csv_avg

        return outputs