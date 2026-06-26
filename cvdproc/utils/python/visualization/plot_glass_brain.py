import os
import nibabel as nib
import numpy as np
import pyvista as pv
from nibabel.streamlines import load as load_tck
from skimage.measure import marching_cubes


class GlassBrainPlotter:
    def __init__(
        self,
        t1_path,
        mask_file=None,
        brain_iso_value=120,
        brain_opacity=0.10,
        background="white",
    ):
        self.t1_path = t1_path
        self.mask_file = mask_file
        self.brain_iso_value = brain_iso_value
        self.brain_opacity = brain_opacity
        self.background = background

        self.roi_configs = []
        self.tract_configs = []

        self.plotter = pv.Plotter()
        self.plotter.enable_anti_aliasing()
        self.plotter.set_background(self.background)

    @staticmethod
    def apply_affine_to_points(points, affine):
        points_h = np.c_[points, np.ones(points.shape[0])]
        points_world = (affine @ points_h.T).T[:, :3]
        return points_world

    @staticmethod
    def create_isosurface_mesh(
        data,
        iso_value,
        affine,
        smooth_iter=400,
        relaxation_factor=0.03,
        taubin_iter=200,
        taubin_passband=0.1,
    ):
        verts_voxel, faces, _, _ = marching_cubes(
            volume=data,
            level=iso_value
        )

        verts_world = GlassBrainPlotter.apply_affine_to_points(
            verts_voxel,
            affine
        )

        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)

        mesh = pv.PolyData(verts_world, faces_pv)

        if smooth_iter > 0:
            mesh = mesh.smooth(
                n_iter=smooth_iter,
                relaxation_factor=relaxation_factor,
                feature_smoothing=False,
                boundary_smoothing=True
            )

        if taubin_iter > 0:
            mesh = mesh.smooth_taubin(
                n_iter=taubin_iter,
                pass_band=taubin_passband
            )

        mesh = mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            inplace=False
        )

        return mesh

    @staticmethod
    def create_direction_encoded_streamline_polydata(
        streamlines_world,
        max_streamlines=None
    ):
        if max_streamlines is not None:
            streamlines_world = streamlines_world[:max_streamlines]

        all_points = []
        all_lines = []
        all_colors = []

        point_offset = 0

        for sl in streamlines_world:
            sl = np.asarray(sl, dtype=np.float32)

            if sl.shape[0] < 2:
                continue

            n_points = sl.shape[0]
            all_points.append(sl)

            line = np.concatenate(
                [[n_points], np.arange(point_offset, point_offset + n_points)]
            )
            all_lines.append(line)

            directions = np.diff(sl, axis=0)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1.0

            directions = directions / norms
            segment_colors = np.abs(directions)

            point_colors = np.zeros((n_points, 3), dtype=np.uint8)
            point_colors[:-1] = (segment_colors * 255).astype(np.uint8)
            point_colors[-1] = point_colors[-2]

            all_colors.append(point_colors)
            point_offset += n_points

        if len(all_points) == 0:
            raise ValueError("No valid streamlines were found.")

        points = np.vstack(all_points)
        lines = np.hstack(all_lines).astype(np.int64)
        colors = np.vstack(all_colors)

        fiber_polydata = pv.PolyData()
        fiber_polydata.points = points
        fiber_polydata.lines = lines
        fiber_polydata["RGB"] = colors

        return fiber_polydata

    def add_brain(self):
        t1_img = nib.load(self.t1_path)
        t1_data = t1_img.get_fdata()

        if self.mask_file is not None and os.path.exists(self.mask_file):
            mask_img = nib.load(self.mask_file)
            mask_data = mask_img.get_fdata()

            if mask_data.shape != t1_data.shape:
                raise ValueError("Mask shape does not match T1 shape.")

            t1_data[mask_data == 0] = 0

        brain_mesh = self.create_isosurface_mesh(
            data=t1_data,
            iso_value=self.brain_iso_value,
            affine=t1_img.affine,
            smooth_iter=400,
            relaxation_factor=0.03,
            taubin_iter=200,
            taubin_passband=0.1
        )

        self.plotter.add_mesh(
            brain_mesh,
            color="lightgray",
            opacity=self.brain_opacity,
            smooth_shading=True,
            specular=0.25,
            specular_power=15
        )

    def add_roi(
        self,
        path,
        color,
        opacity=1.0,
        iso_value=0.5,
        smooth_iter=80,
        relaxation_factor=0.05,
        taubin_iter=50,
        taubin_passband=0.1,
    ):
        self.roi_configs.append(
            {
                "path": path,
                "color": color,
                "opacity": opacity,
                "iso_value": iso_value,
                "smooth_iter": smooth_iter,
                "relaxation_factor": relaxation_factor,
                "taubin_iter": taubin_iter,
                "taubin_passband": taubin_passband,
            }
        )

    def add_tract(
        self,
        path,
        line_width=3,
        opacity=1.0,
        max_streamlines=None,
        render_lines_as_tubes=True,
    ):
        self.tract_configs.append(
            {
                "path": path,
                "line_width": line_width,
                "opacity": opacity,
                "max_streamlines": max_streamlines,
                "render_lines_as_tubes": render_lines_as_tubes,
            }
        )

    def render_rois(self):
        for roi in self.roi_configs:
            roi_img = nib.load(roi["path"])
            roi_data = roi_img.get_fdata()

            roi_mesh = self.create_isosurface_mesh(
                data=roi_data,
                iso_value=roi["iso_value"],
                affine=roi_img.affine,
                smooth_iter=roi["smooth_iter"],
                relaxation_factor=roi["relaxation_factor"],
                taubin_iter=roi["taubin_iter"],
                taubin_passband=roi["taubin_passband"]
            )

            self.plotter.add_mesh(
                roi_mesh,
                color=roi["color"],
                opacity=roi["opacity"],
                smooth_shading=True,
                specular=0.35,
                specular_power=20
            )

    def render_tracts(self):
        for tract in self.tract_configs:
            tractogram = load_tck(tract["path"])
            streamlines_world = tractogram.streamlines

            print(f"Tract file: {tract['path']}")
            print(f"Number of streamlines: {len(streamlines_world)}")

            fiber_polydata = self.create_direction_encoded_streamline_polydata(
                streamlines_world=streamlines_world,
                max_streamlines=tract["max_streamlines"]
            )

            print(f"Number of fiber points: {fiber_polydata.n_points}")
            print(f"Number of fiber lines: {fiber_polydata.n_lines}")

            self.plotter.add_mesh(
                fiber_polydata,
                scalars="RGB",
                rgb=True,
                line_width=tract["line_width"],
                opacity=tract["opacity"],
                render_lines_as_tubes=tract["render_lines_as_tubes"]
            )

    def show(self, camera_position="iso"):
        self.add_brain()
        self.render_rois()
        self.render_tracts()

        self.plotter.camera_position = camera_position
        self.plotter.show()


if __name__ == "__main__":

    plotter = GlassBrainPlotter(
        t1_path=r"/mnt/e/Codes/cvdproc/cvdproc/data/standard/MNI152/MNI152_T1_1mm_brain.nii.gz",
        mask_file="",
        brain_iso_value=4750,
        brain_opacity=0.10,
        background="white",
    )

    plotter.add_roi(
    path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/lh.label_234.nii.gz",
    color=(0.90, 0.35, 0.30),
    opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/rh.label_234.nii.gz",
        color=(0.90, 0.35, 0.30),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/lh.label_236.nii.gz",
        color=(0.95, 0.65, 0.20),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/rh.label_236.nii.gz",
        color=(0.95, 0.65, 0.20),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/lh.label_238.nii.gz",
        color=(0.20, 0.70, 0.75),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/rh.label_238.nii.gz",
        color=(0.20, 0.70, 0.75),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/lh.label_240.nii.gz",
        color=(0.55, 0.40, 0.85),
        opacity=1.0,
    )

    plotter.add_roi(
        path=r"/mnt/e/Codes/cvdproc/cvdproc/data/demo/cvs_avg35_inMNI152_hipsta/rh.label_240.nii.gz",
        color=(0.55, 0.40, 0.85),
        opacity=1.0,
    )

    # plotter.add_tract(
    #     path=r"/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0106/ses-baseline/visual_pathway_analysis/sub-HC0106_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OT_streamlines.tck",
    #     line_width=3,
    #     opacity=1.0,
    #     max_streamlines=None,
    # )

    plotter.show(camera_position="iso")