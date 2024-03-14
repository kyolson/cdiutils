import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import silx.io.h5py_utils
from tabulate import tabulate
import textwrap
import vtk
from vtk.util import numpy_support

from cdiutils.utils import (
    find_isosurface,
    zero_to_nan,
    make_support,
    symmetric_pad,
    CroppingHandler,
    rebin,
    oversampling_ratio
)
from cdiutils.load.bliss import BlissLoader
from cdiutils.load.spec import SpecLoader
from cdiutils.load.sixs import SIXS2022Loader
from cdiutils.load.p10 import P10Loader
from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.process.postprocess import PostProcessor
from cdiutils.process.plot import (
    preprocessing_detector_data_plot,
    summary_slice_plot,
    plot_direct_lab_orthogonalization_process,
    plot_q_lab_orthogonalization_process,
    plot_final_object_fft
)
from cdiutils.plot.colormap import RED_TO_TEAL
from cdiutils.plot.volume import plot_3d_surface_projections


def loader_factory(metadata: dict) -> BlissLoader | SpecLoader | SIXS2022Loader:
    """
    Load the right loader based on the beamline_setup parameter
    in the metadata dictionary
    """
    if metadata["beamline_setup"] == "ID01BLISS":
        return BlissLoader(
            experiment_file_path=metadata["experiment_file_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
            flatfield=metadata["flatfield_path"],
            # alien_mask=metadata["alien_mask"]

        )
    if metadata["beamline_setup"] == "ID01SPEC":
        return SpecLoader(
            experiment_file_path=metadata["experiment_file_path"],
            detector_data_path=metadata["detector_data_path"],
            edf_file_template=metadata["edf_file_template"],
            detector_name=metadata["detector_name"]
        )
    if metadata["beamline_setup"] == "SIXS2022":
        return SIXS2022Loader(
            experiment_data_dir_path=metadata["experiment_data_dir_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
        )
    if metadata["beamline_setup"] == "P10":
        return P10Loader(
            experiment_data_dir_path=metadata["experiment_data_dir_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
        )
    raise NotImplementedError("The provided beamline_setup is not valid.")


class BcdiProcessor:
    """
    A class to handle pre and post processing in a bcdi data analysis
    workflow.
    """
    def __init__(
            self,
            parameters: dict
    ) -> None:
        self.params = parameters

        self.loader = None
        self.space_converter = None

        self.detector_data = None

        # initialize the diffractometer angles (corresponding to eta,
        # phi, delta, nu at ID01)
        self.angles = {
            "sample_outofplane_angle": None,
            "sample_inplane_angle": None,
            "detector_outofplane_angle": None,
            "detector_inplane_angle": None
        }

        self.cropped_detector_data = None
        self.mask = None

        self.orthogonalized_object = None
        self.orthogonalized_intensity = None
        self.voxel_size = None
        self.structural_properties = {}
        self.averaged_dspacing = None
        self.averaged_lattice_parameter = None

        self.dump_dir = self.params["metadata"]["dump_dir"]
        self.scan = self.params["metadata"]["scan"]
        self.sample_name = self.params["metadata"]["sample_name"]

        # initialize figures
        self.figures = {
            "preprocessing": {
                "name": "centering_cropping_detector_data_plot",
                "debug": False
            },
            "postprocessing": {
                "name": "summary_slice_plot",
                "debug": False
            },
            "strain": {
                "name": "different_strain_methods",
                "debug": False
            },
            "displacement_gradient": {
                "name": "displacement_gradient",
                "debug": False
            },
            "amplitude": {
                "name": "amplitude_distribution_plot",
                "debug": False
            },
            "direct_lab_orthogonalization": {
                "name": "direct_lab_orthogonalization_plot",
                "debug": True
            },
            "q_lab_orthogonalization": {
                "name": "q_lab_orthogonalization_plot",
                "debug": False
            },
            "final_object_fft": {
                "name": "final_object_fft",
                "debug": False
            },
            "3d_strain": {
                "name": "3d_strain",
                "debug": False
            }
        }
        for value in self.figures.values():
            value["figure"] = None

        # initialise the loader and Space converter
        self.loader = loader_factory(self.params["metadata"])
        self.space_converter = SpaceConverter(
            energy=self.params["energy"],
            geometry=Geometry.from_name(
                self.params["metadata"]["beamline_setup"])
        )

    def load_data(
            self,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None
    ) -> None:
        """
        Load the raw detector data and motor positions.

        Args:
            roi (tuple[slice], optional): the region of interest on the
            detector frame. Defaults to None.
            binning_along_axis0 (int, optional): whether or not to bin
            the data along the axis0, ie the rocking curve direction.
            Defaults to None.

        Raises:
            ValueError: check whether detector data, motor positions and
            mask have been correctly loaded.
        """
        self.detector_data = self.loader.load_detector_data(
            scan=self.scan,
            roi=roi,
            binning_along_axis0=binning_along_axis0
        )
        self.angles.update(
            self.loader.load_motor_positions(
                scan=self.scan,
                roi=roi,
                binning_along_axis0=binning_along_axis0
            )
        )
        self.mask = self.loader.get_mask(
            channel=self.detector_data.shape[0],
            detector_name=self.params["metadata"]["detector_name"],
            roi=(slice(None), roi[1], roi[2]) if roi else None
        )
        if any(
                data is None
                for data in (self.detector_data, self.angles, self.mask)
        ):
            raise ValueError("Something went wrong during data loading.")

    def verbose_print(self, text: str, wrap: bool = True, **kwargs) -> None:
        if self.params["verbose"]:
            if wrap:
                wrapper = textwrap.TextWrapper(
                    width=80,
                    break_long_words=False,
                    replace_whitespace=False
                )
                text = "\n".join(wrapper.wrap(text))

            print(text,  **kwargs)

    # TODO: cyclomatic complexity of 28, this method should
    # be broken down into multiples functions, or a class
    # see: https://github.com/clatlan/cdiutils/issues/32
    def preprocess_data(self) -> None:
        """
        Run the preprocessing.

        Raises:
            ValueError: when light loading, user must specify either
            the exact final shape, with three integers, or provide a
            value for binning_along_axis0.
            ValueError: when light loading, user must specify the voxel
            where to crop the data at.
        """
        final_shape = tuple(self.params["preprocessing_output_shape"])

        if self.params["light_loading"]:
            if (
                    len(final_shape) != 3
                    and self.params["binning_along_axis0"] is None
            ):
                raise ValueError(
                    "final_shape must include 3 axis lengths or you must "
                    "provide binning_along_axis0 parameter if you want "
                    "light loading."
                )

            det_ref = self.params["det_reference_voxel_method"][-1]
            if isinstance(det_ref, str):
                raise ValueError(
                    "When light loading, det_reference_voxel_method must "
                    "contain a tuple indicating the position of the voxel you "
                    "want to crop the data at. Ex: [(100, 200, 200)]"
                )
            if len(det_ref) == 2 and len(final_shape) == 3:
                final_shape = final_shape[1:]
            elif len(det_ref) == 3 and len(final_shape) == 2:
                det_ref = det_ref[1:]

            roi = CroppingHandler.get_roi(final_shape, det_ref)
            if len(roi) == 4:
                roi = [None, None, roi[0], roi[1], roi[2], roi[3]]

            self.verbose_print(
                f"Light loading requested, will use ROI {roi} and bin along "
                "rocking curve direction by "
                f"{self.params['binning_along_axis0']} during data loading."
            )

            self.load_data(
                roi=CroppingHandler.roi_list_to_slices(roi),
                binning_along_axis0=self.params["binning_along_axis0"]
            )
            self.cropped_detector_data = self.detector_data

            # if user only specify position in 2D, extend it in 3D
            if len(det_ref) == 2:
                det_ref = (self.detector_data.shape[0] // 2, ) + tuple(det_ref)

            # check if shape dimensions are even
            checked_shape = tuple(
                s-1 if s % 2 == 1 else s for s in final_shape)
            if checked_shape != final_shape:
                self.verbose_print(
                    f"PyNX needs even input shapes, requested shape "
                    f"{final_shape} will be cropped to {checked_shape}."
                )
                final_shape = checked_shape
                roi = CroppingHandler.get_roi(final_shape, det_ref)
                self.cropped_detector_data = self.cropped_detector_data[
                    CroppingHandler.roi_list_to_slices(roi)
                ]
                self.verbose_print(f"New ROI is: {roi}.")

            # find the position in the cropped detector frame
            cropped_det_ref = tuple(
                    p - r if r else p  # if r is None, p-r must be p
                    for p, r in zip(det_ref, roi[::2])
            )
            full_det_max, full_det_com = None, None

            # we do not need a copy of self.cropped_detector_data
            self.detector_data = None

            # if the final_shape is 2D convert it in 3D
            if len(final_shape) == 2:
                final_shape = (
                    (self.cropped_detector_data.shape[0], )
                    + final_shape
                )

        else:
            # Check if binning is required
            if tuple(self.params["binning_factors"]) != (1, 1, 1):
                self.verbose_print(
                    "[BINNING] Binning requested "
                    f"(binning_factors = {self.params['binning_factors']})."
                )
                pixel_size_1 = (
                    self.params["det_calib_parameters"]["pwidth1"]
                    * self.params["binning_factors"][1]
                )
                pixel_size_2 = (
                    self.params["det_calib_parameters"]["pwidth2"]
                    * self.params["binning_factors"][2]
                )
                cch1 = (
                    self.params["det_calib_parameters"]["cch1"]
                    / self.params["binning_factors"][1]
                )
                cch2 = (
                    self.params["det_calib_parameters"]["cch2"]
                    / self.params["binning_factors"][2]
                )
                self.params["det_calib_parameters"].update(
                    {
                        "cch1": cch1,
                        "cch2": cch2,
                        "pwidth1": pixel_size_1,
                        "pwidth2": pixel_size_2,
                    }
                )
                self.load_data(
                    binning_along_axis0=self.params["binning_factors"][0]
                )
                initial_shape = self.detector_data.shape
                self.detector_data = rebin(
                    self.detector_data,
                    rebin_f=(1,) + tuple(self.params["binning_factors"][1:])
                )
                self.verbose_print(
                    "[BINNING] Data shape has been reduced from "
                    f"{initial_shape} to {self.detector_data.shape}.\n"
                    "Direct beam position has been updated to "
                    f"{cch1:.2f} (vertical) {cch2:.2f} (horizontal).\n"
                    "Pixel size has been updated to "
                    f"{pixel_size_1*1e6} x {pixel_size_2*1e6} um**2.\n"
                )
            else:
                self.load_data()
            # self.load_data()

            # if the final_shape is 2D convert it in 3D
            if len(final_shape) == 2:
                final_shape = (self.detector_data.shape[0], ) + final_shape

            # check if shape dimensions are even
            checked_shape = tuple(
                s-1 if s % 2 == 1 else s for s in final_shape)

            if checked_shape != final_shape:
                self.verbose_print(
                    f"[SHAPE & CROPPING] PyNX needs even input dimensions, "
                    f"requested shape {final_shape} will be cropped to "
                    f"{checked_shape}."
                )
                final_shape = checked_shape
            self.verbose_print(
                "[SHAPE & CROPPING] The preprocessing output shape is: "
                f"{final_shape} and will be used for ROI dimensions."
            )
            self.verbose_print(
                    "[SHAPE & CROPPING] Method(s) employed for the reference "
                    "voxel determination are "
                    f"{self.params['det_reference_voxel_method']}."
                )

            (
                self.cropped_detector_data,
                det_ref,
                cropped_det_ref,
                roi
            ) = CroppingHandler.chain_centering(
                self.detector_data,
                final_shape,
                methods=self.params["det_reference_voxel_method"]
            )
            # position of the max and com in the full detector frame
            full_det_max = CroppingHandler.get_position(
                self.detector_data, "max")
            full_det_com = CroppingHandler.get_position(
                self.detector_data, "com")

            # convert numpy.int64 to int to make them serializable and
            # store the det_reference_voxel in the parameters which will
            # be saved later
            self.params["det_reference_voxel"] = tuple(
                int(e) for e in det_ref
            )

            # center and crop the mask
            # self.mask = center(self.mask, where=det_ref)
            # self.mask = crop_at_center(self.mask, final_shape=final_shape)
            self.mask = self.mask[CroppingHandler.roi_list_to_slices(roi)]

            self.verbose_print(
                "[SHAPE & CROPPING] The reference voxel was found at "
                f"{det_ref} in the uncropped data frame\n"
                f"The processing_out_put_shape being {final_shape}, the roi "
                f"used to crop the data is {roi}.\n"
            )

            # set the q space area with the sample and detector angles that
            # correspond to the requested preprocessing_output_shape
            for key, value in self.angles.items():
                if isinstance(value, (list, np.ndarray)):
                    self.angles[key] = value[np.s_[roi[0]:roi[1]]]

        # position of the max and com in the cropped detector frame
        cropped_det_max = CroppingHandler.get_position(
            self.cropped_detector_data, "max")
        cropped_det_com = CroppingHandler.get_position(
            self.cropped_detector_data, "com")

        # self.init_space_converter(roi=roi[2:]) # we only need the 2D roi
        self.space_converter.init_q_space_area(
            roi=roi[2:],
            det_calib_parameters=self.params["det_calib_parameters"]
        )
        self.space_converter.set_q_space_area(**self.angles)

        table = [
            ["voxel", "uncroped det. pos.", "cropped det. pos.",
             "dspacing (A)", "lat. par. (A)"]
        ]

        # get the position of the reference, max and det voxels in the
        # q lab space
        self.verbose_print("Summary table:")
        self.verbose_print(
            "(max and com in the cropped frame are different to max and com in"
            " the uncropped detector frame.)"
        )
        for key, det_voxel, cropped_det_voxel in zip(
                ["q_lab_reference", "q_lab_max", "q_lab_com"],
                [det_ref, full_det_max, full_det_com],
                [cropped_det_ref, cropped_det_max, cropped_det_com]
        ):
            (
                self.params[key]
            ) = self.space_converter.index_det_to_q_lab(cropped_det_voxel)

            # compute the corresponding dpsacing and lattice parameter
            # for printing
            dspacing = self.space_converter.dspacing(
                    self.params[key]
            )
            lattice = self.space_converter.lattice_parameter(
                    self.params[key],
                    self.params["hkl"]
            )
            table.append(
                [key.split('_')[-1], det_voxel, cropped_det_voxel,
                 f"{dspacing:.5f}", f"{lattice:.5f}"]
            )

        self.verbose_print(
            tabulate(table, headers="firstrow", tablefmt="fancy_grid"),
            wrap=False
        )

        # Check if binning is required
        # if self.params["binning_factors"] != (1, 1, 1):
        #     self.verbose_print(
        #         "[INFO] Binning requested "
        #         f"(binning_factors = {self.params['binning_factors']})."
        #     )
        #     pixel_size_1 = (
        #         self.params["det_calib_parameters"]["pwidth1"]
        #         * self.params["binning_factors"][1]
        #     )
        #     pixel_size_2 = (
        #         self.params["det_calib_parameters"]["pwidth2"]
        #         * self.params["binning_factors"][2]
        #     )
        #     self.params["det_calib_parameters"].update(
        #         {
        #             "pwidth1": pixel_size_1,
        #             "pwidth2": pixel_size_2
        #         }
        #     )
        #     self.verbose_print(
        #         "Pixel size has been updated to "
        #         fr"{pixel_size_1*1e6} x {pixel_size_2*1e6} um**2."
        #     )

        #     self.cropped_detector_data = rebin(
        #         self.detector_data,
        #         rebin_f=self.params["binning_factors"]
        #     )
        #     self.space_converter.q_space_transitions =  rebin(
        #         self.space_converter.q_space_transitions,
        #         rebin_f=(1, )+self.params["binning_factors"],
        #         scale="average"
        #     )
        #     self.space_converter.cropped_shape = (
        #         self.cropped_detector_data.shape
        # )
        #     self.space_converter.full_shape = (
        #         self.cropped_detector_data.shape
        # )

        if self.params["orthogonalize_before_phasing"]:
            self.verbose_print(
                "[INFO] orthogonalization required before phasing.\n"
                "Will use xrayutilities Fuzzy Gridding without linear "
                "approximation."
            )
            self.orthogonalized_intensity = (
                self.space_converter.orthogonalize_to_q_lab(
                     self.cropped_detector_data,
                     method="xrayutilities"
                )
            )
            # we must orthogonalize the mask and orthogonalized_intensity must
            # be saved as the pynx input
            self.mask = self.space_converter.orthogonalize_to_q_lab(
                self.mask,
                method="xrayutilities"
            )
            self.cropped_detector_data = self.orthogonalized_intensity
            # Find where to plot slices in the reciprocal space
            where_in_ortho_space = (
                self.space_converter.index_det_to_index_of_q_lab(
                    cropped_det_ref,
                    interpolation_method="xrayutilities"
                )
            )
            q_lab_regular_grid = (
                self.space_converter.get_xu_q_lab_regular_grid()
            )
        else:
            self.verbose_print(
                "[INFO] Will linearize the transformation between detector and"
                " lab space."
            )
            # Initialise the interpolator so we won't need to reload raw
            # data during the post processing. The converter will be saved.
            self.space_converter.init_interpolator(
                self.cropped_detector_data,
                final_shape,
                space="both",
                direct_space_voxel_size=self.params["voxel_size"]
            )

            # Run the interpolation in the reciprocal space so we don't
            # do it later
            self.orthogonalized_intensity = (
                self.space_converter.orthogonalize_to_q_lab(
                    self.cropped_detector_data,
                    method="cdiutils"
                )
            )

            # Find where to plot slices in the reciprocal space
            where_in_ortho_space = (
                self.space_converter.index_det_to_index_of_q_lab(
                    cropped_det_ref
                )
            )
            q_lab_regular_grid = self.space_converter.get_q_lab_regular_grid()

        # Plot the reciprocal space data in the detector and lab frames
        self.figures["q_lab_orthogonalization"]["figure"] = (
            plot_q_lab_orthogonalization_process(
                self.cropped_detector_data.copy(),
                self.orthogonalized_intensity.copy(),
                q_lab_regular_grid,
                cropped_det_ref,
                where_in_ortho_space,
                title=(
                    r"From \textbf{detector frame} to \textbf{q lab frame}"
                    f", {self.sample_name}, {self.scan}"
                )
            )
        )

        # Update the preprocessing_output_shape and the det_reference_voxel
        self.params["preprocessing_output_shape"] = final_shape
        self.params["det_reference_voxel"] = det_ref

        # plot the detector data in the full detector frame and in the
        # final frame
        self.figures["preprocessing"]["figure"] = (
            preprocessing_detector_data_plot(
                cropped_data=self.cropped_detector_data.copy(),
                detector_data=(
                    None if self.detector_data is None
                    else self.detector_data.copy()
                ),
                det_reference_voxel=det_ref,
                det_max_voxel=full_det_max,
                det_com_voxel=full_det_com,
                cropped_max_voxel=cropped_det_max,
                cropped_com_voxel=cropped_det_com,
                title=(
                    "Detector data preprocessing, "
                    f"{self.sample_name}, {self.scan}"
                )
            )
        )

    def save_preprocessed_data(self):

        pynx_phasing_dir = self.dump_dir + "/pynx_phasing/"

        np.savez(
            f"{pynx_phasing_dir}S{self.scan}_pynx_input_data.npz",
            data=self.cropped_detector_data
        )
        np.savez(
            f"{pynx_phasing_dir}S{self.scan}_pynx_input_mask.npz",
            mask=self.mask
        )

        template_path = (
            f"{self.dump_dir}/cdiutils_S"
            f"{self.scan}"
        )
        self.figures["preprocessing"]["figure"].savefig(
            f"{template_path}_{self.figures['preprocessing']['name']}.png",
            bbox_inches="tight",
            dpi=200
        )

        if self.params["orthogonalize_before_phasing"]:
            regulard_grid_func = self.space_converter.get_xu_q_lab_regular_grid
        else:
            regulard_grid_func = self.space_converter.get_q_lab_regular_grid
            self.space_converter.save_interpolation_parameters(
                f"{template_path}_interpolation_parameters.npz"
            )

        np.savez(
            f"{template_path}_orthogonalized_intensity.npz",
            q_xlab=regulard_grid_func()[0],
            q_ylab=regulard_grid_func()[1],
            q_zlab=regulard_grid_func()[2],
            orthogonalized_intensity=self.orthogonalized_intensity
        )

        self.save_figures()

    def show_figures(self) -> None:
        """
        Show the figures that were plotted during the processing.
        """
        if any(value["figure"] for value in self.figures.values()):
            plt.show()

    def save_figures(self) -> None:
        """
        Save figures made during pre- or post-processing.
        """
        # create the debug directory
        if self.params["debug"]:
            debug_dir = f"{self.dump_dir}debug/"
            os.makedirs(
                debug_dir,
                exist_ok=True
            )
            if os.path.isdir(debug_dir):
                self.verbose_print(f"[INFO] Debug directory is:\n{debug_dir}")
            else:
                raise FileNotFoundError(
                    "Could not create the directory:\n{debug_dir}"
                )
        for fig in self.figures.values():
            if fig["figure"] is not None:
                fig_path = (
                    f"{debug_dir if fig['debug'] else self.dump_dir}"
                    f"cdiutils_S{self.scan}_"
                    f"{fig['name']}.png"
                )
                fig["figure"].savefig(
                    fig_path,
                    dpi=200,
                    bbox_inches="tight"
                )

    def orthogonalize(self):
        """
        Orthogonalize detector data to the lab frame.
        """
        reconstruction_file_path = (
            self.params["metadata"]["reconstruction_file"]
        )
        if not os.path.isabs(reconstruction_file_path):
            reconstruction_file_path = (
                self.dump_dir + "pynx_phasing/"
                + reconstruction_file_path
            )
        if not os.path.isfile(reconstruction_file_path):
            raise FileNotFoundError(
                f"File was not found at: {reconstruction_file_path}"
            )
        interpolation_file_path = (
            f"{self.dump_dir}cdiutils_S"
            f"{self.scan}_interpolation_parameters.npz"
        )
        if not os.path.isfile(interpolation_file_path):
            raise FileNotFoundError(
                f"File was not found at: {interpolation_file_path}"
            )
        reconstructed_object = self._load_reconstruction_file(
            reconstruction_file_path
        )
        # check if the reconstructed object is correctly centered
        # we need an isosurface for that, but we do not need
        # a precise value, it's just for the center of mass
        reconstructed_amplitude = np.abs(reconstructed_object)

        # Need to hard code the isosurface if it is < 0 or > 1
        isosurface = find_isosurface(reconstructed_amplitude)
        if isosurface < 0 or isosurface > 1:
            isosurface = 0.3
        support = make_support(
            reconstructed_amplitude,
            isosurface=isosurface
        )
        com = CroppingHandler.get_position(support, "com")
        reconstructed_object = CroppingHandler.force_centered_cropping(
            reconstructed_object,
            where=com,
        )
        support = CroppingHandler.force_centered_cropping(
            support,
            where=com,
        )

        # check if data were cropped during phase retrieval
        final_shape = tuple(self.params["preprocessing_output_shape"])
        if support.shape != final_shape:
            print(
                f"Shapes before {final_shape} "
                f"and after {support.shape} Phase Retrieval are different.\n"
                "Check out PyNX parameters (ex.: auto_center_resize)."
            )

        # Print the oversampling ratio
        ratio = oversampling_ratio(support)
        self.verbose_print(
            "[INFO] The oversampling ratios in each direction are "
            f"axis0: {ratio[0]:.1f}, axis1: {ratio[1]:.1f}, "
            f"axis2: {ratio[2]:.1f}"
        )

        self.space_converter.load_interpolation_parameters(
            interpolation_file_path,
            direct_space_voxel_size=self.params['voxel_size']
        )

        # might be useless
        for key in ["q_lab_reference", "q_lab_max", "q_lab_com"]:
            # compute the corresponding dspacing and lattice parameter
            self.params[f'dspacing_{key.split("_")[-1]}'] = (
                self.space_converter.dspacing(self.params[key])
            )

            self.params[f'lattice_parameter_{key.split("_")[-1]}'] = (
                self.space_converter.lattice_parameter(
                    self.params[key],
                    self.params["hkl"]
                )
            )

        # initialize the interpolator in reciprocal and direct spaces
        # (note that the orthogonalization is done in the lab frame
        # with xrayutilities conventions. We switch to cxi convention
        # afterwards).
        if self.params['voxel_size']:
            self.verbose_print(
                "[INFO] Voxel size in the direct lab frame provided by user: "
                f"{self.params['voxel_size']} nm"
            )
        if self.params["orthogonalize_before_phasing"]:
            self.orthogonalized_object = reconstructed_object
            if self.params['voxel_size']:
                self.verbose_print(
                    "[INFO] Orthogonalization was run before phasing, provided"
                    " voxel won't be used."
                )
        else:
            self.orthogonalized_object = (
                self.space_converter.orthogonalize_to_direct_lab(
                    reconstructed_object,
                )
            )

        if self.params["debug"]:
            self.figures["direct_lab_orthogonalization"]["figure"] = (
                plot_direct_lab_orthogonalization_process(
                    np.abs(self.orthogonalized_object),
                    self.space_converter.get_direct_lab_regular_grid(),
                    detector_direct_space_data=reconstructed_amplitude,
                    title=(
                        r"From \textbf{detector frame} to "
                        r"\textbf{direct lab frame}, "
                        f"S{self.scan}"
                    )
                )
            )
        self.orthogonalized_object = (
                self.space_converter.lab_to_cxi_conventions(
                    self.orthogonalized_object
                )
        )

        self.voxel_size = self.space_converter.lab_to_cxi_conventions(
            self.space_converter.direct_lab_interpolator.target_voxel_size
        )
        self.verbose_print(
            f"[INFO] Voxel size finally used is: {self.voxel_size} nm "
            "in the CXI convention"
        )

    def _load_reconstruction_file(self, file_path: str) -> np.ndarray:
        """
        Load a h5 reconstruction file and return only the
        entry_1/data_1/data entry.

        Args:
            file_path (str): the path to the file.

        Returns:
            np.ndarray: the data stored in entry_1/data_1/data.
        """
        with silx.io.h5py_utils.File(file_path) as h5file:
            reconstructed_object = h5file["entry_1/data_1/data"][0]
        return reconstructed_object

    def load_orthogonalized_object(self, file_path: str) -> None:
        """
        Load a reconstruction that has already been orthogonolized.
        This is used when bcdi backend is used, which is deprecated.

        Args:
            file_path (str): the path to the file.

        Raises:
            NotImplementedError: if the file is not a .npz file.
        """
        if file_path.endswith(".npz"):
            with np.load(file_path) as file:
                self.orthogonalized_object = file["data"]
                self.voxel_size = file["voxel_sizes"]
        else:
            raise NotImplementedError("Please provide a .npz file")

    def postprocess(self) -> None:
        """
        Run the postprocessing.

        Raises:
            TypeError: handle error during plotting.
        """
        if self.params["flip"]:
            self.orthogonalized_object = PostProcessor.flip_reconstruction(
                self.orthogonalized_object
            )

        if self.params["apodize"]:
            self.verbose_print(
                "[POST-PROCESSING] Apodizing the complex array."
            )
            self.orthogonalized_object = PostProcessor.apodize(
                self.orthogonalized_object
            )

        # first compute the histogram of the amplitude to get an
        # isosurface estimate
        self.verbose_print(
            "[POST-PROCESSING] Finding an isosurface estimate based on the "
            "reconstructed Bragg electron density histogram:",
            end=" "
        )
        isosurface, self.figures["amplitude"]["figure"] = find_isosurface(
            np.abs(self.orthogonalized_object),
            nbins=100,
            sigma_criterion=3,
            plot=True  # plot in any case
        )
        self.verbose_print(f"isosurface estimated at {isosurface}.")

        if self.params["isosurface"] is not None:
            self.verbose_print(
                "[INFO] Isosurface provided by user will be used: "
                f"{self.params['isosurface']}."
            )

        elif isosurface < 0 or isosurface > 1:
            self.params["isosurface"] = 0.3
            self.verbose_print(
                "[INFO] isosurface < 0 or > 1 is set to 0.3.")
        else:
            self.params["isosurface"] = isosurface

        # store the the averaged dspacing and lattice constant in
        # variables so they can be saved later in the output file
        self.verbose_print(
            "[INFO] The theoretical probed Bragg peak reflection is "
            f"{self.params['hkl']}."
        )

        if self.params["handle_defects"]:
            self.verbose_print(
                "[POST-PROCESSING] Defect handling requested."
            )
        self.verbose_print(
            "[POST-PROCESSING] Computing the structural properties:"
            "\n\t- phase \n\t- displacement"
            "\n\t- het. (heterogeneous) strain"
            "\n\t- d-spacing\n\t- lattice parameter."
            "\nhet. strain maps are computed using various methods, either"
            " phase ramp removal or d-spacing method.",
            wrap=False
        )
        self.structural_properties = (
                PostProcessor.get_structural_properties(
                    self.orthogonalized_object,
                    isosurface=self.params["isosurface"],
                    g_vector=SpaceConverter.lab_to_cxi_conventions(
                        self.params["q_lab_reference"]
                    ),
                    hkl=self.params["hkl"],
                    voxel_size=self.voxel_size,
                    phase_factor=-1,  # it came out pynx
                    handle_defects=self.params["handle_defects"],
                    parameters=self.params,
                )
        )

        self.averaged_dspacing = np.nanmean(
            self.structural_properties["dspacing"]
        )
        self.averaged_lattice_parameter = np.nanmean(
            self.structural_properties["lattice_parameter"]
        )

        # plot the results
        final_plots = {
            k: self.structural_properties[k]
            for k in ["amplitude", "phase", "displacement",
                      "het_strain", "lattice_parameter"]
        }
        try:
            self.figures["postprocessing"]["figure"] = summary_slice_plot(
                title=f"Summary figure, {self.sample_name}, S{self.scan}",
                support=zero_to_nan(self.structural_properties["support"]),
                dpi=200,
                voxel_size=self.voxel_size,
                isosurface=self.params["isosurface"],
                det_reference_voxel=self.params["det_reference_voxel"],
                averaged_dspacing=self.averaged_dspacing,
                averaged_lattice_parameter=self.averaged_lattice_parameter,
                **final_plots
            )
        except TypeError as exc:
            raise TypeError(
                "Something went wrong during plotting. "
                "Won't plot summary slice plot."
            ) from exc
        self.figures["3d_strain"]["figure"] = plot_3d_surface_projections(
            data=self.structural_properties["het_strain"],
            support=self.structural_properties["support"],
            voxel_size=self.voxel_size,
            cmap="cet_CET_D13",
            vmin=-np.nanmax(np.abs(self.structural_properties["het_strain"])),
            vmax=np.nanmax(np.abs(self.structural_properties["het_strain"])),
            cbar_title="Strain (\%)",
            title=f"3D views of the strain, {self.sample_name}, S{self.scan}"
        )

        strain_plots = {
            k: self.structural_properties[k]
            for k in [
                "het_strain", "het_strain_from_dspacing",
                "het_strain_from_dspacing", "numpy_het_strain",
                "het_strain_with_ramp"
            ]
        }
        self.figures["strain"]["figure"] = summary_slice_plot(
            title=f"Strain check figure, {self.sample_name}, S{self.scan}",
            support=zero_to_nan(self.structural_properties["support"]),
            dpi=200,
            voxel_size=self.voxel_size,
            isosurface=self.params["isosurface"],
            det_reference_voxel=self.params["det_reference_voxel"],
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_parameter=self.averaged_lattice_parameter,
            single_vmin=-self.structural_properties["het_strain"].ptp()/2,
            single_vmax=self.structural_properties["het_strain"].ptp()/2,
            **strain_plots
        )

        # take care of the axis names for the displacement gradient
        # plots
        axis_names = [
            r"z_{cxi}", r"y_{cxi}", r"x_{cxi}"
        ]
        if self.params["usetex"]:
            axis_title_template = (
                r"$\frac{\partial u_" + "{"
                + f"{''.join([str(e) for e in self.params['hkl']])}"
                + "}}"
            )
            titles = [
                axis_title_template + r"{\partial " + axis_names[i] + "}$"
                for i in range(3)
            ]
        else:
            axis_title_template = (
                "du_" + "{"
                + f"{''.join([str(e) for e in self.params['hkl']])}" + "}"
            )
            titles = [
                fr"${axis_title_template}/d{axis_names[i]}$"
                for i in range(3)
            ]

        displacement_gradient_plots = {
            titles[i]: (
                self.structural_properties["displacement_gradient"][i]
            )
            for i in range(3)
        }
        ptp_value = (
            np.nanmax(
                self.structural_properties["displacement_gradient"][0])
            - np.nanmin(
                self.structural_properties["displacement_gradient"][0])
        )
        self.figures["displacement_gradient"]["figure"] = (
            summary_slice_plot(
                title=(
                    "Displacement gradient, "
                    f"{self.sample_name}, S{self.scan}"
                ),
                support=zero_to_nan(self.structural_properties["support"]),
                dpi=200,
                voxel_size=self.voxel_size,
                isosurface=self.params["isosurface"],
                det_reference_voxel=self.params["det_reference_voxel"],
                averaged_dspacing=self.averaged_dspacing,
                averaged_lattice_parameter=(
                    self.averaged_lattice_parameter
                ),
                single_vmin=-ptp_value/2,
                single_vmax=ptp_value/2,
                cmap=RED_TO_TEAL,
                **displacement_gradient_plots
            )
        )

        # load the orthogonalized intensity computed during
        # preprocessing
        file_path = (
            f"{self.dump_dir}cdiutils_S{self.scan}"
            "_orthogonalized_intensity.npz"
        )
        with np.load(file_path) as npzfile:
            orthogonalized_intensity = npzfile["orthogonalized_intensity"]
            exp_data_q_lab_grid = [
                npzfile["q_xlab"],
                npzfile["q_ylab"],
                npzfile["q_zlab"]
            ]
        shape = orthogonalized_intensity.shape
        # convert to lab conventions and pad the data
        # We must multiply by -1 the phase to compare with the
        # measured intensity.
        final_object_fft = symmetric_pad(
            self.space_converter.cxi_to_lab_conventions(
                self.structural_properties["amplitude"]
                * np.exp(-1j*self.structural_properties["phase"])
            ),
            final_shape=shape
        )
        final_object_fft = np.abs(np.fft.ifftshift(
            np.fft.fftn(
                np.fft.fftshift(final_object_fft)
            )
        )) ** 2

        extension = np.multiply(self.voxel_size, shape)
        voxel_size_of_fft_object = 2 * np.pi / (10 * extension)

        final_object_q_lab_grid = (
            np.arange(
                -shape[0]//2, shape[0]//2, 1
            ) * voxel_size_of_fft_object[0],
            np.arange(
                -shape[1]//2, shape[1]//2, 1
            ) * voxel_size_of_fft_object[1],
            np.arange(
                -shape[2]//2, shape[2]//2, 1
            ) * voxel_size_of_fft_object[2],
        )
        final_object_q_lab_grid = tuple(
            shift + grid
            for shift, grid in zip(
                self.space_converter.q_space_shift,
                final_object_q_lab_grid
            )
        )

        # find the position in the cropped detector frame
        roi = CroppingHandler.get_roi(
            self.params["preprocessing_output_shape"],
            self.params["det_reference_voxel"]
        )
        cropped_det_ref = tuple(
                p - r if r else p  # if r is None, p-r must be p
                for p, r in zip(
                    self.params["det_reference_voxel"], roi[::2])
        )
        where_in_ortho_space = (
            self.space_converter.index_det_to_index_of_q_lab(
                cropped_det_ref
            )
        )

        self.figures["final_object_fft"]["figure"] = plot_final_object_fft(
            final_object_fft,
            orthogonalized_intensity,
            final_object_q_lab_grid,
            exp_data_q_lab_grid,
            where_in_ortho_space=where_in_ortho_space,
            title=(
                r"FFT of final object \textit{vs.} experimental data"
                f", {self.sample_name}, S{self.scan}"
            )
        )

    def save_postprocessed_data(self) -> None:
        """
        Save the postprocessed data, which correspond to the structural
        properties of the reconstructed crystal.
        """
        # save the results in a npz file
        template_path = (
            f"{self.dump_dir}/"
            f"cdiutils_S{self.scan}"
        )
        self.verbose_print(
            "[INFO] Saving the results to the following path:\n"
            f"{template_path}_structural_properties.npz"
        )

        np.savez(
            f"{template_path}_structural_properties.npz",
            q_lab_reference=self.params["q_lab_reference"],
            q_lab_max=self.params["q_lab_max"],
            q_lab_com=self.params["q_lab_com"],
            q_cxi_reference=SpaceConverter.lab_to_cxi_conventions(
                self.params["q_lab_reference"]
            ),
            q_cxi_max=SpaceConverter.lab_to_cxi_conventions(
                self.params["q_lab_max"]
            ),
            q_cxi_com=SpaceConverter.lab_to_cxi_conventions(
                self.params["q_lab_com"]
            ),
            dspacing_reference=self.params["dspacing_reference"],
            dspacing_max=self.params["dspacing_max"],
            dspacing_com=self.params["dspacing_com"],
            lattice_parameter_reference=(
                self.params["lattice_parameter_reference"]
            ),
            lattice_parameter_max=self.params["lattice_parameter_max"],
            lattice_parameter_com=self.params["lattice_parameter_com"],
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_parameter=self.averaged_lattice_parameter,
            processing_isosurface=self.params["isosurface"],
            **self.structural_properties,
        )

        to_save_as_vti = {
            k: self.structural_properties[k]
            for k in ["amplitude", "support", "phase", "displacement",
                      "het_strain", "het_strain_from_dspacing",
                      "lattice_parameter", "numpy_het_strain", "dspacing"]
        }

        # add the dspacing average and lattice constant average around 
        # the NP to avoid nan values that are annoying for 3D
        # visualisation
        to_save_as_vti["dspacing"] = np.where(
            np.isnan(to_save_as_vti["dspacing"]),
            self.averaged_dspacing,
            to_save_as_vti["dspacing"]
        )
        to_save_as_vti["lattice_parameter"] = np.where(
            np.isnan(to_save_as_vti["lattice_parameter"]),
            self.averaged_lattice_parameter,
            to_save_as_vti["lattice_parameter"]
        )

        # save to vti file
        self.save_to_vti(
            f"{template_path}_structural_properties.vti",
            voxel_size=self.voxel_size,
            cxi_convention=True,
            **to_save_as_vti
        )

        with h5py.File(f"{template_path}_structural_properties.h5", "w") as hf:
            volumes = hf.create_group("volumes")
            scalars = hf.create_group("scalars")

            volumes.create_dataset(
                "amplitude",
                data=self.structural_properties["amplitude"]
            )
            volumes.create_dataset(
                "support",
                data=self.structural_properties["support"]
            )
            volumes.create_dataset(
                "surface",
                data=self.structural_properties["surface"]
            )
            volumes.create_dataset(
                "phase",
                data=self.structural_properties["phase"]
            )
            volumes.create_dataset(
                "displacement",
                data=self.structural_properties["displacement"]
            )
            volumes.create_dataset(
                "het_strain",
                data=self.structural_properties["het_strain"]
            )
            volumes.create_dataset(
                "het_strain_from_dspacing",
                data=self.structural_properties["het_strain_from_dspacing"]
            )
            volumes.create_dataset(
                "numpy_het_strain",
                data=self.structural_properties["numpy_het_strain"]
            )
            volumes.create_dataset(
                "dspacing_from_het_strain",
                data=self.structural_properties["dspacing"]
            )
            volumes.create_dataset(
                "lattice_parameter",
                data=self.structural_properties["lattice_parameter"]
            )
            scalars.create_dataset(
                "q_lab_reference",
                data=self.params["q_lab_reference"]
            )
            scalars.create_dataset(
                "q_lab_max",
                data=self.params["q_lab_max"]
            )
            scalars.create_dataset(
                "q_lab_com",
                data=self.params["q_lab_com"]
            )
            scalars.create_dataset(
                "q_cxi_reference",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.params["q_lab_reference"]
                )
            )
            scalars.create_dataset(
                "q_cxi_max",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.params["q_lab_max"]
                )
            )
            scalars.create_dataset(
                "q_cxi_com",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.params["q_lab_com"]
                )
            )
            scalars.create_dataset(
                "dspacing_reference",
                data=self.params["dspacing_reference"]
            )
            scalars.create_dataset(
                "dspacing_max",
                data=self.params["dspacing_max"]
            )
            scalars.create_dataset(
                "dspacing_com",
                data=self.params["dspacing_com"]
            )
            scalars.create_dataset(
                "lattice_parameter_reference",
                data=self.params["lattice_parameter_reference"]
            )
            scalars.create_dataset(
                "lattice_parameter_max",
                data=self.params["lattice_parameter_max"]
            )
            scalars.create_dataset(
                "lattice_parameter_com",
                data=self.params["lattice_parameter_com"]
            )
            scalars.create_dataset(
                "averaged_dspacing",
                data=self.averaged_dspacing
            )
            scalars.create_dataset(
                "averaged_lattice_parameter",
                data=self.averaged_lattice_parameter
            )
            scalars.create_dataset(
                "voxel_size",
                data=self.voxel_size
            )
            scalars.create_dataset(
                "hkl",
                data=self.params["hkl"]
            )
            scalars.create_dataset(
                "processing_isosurface",
                data=self.params["isosurface"]
            )

        self.save_figures()

    @staticmethod
    def save_to_vti(
            output_path: str,
            voxel_size: tuple | list | np.ndarray,
            cxi_convention: bool = False,
            origin: tuple = (0, 0, 0),
            **np_arrays: dict[np.ndarray]
    ) -> None:
        """
        Save numpy arrays to .vti file.
        """
        voxel_size = tuple(voxel_size)
        nb_arrays = len(np_arrays)

        if not nb_arrays:
            raise ValueError(
                "np_arrays is empty, please provide a dictionary of "
                "(fieldnames: np.ndarray) you want to save."
            )
        is_init = False
        for i, (key, array) in enumerate(np_arrays.items()):
            if not is_init:
                shape = array.shape
                if cxi_convention:
                    voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])
                    shape = (shape[2], shape[1], shape[0])
                image_data = vtk.vtkImageData()
                image_data.SetOrigin(origin)
                image_data.SetSpacing(voxel_size)
                image_data.SetExtent(
                    0, shape[0] - 1,
                    0, shape[1] - 1,
                    0, shape[2] - 1
                )
                point_data = image_data.GetPointData()
                is_init = True

            vtk_array = numpy_support.numpy_to_vtk(array.ravel())
            point_data.AddArray(vtk_array)
            point_data.GetArray(i).SetName(key)
            point_data.Update()

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(image_data)
        writer.Write()


