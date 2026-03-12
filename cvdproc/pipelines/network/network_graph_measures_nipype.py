import os
import numpy as np
import pandas as pd

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
)

from cvdproc.pipelines.network.network_measures import (
    load_connectivity_matrix,
    compute_global_efficiency,
    compute_local_efficiency,
    compute_nodal_strength,
    compute_mean_strength,
    compute_nodal_clustering_coefficient,
    compute_mean_clustering_coefficient,
    compute_characteristic_path_length,
    compute_nodal_betweenness_centrality,
    compute_mean_betweenness_centrality,
)


class LoadConnectivityMatrixInputSpec(BaseInterfaceInputSpec):
    csv_file = File(
        exists=True,
        mandatory=True,
        desc="Path to connectivity CSV file",
    )

    header = traits.Any(
        None,
        usedefault=True,
        desc="Header argument passed to pandas.read_csv",
    )

    index_col = traits.Any(
        None,
        usedefault=True,
        desc="Index column argument passed to pandas.read_csv",
    )

    force_square = traits.Bool(
        True,
        usedefault=True,
        desc="Whether to require a square matrix",
    )

    force_symmetric = traits.Bool(
        True,
        usedefault=True,
        desc="Whether to force symmetry by averaging A and A.T",
    )

    zero_diagonal = traits.Bool(
        True,
        usedefault=True,
        desc="Whether to set diagonal entries to zero",
    )

    remove_negative = traits.Bool(
        True,
        usedefault=True,
        desc="Whether to set negative weights to zero",
    )

    output_npy = File(
        desc="Optional output .npy file for the loaded matrix",
    )


class LoadConnectivityMatrixOutputSpec(TraitedSpec):
    matrix_file = File(
        exists=True,
        desc="Saved connectivity matrix in .npy format",
    )

    n_nodes = traits.Int(
        desc="Number of nodes in the connectivity matrix",
    )


class LoadConnectivityMatrix(BaseInterface):
    input_spec = LoadConnectivityMatrixInputSpec
    output_spec = LoadConnectivityMatrixOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self._results = {}

    def _run_interface(self, runtime):
        matrix = load_connectivity_matrix(
            csv_file=self.inputs.csv_file,
            header=self.inputs.header,
            index_col=self.inputs.index_col,
            force_square=self.inputs.force_square,
            force_symmetric=self.inputs.force_symmetric,
            zero_diagonal=self.inputs.zero_diagonal,
            remove_negative=self.inputs.remove_negative,
        )

        if self.inputs.output_npy:
            matrix_file = os.path.abspath(self.inputs.output_npy)
        else:
            csv_base = os.path.splitext(os.path.basename(self.inputs.csv_file))[0]
            matrix_file = os.path.abspath(f"{csv_base}_matrix.npy")

        np.save(matrix_file, matrix)

        self._results["matrix_file"] = matrix_file
        self._results["n_nodes"] = int(matrix.shape[0])

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["matrix_file"] = self._results.get("matrix_file")
        outputs["n_nodes"] = self._results.get("n_nodes")
        return outputs


class ComputeGraphMeasuresInputSpec(BaseInterfaceInputSpec):
    matrix_file = File(
        exists=True,
        mandatory=True,
        desc="Input connectivity matrix in .npy format",
    )

    global_metrics_csv = File(
        desc="Optional output CSV file for global graph metrics",
    )

    nodal_metrics_csv = File(
        desc="Optional output CSV file for nodal graph metrics",
    )


class ComputeGraphMeasuresOutputSpec(TraitedSpec):
    global_efficiency = traits.Float(desc="Global efficiency")
    mean_local_efficiency = traits.Float(desc="Mean local efficiency")
    mean_strength = traits.Float(desc="Mean nodal strength")
    mean_clustering_coefficient = traits.Float(desc="Mean clustering coefficient")
    characteristic_path_length = traits.Float(desc="Characteristic path length")
    mean_betweenness_centrality = traits.Float(desc="Mean betweenness centrality")

    global_metrics_csv = File(
        exists=True,
        desc="CSV file containing global graph metrics",
    )

    nodal_metrics_csv = File(
        exists=True,
        desc="CSV file containing nodal graph metrics",
    )


class ComputeGraphMeasures(BaseInterface):
    input_spec = ComputeGraphMeasuresInputSpec
    output_spec = ComputeGraphMeasuresOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self._results = {}

    def _run_interface(self, runtime):
        matrix = np.load(self.inputs.matrix_file)
        matrix = np.asarray(matrix, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Input matrix must be square, but got shape {matrix.shape}"
            )

        global_efficiency = compute_global_efficiency(matrix)
        mean_local_efficiency = compute_local_efficiency(
            matrix, return_nodal=False
        )
        nodal_local_efficiency = compute_local_efficiency(
            matrix, return_nodal=True
        )

        mean_strength = compute_mean_strength(matrix)
        nodal_strength = compute_nodal_strength(matrix)

        mean_clustering_coefficient = compute_mean_clustering_coefficient(matrix)
        nodal_clustering_coefficient = compute_nodal_clustering_coefficient(matrix)

        characteristic_path_length = compute_characteristic_path_length(matrix)

        mean_betweenness_centrality = compute_mean_betweenness_centrality(matrix)
        nodal_betweenness_centrality = compute_nodal_betweenness_centrality(matrix)

        if self.inputs.global_metrics_csv:
            global_metrics_csv = os.path.abspath(self.inputs.global_metrics_csv)
        else:
            global_metrics_csv = os.path.abspath("global_metrics.csv")

        if self.inputs.nodal_metrics_csv:
            nodal_metrics_csv = os.path.abspath(self.inputs.nodal_metrics_csv)
        else:
            nodal_metrics_csv = os.path.abspath("nodal_metrics.csv")

        global_df = pd.DataFrame(
            [
                {
                    "global_efficiency": float(global_efficiency),
                    "mean_local_efficiency": float(mean_local_efficiency),
                    "mean_strength": float(mean_strength),
                    "mean_clustering_coefficient": float(
                        mean_clustering_coefficient
                    ),
                    "characteristic_path_length": float(
                        characteristic_path_length
                    ),
                    "mean_betweenness_centrality": float(
                        mean_betweenness_centrality
                    ),
                }
            ]
        )
        global_df.to_csv(global_metrics_csv, index=False)

        nodal_df = pd.DataFrame(
            {
                "node_index": np.arange(matrix.shape[0], dtype=int),
                "nodal_local_efficiency": np.asarray(
                    nodal_local_efficiency, dtype=np.float64
                ),
                "nodal_strength": np.asarray(
                    nodal_strength, dtype=np.float64
                ),
                "nodal_clustering_coefficient": np.asarray(
                    nodal_clustering_coefficient, dtype=np.float64
                ),
                "nodal_betweenness_centrality": np.asarray(
                    nodal_betweenness_centrality, dtype=np.float64
                ),
            }
        )
        nodal_df.to_csv(nodal_metrics_csv, index=False)

        self._results["global_efficiency"] = float(global_efficiency)
        self._results["mean_local_efficiency"] = float(mean_local_efficiency)
        self._results["mean_strength"] = float(mean_strength)
        self._results["mean_clustering_coefficient"] = float(
            mean_clustering_coefficient
        )
        self._results["characteristic_path_length"] = float(
            characteristic_path_length
        )
        self._results["mean_betweenness_centrality"] = float(
            mean_betweenness_centrality
        )
        self._results["global_metrics_csv"] = global_metrics_csv
        self._results["nodal_metrics_csv"] = nodal_metrics_csv

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["global_efficiency"] = self._results.get("global_efficiency")
        outputs["mean_local_efficiency"] = self._results.get("mean_local_efficiency")
        outputs["mean_strength"] = self._results.get("mean_strength")
        outputs["mean_clustering_coefficient"] = self._results.get(
            "mean_clustering_coefficient"
        )
        outputs["characteristic_path_length"] = self._results.get(
            "characteristic_path_length"
        )
        outputs["mean_betweenness_centrality"] = self._results.get(
            "mean_betweenness_centrality"
        )
        outputs["global_metrics_csv"] = self._results.get("global_metrics_csv")
        outputs["nodal_metrics_csv"] = self._results.get("nodal_metrics_csv")
        return outputs


if __name__ == "__main__":
    csv_file = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0008/ses-baseline/connectome/sub-SSI0008_ses-baseline_acq-DTIb1000_space-preprocdwi_seg-aparc_connectivity.csv"

    print("Running LoadConnectivityMatrix interface...")

    load_node = LoadConnectivityMatrix()
    load_node.inputs.csv_file = csv_file
    load_node.inputs.header = None
    load_node.inputs.index_col = None
    load_node.inputs.force_square = True
    load_node.inputs.force_symmetric = True
    load_node.inputs.zero_diagonal = True
    load_node.inputs.remove_negative = True

    load_res = load_node.run()

    matrix_file = load_res.outputs.matrix_file
    n_nodes = load_res.outputs.n_nodes

    print("Matrix file:", matrix_file)
    print("Number of nodes:", n_nodes)

    print("\nRunning ComputeGraphMeasures interface...")

    measure_node = ComputeGraphMeasures()
    measure_node.inputs.matrix_file = matrix_file

    measure_res = measure_node.run()

    print("\nGlobal graph metrics:")
    print("Global efficiency:", measure_res.outputs.global_efficiency)
    print("Mean local efficiency:", measure_res.outputs.mean_local_efficiency)
    print("Mean strength:", measure_res.outputs.mean_strength)
    print("Mean clustering coefficient:", measure_res.outputs.mean_clustering_coefficient)
    print("Characteristic path length:", measure_res.outputs.characteristic_path_length)
    print("Mean betweenness centrality:", measure_res.outputs.mean_betweenness_centrality)

    print("\nOutput files:")
    print("Global metrics CSV:", measure_res.outputs.global_metrics_csv)
    print("Nodal metrics CSV:", measure_res.outputs.nodal_metrics_csv)

    print("\nTest completed successfully.")