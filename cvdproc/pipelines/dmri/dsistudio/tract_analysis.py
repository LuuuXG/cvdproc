import os
import glob
import shutil

from nipype.interfaces.base import (
    File,
    Str,
    CommandLine,
    CommandLineInputSpec,
    TraitedSpec,
    traits,
)

from cvdproc.config.paths import get_package_path

dsi_studio_path = get_package_path(
    "data", "lqt", "extdata", "DSI_studio", "dsi-studio", "dsi_studio"
)


class TractStatsInputSpec(CommandLineInputSpec):
    source = File(
        exists=True,
        desc="Input source file (.fib.gz or .fz).",
        mandatory=True,
        argstr="--source=%s",
    )
    tract = File(
        exists=True,
        desc="Input tract file (.tt.gz).",
        mandatory=True,
        argstr="--tract=%s",
    )
    export = Str(
        desc="Export format. Typically 'stat'.",
        mandatory=True,
        argstr="--export=%s",
    )

    # Not passed to command line
    output_txt = Str(
        desc="Desired output text file path for tract statistics (renamed from DSI Studio default).",
        mandatory=True,
        usedefault=False,
    )


class TractStatsOutputSpec(TraitedSpec):
    stats = File(desc="Output text file for tract statistics.", exists=True)


class TractStatsInterface(CommandLine):
    input_spec = TractStatsInputSpec
    output_spec = TractStatsOutputSpec

    _cmd = f"{dsi_studio_path} --action=ana"

    def _ensure_outdir(self):
        out_txt = os.path.abspath(self.inputs.output_txt)
        out_dir = os.path.dirname(out_txt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return out_txt

    @staticmethod
    def _candidate_default_outputs(tract_path):
        """
        DSI Studio versions may produce slightly different default names.
        We generate a list of plausible candidates.
        """
        tract_abs = os.path.abspath(tract_path)
        tract_dir = os.path.dirname(tract_abs)
        tract_base = os.path.basename(tract_abs)

        # Common: <tract>.stat.txt
        cands = [
            tract_abs + ".stat.txt",
            os.path.join(tract_dir, tract_base + ".stat.txt"),
        ]

        # Sometimes: remove extensions before appending
        base_no_gz = tract_base[:-3] if tract_base.endswith(".gz") else tract_base
        base_root, _ = os.path.splitext(base_no_gz)  # removes .tt or similar
        cands += [
            os.path.join(tract_dir, base_root + ".stat.txt"),
            os.path.join(tract_dir, base_no_gz + ".stat.txt"),
        ]

        # Sometimes: export name in different casing or separators
        cands += [
            tract_abs + ".STAT.txt",
            os.path.join(tract_dir, tract_base + ".STAT.txt"),
        ]

        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for p in cands:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    @staticmethod
    def _find_stat_txt(tract_path):
        tract_abs = os.path.abspath(tract_path)
        tract_dir = os.path.dirname(tract_abs)

        # First, try known candidates
        for p in TractStatsInterface._candidate_default_outputs(tract_abs):
            if os.path.isfile(p):
                return p

        # Fallback: glob search in the tract directory
        tract_base = os.path.basename(tract_abs)
        patterns = [
            os.path.join(tract_dir, tract_base + "*.stat*.txt"),
            os.path.join(tract_dir, "*stat*.txt"),
        ]
        matches = []
        for pat in patterns:
            matches.extend(glob.glob(pat))

        matches = [m for m in matches if os.path.isfile(m)]
        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            # Prefer files that contain tract base name
            prefer = [m for m in matches if tract_base in os.path.basename(m)]
            if len(prefer) == 1:
                return prefer[0]
            # Otherwise pick the most recently modified file
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return matches[0]

        return None

    def _post_run_hook(self, runtime):
        """
        After DSI Studio finishes, locate the produced stat file and rename/move to output_txt.
        """
        out_txt = self._ensure_outdir()

        produced = self._find_stat_txt(self.inputs.tract)
        if produced is None:
            raise RuntimeError(
                "DSI Studio finished but no stat text file was found. "
                "Please verify '--export=stat' behavior for your DSI Studio build."
            )

        produced = os.path.abspath(produced)

        # If produced path is already the requested path, just keep it
        if os.path.abspath(out_txt) != produced:
            shutil.move(produced, out_txt)

        # Sanity check
        if not os.path.isfile(out_txt):
            raise RuntimeError(f"Failed to create output stat file: {out_txt}")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["stats"] = os.path.abspath(self.inputs.output_txt)
        return outputs
    
# Example usage:
if __name__ == "__main__":
    from nipype import Node
    n = Node(TractStatsInterface(), name="tract_stats")
    n.inputs.source = "/mnt/f/BIDS/WCH_AF_Project/derivatives/qsirecon-DSIStudio/sub-AFib0241/ses-baseline/dwi/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-gqi_dwimap.fib.gz"
    n.inputs.tract = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_desc-voxelspace_streamlines.tt.gz"
    n.inputs.export = "stat"
    n.inputs.output_txt = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_desc-voxelspace_streamlines_stat.txt"
    res = n.run()
    print(res.outputs.stats)