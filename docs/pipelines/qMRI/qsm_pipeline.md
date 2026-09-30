# QSM Pipeline

::: cvdproc.pipelines.qmri.qsm_pipeline.QSMPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline computes Quantitative Susceptibility Mapping (QSM) and associated scalar maps from multi-echo gradient echo (GRE) data.

### Processing Steps

#### 1. QSM Reconstruction (MATLAB-based, `qsm_pipeline_part1`)

The first stage runs a MATLAB script (`QSM_pipeline_part1.m`) that performs QSM reconstruction using a combined approach from the [SEPIA](https://github.com/kschan0214/sepia) toolbox [@chan2021sepia] and the Chisep toolbox:

- **Phase unwrapping** — ROMEO (Rapid Opensource Minimum spanning trEe algOrithm) [@dymerska2021romeo] unwraps the multi-echo phase images for total field calculation.
- **Background field removal** — V-SHARP (Variable-kernel Sophisticated Harmonic Artifact Reduction for Phase data) [@schweser2011sharp] removes background field contributions to isolate the local tissue field.
- **Dipole inversion** — iLSQR [@li2015ilsqr] performs the dipole inversion to compute the final susceptibility map from the local field.
- **Source separation** — Chi-separation decomposes the QSM signal into diamagnetic (chi\_dia, primarily myelin) and paramagnetic (chi\_para, primarily iron) components.

This stage outputs the following scalar maps:

| Map | Description |
|-----|-------------|
| `Chimap` | Quantitative susceptibility map (QSM) |
| `R2starmap` | R₂\* relaxation rate map |
| `S0map` | Signal intensity at TE=0 |
| `T2starmap` | T₂\* relaxation time map |
| `Chidia` | Diamagnetic susceptibility component (myelin-related) |
| `Chipara` | Paramagnetic susceptibility component (iron-related) |
| `Chitotal` | Total susceptibility (chi\_dia + chi\_para) |

#### 2. OEF Estimation (QQ-Net)

The second stage uses [QQ-Net](https://github.com/junghun87/QQNET) [@cho2018qqnet], a deep learning-based method for estimating the oxygen extraction fraction (OEF) from the combined QSM and quantitative BOLD (qBOLD) signals:

- Inputs: multi-echo magnitude (4D), QSM map, R₂\* map, S₀ map, and brain mask from stage 1.
- Output: OEF map in native QSM space.

#### 3. Spatial Normalization (optional, `normalize=True`)

If enabled, the QSM scalar maps are registered to T1w space (via 6-DOF FLIRT using the first-echo magnitude image) and subsequently to MNI space (MNI152NLin6Asym) using SynthMorph non-linear registration. Outputs include all scalar maps in both T1w and MNI spaces.

### Modalities

- GRE (multi-echo gradient echo, required — auto-discovered via `qsm_files`)
- T1w (required for registration)

### Configuration Example

```yaml
pipelines:
    qsm_pipeline:
        use_which_t1w: "acq-highres"
        normalize: true
        phase_image_correction: false
        reverse_phase: 0
```

### Parameters

- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename.
- `normalize`: If `True`, register QSM scalar maps to T1w and MNI152NLin6Asym space (default: `False`).
- `phase_image_correction`: If `True`, correct inter-slice phase polarity differences in GE data (see [SEPIA discussion](https://github.com/kschan0214/sepia/discussions/93)).
- `reverse_phase`: Set to `1` to invert phase polarity for GE scanners (default: `0`).

### Notes

- The MATLAB script `QSM_pipeline_part1.m` requires MATLAB and the SEPIA toolbox to be installed and on the MATLAB path.
- QQ-Net requires PyTorch and the trained model weights (downloaded automatically if not present).
- The SEPIA QSM pipeline ([sepia_qsm](sepia_qsm.md)) is the predecessor of this pipeline and is now deprecated.

### References

\bibliography
