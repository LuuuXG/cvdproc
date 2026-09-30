# T1w-to-MNI Registration

::: cvdproc.pipelines.smri.t1_register

-----

## Description

[Synthmorph](https://martinos.org/malte/synthmorph/) [@hoffmann2022synthmorph] is used for fast and robust T1w-to-MNI registration (compared to traditional FNIRT or ANTs methods).

## Parameters

### `use_which_t1w` (str, optional)

Keyword to select a specific T1w image when multiple are available. Default: first available.

### `template_space` (str)

Target MNI template space. Supported values:

| Space | Description |
|-------|-------------|
| `MNI152NLin6Asym` | Adult MNI ICBM 152 nonlinear asymmetric (default) |
| `MNI152NLin2009cAsym` | Adult MNI ICBM 152 2009c asymmetric |
| `MNIPediatricAsym` | Pediatric MNI asymmetric (requires `cohort`) |

Default: `"MNI152NLin6Asym"`.

### `cohort` (str)

Pediatric cohort number. Only used when `template_space = "MNIPediatricAsym"`. Supported values:

| Value | Template file |
|-------|---------------|
| `"1"` | `tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz` |

Default: `""`.

### `resolution` (float)

Template resolution in mm. Available options depend on the template space:

| Space | Supported resolutions |
|-------|-----------------------|
| `MNI152NLin6Asym` | `1`, `0.5` |
| `MNI152NLin2009cAsym` | `1` |
| `MNIPediatricAsym` | `1` |

Default: `1`.

## Example config

```yaml
# Adult MNI (default)
t1_register:
  template_space: "MNI152NLin6Asym"
  resolution: 1

# Pediatric MNI, cohort 1
t1_register:
  template_space: "MNIPediatricAsym"
  cohort: "1"
  resolution: 1

# MNI 2009c
t1_register:
  template_space: "MNI152NLin2009cAsym"
  resolution: 1
```

## Output files

Output filenames use the configured `template_space` value (e.g., `MNI152NLin6Asym`, `MNI152NLin2009cAsym`, `MNIPediatricAsym`).

| File | Description |
|------|-------------|
| `*_space-{template_space}_T1w.nii.gz` | T1w in MNI space |
| `*_from-T1w_to-{template_space}_warp.nii.gz` | Forward warp field |
| `*_from-{template_space}_to-T1w_warp.nii.gz` | Inverse warp field |
| `*_space-T1w_desc-brain_T1w.nii.gz` | Skull-stripped T1w |
| `*_space-T1w_desc-brain_mask.nii.gz` | Brain mask |

### References

\bibliography
