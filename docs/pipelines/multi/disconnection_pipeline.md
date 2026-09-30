# Disconnection Pipeline

::: cvdproc.pipelines.multi.disconnection_pipeline.DisconnectionPipeline
    options:
      show_signature: false

----

## Structural damage models

The structural pipeline supports normative IIT and individual tractograms. Each model uses the same fixed denominators derived from all streamlines and produces traversal, endpoint, regional, network, surface, and QC outputs.

- `any_hit` assigns a streamline a weight of one after its first lesion intersection and zero otherwise. This saturation makes the model nonlinear in the lesion image.
- `absolute_length` assigns the lesion-intersecting length in millimetres. The resulting map is the mean affected length among all streamlines crossing each output location and is strictly linear in the lesion image.
- `fractional_length` divides each streamline's affected cached length by its total cached length before projection. The resulting dimensionless map is the mean affected streamline proportion among all streamlines crossing each output location. Unlike `any_hit`, it preserves the amount of streamline involvement and remains strictly linear in the lesion image.

For the IIT cache, let `A` be the streamline-by-lesion-voxel length operator, `P` the streamline-to-output-voxel projection, `ell = A 1`, and `Z(n)` the reciprocal fixed output denominator. The fractional operator is

```text
W_frac = Z(n) P^T diag(ell^-1) A,
D = W_frac L.
```

Zero-length streamlines receive a fractional weight of zero. Lesion values for `fractional_length` must lie in `[0, 1]` so the result retains its proportion interpretation.

## Configuration example

```yaml
pipelines:
    disconnection:
        methods: [normative, individual]
        structural_methods: [any_hit, absolute_length, fractional_length]
        mni_lesion_mask: lesion_mask
        use_which_mni_lesion_mask: infarction
        t1w_lesion_mask: lesion_mask
        use_which_t1w_lesion_mask: infarction
        individual_connectome_source: mrtrix3
```
