# Publications

## 1. Choroidal Vascular Volume, White Matter Hyperintensity, and Their Interaction With Cognitive Function in Aging Adults

[Choroidal Vascular Volume, White Matter Hyperintensity, and Their Interaction With Cognitive Function in Aging Adults](https://www.ahajournals.org/doi/10.1161/JAHA.124.039369)

The idea of this study was inspired by SMART-MR series studies (e.g., [Keller et al.](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13345), [Ghaznawi et al.](https://www.neurology.org/doi/10.1212/WNL.0000000000011827)). Their work found that shape features of WMH better reflect cognitive decline than traditional volume metrics. We wondered whether the microvascular condition of the eye (assessed by OCTA imaging) is also associated with the shape features of WMH and reflects cognitive function.

The code is now implemented in the [wmh_quantification](./pipelines/sMRI/wmh_quantification.md) pipeline. For WMH segmention, we used the LST-LPA method with FLAIR images. For shape features, the source code is in [wmh_shape_nipype.py](../cvdproc/pipelines/smri/csvd_quantification/wmh/wmh_shape_nipype.py).