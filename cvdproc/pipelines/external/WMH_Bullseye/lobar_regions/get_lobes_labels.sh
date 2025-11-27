#!bin/bash

labdir=$1

frontal=(superiorfrontal rostralmiddlefrontal caudalmiddlefrontal parsopercularis parsorbitalis parstriangularis lateralorbitofrontal medialorbitofrontal precentral paracentral insula frontalpole rostralanteriorcingulate caudalanteriorcingulate)
patietal=(superiorparietal inferiorparietal supramarginal postcentral precuneus posteriorcingulate isthmuscingulate)
temporal=(superiortemporal middletemporal inferiortemporal bankssts fusiform transversetemporal entorhinal temporalpole parahippocampal)
occipital=(lateraloccipital lingual cuneus pericalcarine)
hemis=(lh rh)

for hemi in "${hemis[@]}"; do
  mri_mergelabels -i $labdir/$hemi.superiorfrontal.label -i $labdir/$hemi.rostralmiddlefrontal.label -i $labdir/$hemi.caudalmiddlefrontal.label -i $labdir/$hemi.parsopercularis.label -i $labdir/$hemi.parsorbitalis.label -i $labdir/$hemi.parstriangularis.label -i $labdir/$hemi.lateralorbitofrontal.label -i $labdir/$hemi.medialorbitofrontal.label -i $labdir/$hemi.precentral.label -i $labdir/$hemi.paracentral.label -i $labdir/$hemi.insula.label -i $labdir/$hemi.frontalpole.label -i $labdir/$hemi.rostralanteriorcingulate.label -i $labdir/$hemi.caudalanteriorcingulate.label -o $labdir/$hemi.frontal_lobe.label 
done

for hemi in "${hemis[@]}"; do
  mri_mergelabels -i $labdir/$hemi.superiorparietal.label -i $labdir/$hemi.inferiorparietal.label -i $labdir/$hemi.supramarginal.label -i $labdir/$hemi.postcentral.label -i $labdir/$hemi.precuneus.label -i $labdir/$hemi.posteriorcingulate.label -i $labdir/$hemi.isthmuscingulate.label -o $labdir/$hemi.parietal_lobe.label 
done

for hemi in "${hemis[@]}"; do
  mri_mergelabels -i $labdir/$hemi.superiortemporal.label -i $labdir/$hemi.middletemporal.label -i $labdir/$hemi.inferiortemporal.label -i $labdir/$hemi.bankssts.label -i $labdir/$hemi.fusiform.label -i $labdir/$hemi.transversetemporal.label -i $labdir/$hemi.entorhinal.label -i $labdir/$hemi.temporalpole.label -i $labdir/$hemi.parahippocampal.label -o $labdir/$hemi.temporal_lobe.label 
done

for hemi in "${hemis[@]}"; do
  mri_mergelabels -i $labdir/$hemi.lateraloccipital.label -i $labdir/$hemi.lingual.label -i $labdir/$hemi.cuneus.label -i $labdir/$hemi.pericalcarine.label -o $labdir/$hemi.occipital_lobe.label 
done

