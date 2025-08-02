# pvs_quantification

Current SHiVAi pipeline (2025-06) seems to have problem handling html/pdf reports (for example, when facing NaN value). A workaround is to modify the code in `post.py` to simply skip the report generation step, which is not critical for the pipeline to run.

```python
class SummaryReport(BaseInterface):
    """Make a summary report of preprocessing and prediction"""
    input_spec = SummaryReportInputSpec
    output_spec = SummaryReportOutputSpec

    # def _run_interface(self, runtime):
    #     """
    #     Build the report for the whole workflow. It contains segmentation statistics and
    #     quality control figures.

    #     """
    #     if self.inputs.anonymized:
    #         subject_id = ''
    #     else:
    #         subject_id = self.inputs.subject_id

    #     brain_vol_vox = nib.load(self.inputs.brainmask).get_fdata().astype(bool).sum()  # in voxels
    #     pred_metrics_dict = {}  # Will contain the stats dataframe for each biomarker
    #     pred_census_im_dict = {}  # Will contain the path to the swarm plot for each biomarker
    #     pred_overlay_im_dict = {}  # Will contain the path to the figure with biomarkers overlaid on the brain
    #     models_uid = {}  # Will contain the md5 hash for each file of each predictive model
    #     pred_and_acq = self.inputs.pred_and_acq

    #     # Generate the distribution figures for each prediction and fill models_uid
    #     for pred in pred_and_acq:
    #         lpred = pred.lower()  # "pred" is uppercase, so we also need a lowercase version
    #         if pred == 'LAC':
    #             name_in_plot = 'Lacuna'
    #         else:
    #             name_in_plot = pred
    #         models_uid[pred] = {}
    #         pred_metrics_dict[pred] = pd.read_csv(getattr(self.inputs, f'{lpred}_metrics_csv'))
    #         if pred_metrics_dict[pred]['Number of clusters'].sum() == 0:  # No biomarker detected
    #             pred_census_im_dict[pred] = None
    #         else:
    #             pred_census_im_dict[pred] = violinplot_from_census(getattr(self.inputs, f'{lpred}_census_csv'),
    #                                                                self.inputs.resolution,
    #                                                                name_in_plot)
    #         pred_overlay_im_dict[pred] = getattr(self.inputs, f'{lpred}_overlay')
    #         ids, url = get_md5_from_json(getattr(self.inputs, f'{lpred}_model_descriptor'), get_url=True)
    #         models_uid[pred]['id'] = ids
    #         if url:
    #             models_uid[pred]['url'] = url

    #     # set optional inputs to None if undefined
    #     if isdefined(self.inputs.overlayed_brainmask_1):
    #         overlayed_brainmask_1 = self.inputs.overlayed_brainmask_1
    #     else:
    #         overlayed_brainmask_1 = None
    #     if isdefined(self.inputs.crop_brain_img):
    #         crop_brain_img = self.inputs.crop_brain_img
    #     else:
    #         crop_brain_img = None
    #     if isdefined(self.inputs.isocontour_slides_FLAIR_T1):
    #         isocontour_slides_FLAIR_T1 = self.inputs.isocontour_slides_FLAIR_T1
    #     else:
    #         isocontour_slides_FLAIR_T1 = None
    #     if isdefined(self.inputs.overlayed_brainmask_2):
    #         overlayed_brainmask_2 = self.inputs.overlayed_brainmask_2
    #     else:
    #         overlayed_brainmask_2 = None
    #     if isdefined(self.inputs.wf_graph):
    #         wf_graph = self.inputs.wf_graph
    #     else:
    #         wf_graph = None
    #     if isdefined(self.inputs.db):
    #         db = self.inputs.db
    #     else:
    #         db = ''
    #     # process
    #     html_report = make_report(
    #         pred_metrics_dict=pred_metrics_dict,
    #         pred_census_im_dict=pred_census_im_dict,
    #         pred_overlay_im_dict=pred_overlay_im_dict,
    #         pred_and_acq=pred_and_acq,
    #         brain_vol_vox=brain_vol_vox,
    #         thr_cluster_vals=self.inputs.thr_cluster_vals,
    #         min_seg_size=self.inputs.min_seg_size,
    #         models_uid=models_uid,
    #         bounding_crop=crop_brain_img,
    #         overlayed_brainmask_1=overlayed_brainmask_1,
    #         overlayed_brainmask_2=overlayed_brainmask_2,
    #         isocontour_slides_FLAIR_T1=isocontour_slides_FLAIR_T1,
    #         subject_id=subject_id,
    #         image_size=self.inputs.image_size,
    #         resolution=self.inputs.resolution,
    #         percentile=self.inputs.percentile,
    #         threshold=self.inputs.threshold,
    #         wf_graph=wf_graph
    #     )

    #     with open('Shiva_report.html', 'w', encoding='utf-8') as fid:
    #         fid.write(html_report)

    #     # Convert the HTML file to PDF using CSS
    #     # Creating custom CSSin addition to the main one for the pages header and the logos
    #     postproc_dir = os.path.dirname(postproc_init)
    #     css = CSS(os.path.join(postproc_dir, 'printed_styling.css'))
    #     now = datetime.now(timezone.utc).strftime("%Y/%m/%d - %H:%M (UTC)")
    #     content_sub_id = f'Patient ID: {subject_id} \A ' if subject_id else ''
    #     header = (
    #         '@page {'
    #         '   @top-left {'
    #         f'      content: "{content_sub_id}Data-base: {db}";'
    #         '       font-size: 10pt;'
    #         '       white-space: pre;'
    #         '   }'
    #         '   @top-center {'
    #         f'      content: "{now}";'
    #         '       font-size: 10pt;'
    #         '   }'
    #         '}'
    #     )
    #     css_header = CSS(string=header)
    #     shiva_logo_file = os.path.join(postproc_dir, 'logo_shiva.png')
    #     other_logos_file = os.path.join(postproc_dir, 'logos_for_shiva.png')
    #     with open(shiva_logo_file, 'rb') as f:
    #         image_data = f.read()
    #         shiva_logo = base64.b64encode(image_data).decode()
    #     with open(other_logos_file, 'rb') as f:
    #         image_data = f.read()
    #         other_logo = base64.b64encode(image_data).decode()
    #     logo = (
    #         '@page {'
    #         '   @bottom-left {'
    #         f'      background-image: url(data:image/png;base64,{other_logo});'
    #         '       background-size: 552px 45px;'
    #         '       display: inline-block;'
    #         '       width: 560px; '
    #         '       height: 60px;'
    #         '       content:"";'
    #         '       background-repeat: no-repeat;'
    #         '   }'
    #         '   @top-right-corner {'
    #         f'      background-image: url(data:image/png;base64,{shiva_logo});'
    #         '       background-size: 70px 70px;'
    #         '       display: inline-block;'
    #         '       width: 70px; '
    #         '       height: 70px;'
    #         '       content:"";'
    #         '       background-repeat: no-repeat;'
    #         '   }'
    #         '}'
    #     )
    #     logo_css = CSS(string=logo)
    #     HTML('Shiva_report.html').write_pdf('Shiva_report.pdf',
    #                                         stylesheets=[css, css_header, logo_css])

    #     setattr(self, 'html_report', os.path.abspath('Shiva_report.html'))
    #     setattr(self, 'pdf_report', os.path.abspath('Shiva_report.pdf'))
    #     return runtime

    def _run_interface(self, runtime):
        """Skip actual report generation but create placeholder files."""
        print("SummaryReport disabled: generating empty HTML and PDF files to avoid workflow errors.")

        # Create empty HTML file
        html_path = os.path.abspath('Shiva_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('<html><body><p>Summary report generation skipped.</p></body></html>')

        # Create empty (but valid) PDF file header to avoid downstream rendering issues
        pdf_path = os.path.abspath('Shiva_report.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%EOF\n')  # minimal valid PDF structure

        setattr(self, 'html_report', html_path)
        setattr(self, 'pdf_report', pdf_path)
        return runtime


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['html_report'] = getattr(self, 'html_report')
        outputs['pdf_report'] = getattr(self, 'pdf_report')

        return outputs
```