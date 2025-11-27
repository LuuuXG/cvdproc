function SaveData_Chisep(Data, RunOptions)

outdir = Data.output_root;
mkdir(outdir);

Options = RunOptions;

tmp = pwd;
eval(['cd(''' outdir ''');']);
if strcmp(RunOptions.InputType, 'nifti')
    %save('results.mat','Data' ,'Options','-V7.3')
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_dia, 'ChiDia');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_para, 'ChiPara');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_tot, 'ChiTot');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.qsm_map, 'QSM_map');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.vesselMask_para, 'vesselMask_para');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.vesselMask_dia, 'vesselMask_dia');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.QSM, 'QSM');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.local_field, 'local_field');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.local_field_hz, 'local_field_hz');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.UnwrappedPhase, 'UnwrappedPhase');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.MGRE_Mag_Tukey, 'MGRE_Mag_Tukey');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    %[save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.MGRE_Phs_Tukey, 'MGRE_Phs_Tukey');
    %save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
elseif strcmp(Options.InputType, 'dicom')
    save('results.mat','Data' ,'Options','-V7.3')
    info.SeriesDescription = 'X-para [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 1;
    info.WindowCenter = 50;
    info.WindowWidth = 100;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_para*10000,Data.Dinfo,info,'ChiPara');
    
    info.SeriesDescription = 'X-dia [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 2;
    info.WindowCenter = 50;
    info.WindowWidth = 100;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_dia*10000,Data.Dinfo,info,'ChiDia');
    
    info.SeriesDescription = 'X-total [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 3;
    info.WindowCenter = 0;
    info.WindowWidth = 200;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_tot*10000,Data.Dinfo,info,'ChiTot');
else
    disp('Nifti or DICOM input were not found. Saving result.mat ...')
    msgbox('Nifti or DICOM input were not found. Saving result.mat ...');
    save('results.mat','Data' ,'Options','-V7.3')
end


eval(['cd(''' tmp ''');']);

end

function [save_func, nii_file, save_name]=load_nii_template_and_make_nii(Data, data, save_name)
    voxel_size = Data.VoxelSize;
    
    if isfield(Data,'nifti_template')
        nii_file = Data.nifti_template;
    else
        nii_file = [];
    end
    
    if isempty(nii_file)
        save_func = @save_nii;
        save_name = [save_name, '.nii'];
        origin = [1 1 1];
        nii_file = make_nii(rot90(data,-1), voxel_size, origin);
        [q, nii_file.hdr.hist.pixdim(1)] = CalculateQuatFromB0Dir(Data.B0dir);

        nii_file.hdr.hist.quatern_b = q(2);
        nii_file.hdr.hist.quatern_c = q(3);
        nii_file.hdr.hist.quatern_d = q(4);
        nii_file.hdr.hist.originator = origin;
    else
        save_func = @save_untouch_nii;
        nii_file.img = rot90(data,-1);
    end
    
    nii_file.hdr.dime.datatype = 16;
    nii_file.hdr.dime.dim(5) = size(data,4);
    nii_file.hdr.dime.dim(1) = ndims(data);

    nii_file.hdr.dime.scl_inter = 0;
    nii_file.hdr.dime.scl_slope = 1;

    nii_file.hdr.hist.magic = 'n+1';
end