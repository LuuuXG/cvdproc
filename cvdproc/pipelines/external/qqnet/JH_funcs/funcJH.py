import os
import numpy as np
from numpy import zeros
import time as time
import torch

def batchloader(Output,Target,batch_size,idx0):

#     Input: Output = [x,y,z,te], Target = [x,y,z,unknowns], batch_size = [bx,by,bz], idx0 = [cx,cy,cz] in batch center
#     Output: Output_batch = [1,te,bz,by,bx] Target_batch = [1,unknowns,bz,by,bx]
    
    
    # Indices [i,j,k] in Mask
#     pm_x = np.where(Mask == 1)[0]
#     pm_y = np.where(Mask == 1)[1]
#     pm_z = np.where(Mask == 1)[2]
#     N_Mask = len(pm_x)
    
    
#     # Pick a ramdom voxel in Mask (Uniformly random)
#     tmp = np.random.randint(low=1, high=N_Mask)
#     tmp
#     [pm_x[tmp],pm_y[tmp],pm_z[tmp]]
#     idx0 = [pm_x[tmp],pm_y[tmp],pm_z[tmp]]
        
    # Use it as the central voxel of cropped batch
    ind_x1 = np.around(idx0[0]-batch_size[0]/2).astype(int)
    ind_x2 = np.around(idx0[0]+batch_size[0]/2).astype(int)+1
    ind_y1 = np.around(idx0[1]-batch_size[1]/2).astype(int)
    ind_y2 = np.around(idx0[1]+batch_size[1]/2).astype(int)+1
    ind_z1 = np.around(idx0[2]-batch_size[2]/2).astype(int)
    ind_z2 = np.around(idx0[2]+batch_size[2]/2).astype(int)+1
        
    ind_x1 = np.maximum(0,ind_x1)
    ind_y1 = np.maximum(0,ind_y1)
    ind_z1 = np.maximum(0,ind_z1)
        
    ind_x2 = np.minimum(Output.shape[0]+1,ind_x2)
    ind_y2 = np.minimum(Output.shape[1]+1,ind_y2)
    ind_z2 = np.minimum(Output.shape[2]+1,ind_z2)
        
    # Crop batch
    batch_y_t = Target[ind_x1:ind_x2,ind_y1:ind_y2,ind_z1:ind_z2,:]
    batch_x_t = Output[ind_x1:ind_x2,ind_y1:ind_y2,ind_z1:ind_z2,:]
        
    batch_x_t = np.transpose(batch_x_t,(3,2,1,0));
    batch_y_t = np.transpose(batch_y_t,(3,2,1,0));
        
    # Change to 5D  [1,channel,z,y,x]  
    ppx = batch_x_t.shape
    ppy = batch_y_t.shape
    Output_batch = zeros([1,ppx[0],ppx[1],ppx[2],ppx[3]])
    Target_batch = zeros([1,ppy[0],ppy[1],ppy[2],ppy[3]])
        
    Output_batch[0,:,:,:,:] = batch_x_t[:,:,:,:]
    Target_batch[0,:,:,:,:] = batch_y_t[:,:,:,:]

    return Output_batch, Target_batch


def batchloader1(Target,batch_size,idx0):

#     Input: Target = [x,y,z,unknowns], batch_size = [bx,by,bz], idx0 = [cx,cy,cz] in batch center
#     Target_batch = [1,unknowns,bz,by,bx]
    
    
    # Indices [i,j,k] in Mask
#     pm_x = np.where(Mask == 1)[0]
#     pm_y = np.where(Mask == 1)[1]
#     pm_z = np.where(Mask == 1)[2]
#     N_Mask = len(pm_x)
    
    
#     # Pick a ramdom voxel in Mask (Uniformly random)
#     tmp = np.random.randint(low=1, high=N_Mask)
#     tmp
#     [pm_x[tmp],pm_y[tmp],pm_z[tmp]]
#     idx0 = [pm_x[tmp],pm_y[tmp],pm_z[tmp]]
        
    # Use it as the central voxel of cropped batch
    ind_x1 = np.around(idx0[0]-batch_size[0]/2).astype(int)
    ind_x2 = np.around(idx0[0]+batch_size[0]/2).astype(int)+1
    ind_y1 = np.around(idx0[1]-batch_size[1]/2).astype(int)
    ind_y2 = np.around(idx0[1]+batch_size[1]/2).astype(int)+1
    ind_z1 = np.around(idx0[2]-batch_size[2]/2).astype(int)
    ind_z2 = np.around(idx0[2]+batch_size[2]/2).astype(int)+1
        
    ind_x1 = np.maximum(0,ind_x1)
    ind_y1 = np.maximum(0,ind_y1)
    ind_z1 = np.maximum(0,ind_z1)
        
    ind_x2 = np.minimum(Target.shape[0]+1,ind_x2)
    ind_y2 = np.minimum(Target.shape[1]+1,ind_y2)
    ind_z2 = np.minimum(Target.shape[2]+1,ind_z2)
        
    # Crop batch
    batch_y_t = Target[ind_x1:ind_x2,ind_y1:ind_y2,ind_z1:ind_z2,:]
    batch_y_t = np.transpose(batch_y_t,(3,2,1,0));
        
    # Change to 5D  [1,channel,z,y,x]  
    ppy = batch_y_t.shape
    Target_batch = zeros([1,ppy[0],ppy[1],ppy[2],ppy[3]])
        
    Target_batch[0,:,:,:,:] = batch_y_t[:,:,:,:]

    return Target_batch



def loader_trans(Output,Target):
    batch_y_t = Target
    batch_x_t = Output

    batch_x_t = np.transpose(batch_x_t,(3,2,1,0));
    batch_y_t = np.transpose(batch_y_t,(3,2,1,0));

    # Change to 5D  [1,channel,z,y,x]  
    ppx = batch_x_t.shape
    ppy = batch_y_t.shape
    Output_batch = zeros([1,ppx[0],ppx[1],ppx[2],ppx[3]])
    Target_batch = zeros([1,ppy[0],ppy[1],ppy[2],ppy[3]])

    Output_batch[0,:,:,:,:] = batch_x_t[:,:,:,:]
    Target_batch[0,:,:,:,:] = batch_y_t[:,:,:,:]
    return Output_batch, Target_batch
    
def loader_trans_alone(Output):
    batch_x_t = Output
    batch_x_t = np.transpose(batch_x_t,(3,2,1,0));


        # Change to 5D  [1,channel,z,y,x]  
    ppx = batch_x_t.shape
    Output_batch = zeros([1,ppx[0],ppx[1],ppx[2],ppx[3]])

    Output_batch[0,:,:,:,:] = batch_x_t[:,:,:,:]
      
    return Output_batch
    
    
def QQ(x,te,p_s):
#   Input(long, S0, R2,Y,v,chinb): 5*N_v
#   Output(qBOLD,QSM): 9*N_v
#   te = [1xN_te]
#   reshape default: Matlab -> Fortran, Python -> C 
    S0 = x[0][:]
    R2 = x[1][:]
    Y = x[2][:]
    v = x[3][:]
    chinb = x[4][:]

    #     S0 = np.reshape(S0,(S0.shape[0],1))
    #     R2 = np.reshape(R2,(R2.shape[0],1))
    #     Y = np.reshape(Y,(Y.shape[0],1))
    #     v = np.reshape(v,(v.shape[0],1))
    #     chinb = np.reshape(chinb,(chinb.shape[0],1))
    t0 = time.time() 
    
    S0 = S0.reshape(S0.shape[0],1)
    R2 = R2.reshape(R2.shape[0],1)
    Y = Y.reshape(Y.shape[0],1)
    v = v.reshape(v.shape[0],1)
    chinb = chinb.reshape(chinb.shape[0],1)
    
    print(time.time()-t0)
    
    pi = np.pi


    #   consts
    Hct_v = 0.47;
    Hct_ratio = 0.759;
    Hct = Hct_v*Hct_ratio;
    Hct = 0.3567
    fo = Hct*0.34/1.335;
    fo_vein = Hct_v*0.34/1.335;

    C_Hb = Hct*0.34/(64450*10^-6)
    SaO2 = 0.98;
    Chi_pl = -37.7
    Chi_OxyHb = -813; 
    Chi_DeoxyHb = 11709; 
    dChi = Chi_DeoxyHb-Chi_OxyHb;
    CaO2 = 4*SaO2*C_Hb;
    Chi_aB = (fo*Chi_OxyHb)+(1-fo)*Chi_pl;

    gamma = 267.513 ;
    del_chi0 = 0.27 ;
    B0 = 3; 
    Chi_aB1 = -0.0086;

    const1 = 4*pi/3*gamma*B0;
    const2 = Hct*del_chi0;

    chi_nb_t = chinb/(4*pi)

    
    dw = abs(const1*(const2*(1-Y)+Chi_aB1-chi_nb_t));
    input00 = dw*te
    input0 = input00.reshape(np.prod(input00.shape))
    
    print(input0.shape)
    
    f_s = np.polyval(p_s,input0)
    
 
    gf1_30_1D_n1t = f_s.reshape(input00.shape)
  
    F_BOLD_Yt = np.exp(-v*gf1_30_1D_n1t)
    R2_tempt = np.exp(-R2*te)
    s_qBOLD = S0*F_BOLD_Yt*R2_tempt
    
    #     QSM
    ks01 = fo*dChi; ks02 = Chi_aB; ks03 = SaO2; ks04 = CaO2;
    c0 = 0.77;

    # %ppb (SI)
    c_chi_nb = 1e3;
    chi_nb_11 = chinb*c_chi_nb;

    A3 = (1/c0*(ks02+ks01*(1-ks03)) + ks01*ks03);

    QSM = -ks01*Y*v - 1/c0*v*chi_nb_11 + A3*v + chi_nb_11;

    QSM = QSM/(1e3)

    f = np.concatenate((s_qBOLD,QSM), axis=1)
    f = np.transpose(f,(1,0));
    return f


def QQ_gpu(x,te,p_s):
#   Input(long, S0, R2,Y,v,chinb): 5*N_v
#   Output(qBOLD,QSM): 9*N_v
#   te = [1xN_te]
#   reshape default: Matlab -> Fortran, Python -> C 
    S0 = x[0][:]
    R2 = x[1][:]
    Y = x[2][:]
    v = x[3][:]
    chinb = x[4][:]

    #     S0 = np.reshape(S0,(S0.shape[0],1))
    #     R2 = np.reshape(R2,(R2.shape[0],1))
    #     Y = np.reshape(Y,(Y.shape[0],1))
    #     v = np.reshape(v,(v.shape[0],1))
    #     chinb = np.reshape(chinb,(chinb.shape[0],1))
    
    S0 = S0.reshape(S0.shape[0],1)
    R2 = R2.reshape(R2.shape[0],1)
    Y = Y.reshape(Y.shape[0],1)
    v = v.reshape(v.shape[0],1)
    chinb = chinb.reshape(chinb.shape[0],1)
    
    pi = np.pi

    #   consts
    Hct_v = 0.47;
    Hct_ratio = 0.759;
#     Hct = Hct_v*Hct_ratio;
    Hct = 0.3567
    fo = Hct*0.34/1.335;
    fo_vein = Hct_v*0.34/1.335;

    C_Hb = Hct*0.34/(64450*10^-6)
    SaO2 = 0.98;
    Chi_pl = -37.7
    Chi_OxyHb = -813; 
    Chi_DeoxyHb = 11709; 
    dChi = Chi_DeoxyHb-Chi_OxyHb;
    CaO2 = 4*SaO2*C_Hb;
    Chi_aB = (fo*Chi_OxyHb)+(1-fo)*Chi_pl;

    gamma = 267.513 ;
    del_chi0 = 0.27 ;
    B0 = 3; 
    Chi_aB1 = Chi_aB/(4*pi*1000);

    const1 = 4*pi/3*gamma*B0;
    const2 = Hct*del_chi0;

    chi_nb_t = chinb/(4*pi)

    
    dw = abs(const1*(const2*(1-Y)+Chi_aB1-chi_nb_t));
    
    #     GPU conversion
    te1 = torch.from_numpy(np.array(te)).type(torch.DoubleTensor).cuda()
#     te1 = torch.from_numpy(np.array(te)).cuda()
    
    input00 = torch.mm(dw,te1)
    
#     input00 = torch.ones(dw.shape[0],te.shape[1])
    input0 = input00.reshape(np.prod(input00.shape))
    
#     print(input0.shape)
#   polyval at cpu
    input0 = input0.detach().cpu().numpy()
    f_s = np.polyval(p_s,input0)
    f_s = torch.DoubleTensor(f_s).cuda()
    gf1_30_1D_n1t = f_s.reshape(input00.shape)

#     gf1_30_1D_n1t = input0.reshape(input00.shape)
#     gf1_30_1D_n1t = input00
  
#     F_BOLD_Yt = np.exp(-v*gf1_30_1D_n1t)
#     R2_tempt = np.exp(-R2*te)
    
    F_BOLD_Yt = torch.exp(-v*gf1_30_1D_n1t)
    R2_tempt = torch.exp(-R2*te1)
    
    s_qBOLD = S0*F_BOLD_Yt*R2_tempt
    
    #     QSM
    ks01 = fo*dChi; ks02 = Chi_aB; ks03 = SaO2; ks04 = CaO2;
    c0 = 0.77;
    

    # %ppb (SI)  
    c_chi_nb = 1e3;
    
#     GPU conversion  
    c_chi_nb = torch.from_numpy(np.array(c_chi_nb)).type(torch.DoubleTensor).cuda()
    c0 = torch.from_numpy(np.array(c0)).type(torch.DoubleTensor).cuda()

    ks01 = torch.from_numpy(np.array(ks01)).type(torch.DoubleTensor).cuda()
    ks02 = torch.from_numpy(np.array(ks02)).type(torch.DoubleTensor).cuda()
    ks03 = torch.from_numpy(np.array(ks03)).type(torch.DoubleTensor).cuda()
#     ks01 = torch.from_numpy(ks01)
    
    #     GPU conversion
#     c_chi_nb = torch.FloatTensor(c_chi_nb).cuda()
#     ks01 = torch.FloatTensor(ks01).cuda()
#     ks02 = torch.FloatTensor(ks02).cuda()
#     ks03 = torch.FloatTensor(ks03).cuda()
    
    chi_nb_11 = chinb*c_chi_nb;

    A3 = (1/c0*(ks02+ks01*(1-ks03)) + ks01*ks03);
 
    QSM = -ks01*Y*v - 1/c0*v*chi_nb_11 + A3*v + chi_nb_11;

    QSM = QSM/(1e3)
    

#     f = np.concatenate((s_qBOLD,QSM), axis=1)
#     f = np.transpose(f,(1,0));

    f = torch.cat((s_qBOLD,QSM),1)
    f = torch.transpose(f,0,1)
    
    return f
          
def QQ_gpu_single(x,te,p_s):
    # Input precision float32(single), Output precision float(single)
#   Input(long, S0, R2,Y,v,chinb): 5*N_v
#   Output(qBOLD,QSM): 9*N_v
#   te = [1xN_te]
#   reshape default: Matlab -> Fortran, Python -> C 
    S0 = x[0][:]
    R2 = x[1][:]
    Y = x[2][:]
    v = x[3][:]
    chinb = x[4][:]

    #     S0 = np.reshape(S0,(S0.shape[0],1))
    #     R2 = np.reshape(R2,(R2.shape[0],1))
    #     Y = np.reshape(Y,(Y.shape[0],1))
    #     v = np.reshape(v,(v.shape[0],1))
    #     chinb = np.reshape(chinb,(chinb.shape[0],1))
    
    S0 = S0.reshape(S0.shape[0],1)
    R2 = R2.reshape(R2.shape[0],1)
    Y = Y.reshape(Y.shape[0],1)
    v = v.reshape(v.shape[0],1)
    chinb = chinb.reshape(chinb.shape[0],1)
    
    pi = np.pi

    #   consts
    Hct_v = 0.47;
    Hct_ratio = 0.759;
#     Hct = Hct_v*Hct_ratio;
    Hct = 0.3567
    fo = Hct*0.34/1.335;
    fo_vein = Hct_v*0.34/1.335;

    C_Hb = Hct*0.34/(64450*10^-6)
    SaO2 = 0.98;
    Chi_pl = -37.7
    Chi_OxyHb = -813; 
    Chi_DeoxyHb = 11709; 
    dChi = Chi_DeoxyHb-Chi_OxyHb;
    CaO2 = 4*SaO2*C_Hb;
    Chi_aB = (fo*Chi_OxyHb)+(1-fo)*Chi_pl;

    gamma = 267.513 ;
    del_chi0 = 0.27 ;
    B0 = 3; 
    Chi_aB1 = Chi_aB/(4*pi*1000);

    const1 = 4*pi/3*gamma*B0;
    const2 = Hct*del_chi0;

    chi_nb_t = chinb/(4*pi)

    
    dw = abs(const1*(const2*(1-Y)+Chi_aB1-chi_nb_t));
    
#     dw1 = dw.type(torch.DoubleTensor).cuda() 
    #     GPU conversion
    te1 = torch.from_numpy(np.array(te)).type(torch.FloatTensor).cuda()
#     te1 = torch.from_numpy(np.array(te)).cuda()
    
    input00 = torch.mm(dw,te1)
    
#     input00 = torch.ones(dw.shape[0],te.shape[1])
    input0 = input00.reshape(np.prod(input00.shape))
    
#     print(input0.shape)
#   polyval at cpu
    input0 = input0.detach().cpu().numpy()
    f_s = np.polyval(p_s,input0)
    f_s = torch.FloatTensor(f_s).cuda()
    gf1_30_1D_n1t = f_s.reshape(input00.shape)

#     gf1_30_1D_n1t = input0.reshape(input00.shape)
#     gf1_30_1D_n1t = input00
  
#     F_BOLD_Yt = np.exp(-v*gf1_30_1D_n1t)
#     R2_tempt = np.exp(-R2*te)
    
    F_BOLD_Yt = torch.exp(-v*gf1_30_1D_n1t)
    R2_tempt = torch.exp(-R2*te1)
    
    s_qBOLD = S0*F_BOLD_Yt*R2_tempt
    
    #     QSM
    ks01 = fo*dChi; ks02 = Chi_aB; ks03 = SaO2; ks04 = CaO2;
    c0 = 0.77;
    

    # %ppb (SI)  
    c_chi_nb = 1e3;
    
#     GPU conversion  
#     c_chi_nb = torch.from_numpy(np.array(c_chi_nb)).type(torch.DoubleTensor).cuda()
#     c0 = torch.from_numpy(np.array(c0)).type(torch.DoubleTensor).cuda()

#     ks01 = torch.from_numpy(np.array(ks01)).type(torch.DoubleTensor).cuda()
#     ks02 = torch.from_numpy(np.array(ks02)).type(torch.DoubleTensor).cuda()
#     ks03 = torch.from_numpy(np.array(ks03)).type(torch.DoubleTensor).cuda()
#     ks01 = torch.from_numpy(ks01)
    
    #     GPU conversion
    c_chi_nb = torch.from_numpy(np.array(c_chi_nb)).type(torch.FloatTensor).cuda()
    c0 = torch.from_numpy(np.array(c0)).type(torch.FloatTensor).cuda()
    ks01 = torch.from_numpy(np.array(ks01)).type(torch.FloatTensor).cuda()
    ks02 = torch.from_numpy(np.array(ks02)).type(torch.FloatTensor).cuda()
    ks03 = torch.from_numpy(np.array(ks03)).type(torch.FloatTensor).cuda()
    
    chi_nb_11 = chinb*c_chi_nb;

    A3 = (1/c0*(ks02+ks01*(1-ks03)) + ks01*ks03);
 
    QSM = -ks01*Y*v - 1/c0*v*chi_nb_11 + A3*v + chi_nb_11;

    QSM = QSM/(1e3)
    
#     s_qBOLD = s_qBOLD.type(torch.FloatTensor).cuda()
#     QSM = QSM.type(torch.FloatTensor).cuda()

#     f = np.concatenate((s_qBOLD,QSM), axis=1)
#     f = np.transpose(f,(1,0));

    f = torch.cat((s_qBOLD,QSM),1)
    f = torch.transpose(f,0,1)
    
    return f
          