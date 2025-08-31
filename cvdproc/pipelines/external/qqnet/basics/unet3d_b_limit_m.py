import torch.nn as nn
import numpy as np
import torch

from cvdproc.pipelines.external.qqnet.basics.utils_u_b import unetConv3d, unetUp3d


class unet3d(nn.Module):
    def __init__(self, 
        feature_scale = 2, 
        n_classes = 5, 
        is_deconv = True, 
        in_channels = 9, 
        is_batchnorm = False,
        is_hpool = True,
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv2 = unetConv3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv3 = unetConv3d(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv4 = unetConv3d(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.center = unetConv3d(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs, xmins, xmaxs, Mask):
        # 下采样/上采样主干
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)  # [B, 5, Z, Y, X]

        # ------- 边界裁剪（不再使用 np.asscalar / Hardtanh）-------
        # xmins/xmaxs 是 numpy 数组形状(5,1)，先 squeeze 成(5,)
        # 然后放到与 final 相同的 dtype / device
        xmins_t = torch.as_tensor(np.squeeze(xmins), dtype=final.dtype, device=final.device)  # [5]
        xmaxs_t = torch.as_tensor(np.squeeze(xmaxs), dtype=final.dtype, device=final.device)  # [5]

        # 变成 [1, C, 1, 1, 1] 以便广播到体素维度
        xmins_t = xmins_t.view(1, -1, 1, 1, 1)
        xmaxs_t = xmaxs_t.view(1, -1, 1, 1, 1)

        final_bounded = torch.clamp(final, min=xmins_t, max=xmaxs_t)  # [B,5,Z,Y,X]

        # ------- 乘 mask 并返回 -------
        # 你的 Mask_t1 在外部已经是 [1,1,Z,Y,X]（DoubleTensor, cuda），这里确保 dtype/device 匹配并广播
        Mask = Mask.to(dtype=final_bounded.dtype, device=final_bounded.device)
        final_masked = final_bounded * Mask  # 广播到通道维

        return final_masked    
      

#     def forward(self, inputs, xmins, xmaxs,Mask):
# #     def forward(self, inputs):
        
#         conv1 = self.conv1(inputs)
#         maxpool1 = self.maxpool1(conv1)

#         conv2 = self.conv2(maxpool1)
#         maxpool2 = self.maxpool2(conv2)

#         conv3 = self.conv3(maxpool2)
#         maxpool3 = self.maxpool3(conv3)

#         conv4 = self.conv4(maxpool3)
#         maxpool4 = self.maxpool4(conv4)

#         center = self.center(maxpool4)

#         up4 = self.up_concat4(conv4, center)
#         up3 = self.up_concat3(conv3, up4)
#         up2 = self.up_concat2(conv2, up3)
#         up1 = self.up_concat1(conv1, up2)

#         final = self.final(up1)
        
#         # apply bounds
#         final1 = final.clone()
#         final2 = final1.clone()
        
#         # min, max to a number
#         xmins0 = np.asscalar(xmins[0])
#         xmins1 = np.asscalar(xmins[1])
#         xmins2 = np.asscalar(xmins[2])
#         xmins3 = np.asscalar(xmins[3])
#         xmins4 = np.asscalar(xmins[4])
        
#         xmaxs0 = np.asscalar(xmaxs[0])
#         xmaxs1 = np.asscalar(xmaxs[1])
#         xmaxs2 = np.asscalar(xmaxs[2])
#         xmaxs3 = np.asscalar(xmaxs[3])
#         xmaxs4 = np.asscalar(xmaxs[4])
        
#         # tanh :https://pytorch.org/docs/stable/nn.html
#         self.m0 = nn.Hardtanh(xmins0, xmaxs0);
#         self.m1 = nn.Hardtanh(xmins1, xmaxs1);
#         self.m2 = nn.Hardtanh(xmins2, xmaxs2);
#         self.m3 = nn.Hardtanh(xmins3, xmaxs3);
#         self.m4 = nn.Hardtanh(xmins4, xmaxs4);
        
#         final1[:,0,:,:,:] = self.m0(final[:,0,:,:,:])
#         final1[:,1,:,:,:] = self.m1(final[:,1,:,:,:])
#         final1[:,2,:,:,:] = self.m2(final[:,2,:,:,:])
#         final1[:,3,:,:,:] = self.m3(final[:,3,:,:,:])
#         final1[:,4,:,:,:] = self.m4(final[:,4,:,:,:])
        
#         final2[:,0,:,:,:] = final1[:,0,:,:,:]*Mask[:,:,:]
#         final2[:,1,:,:,:] = final1[:,1,:,:,:]*Mask[:,:,:]
#         final2[:,2,:,:,:] = final1[:,2,:,:,:]*Mask[:,:,:]
#         final2[:,3,:,:,:] = final1[:,3,:,:,:]*Mask[:,:,:]
#         final2[:,4,:,:,:] = final1[:,4,:,:,:]*Mask[:,:,:]
        
#         return final1