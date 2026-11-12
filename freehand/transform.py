"""
Geometric transforms between image, tool (probe marker) and world frames.

All transforms are 4x4 homogeneous matrices applied by left-multiplication:
    T{image->world} = T{tool->world} @ T_cal
    pts{world} = T{tool->world} @ T_cal @ pts{image}
where T_cal = T{image->tool} is the calibration matrix and pts{image} are the
four image corner points (pixels).

The relative transform between two frames,
    T{tool1->tool0} = T{tool0->world}^-1 @ T{tool1->world},
is independent of the world frame; it is the quantity the network predicts.
Chained relative transforms give the pose relative to an earlier frame:
    T{tool2->tool0} = T{tool1->tool0} @ T{tool2->tool1}
"""

import torch


class LabelTransform():

    def __init__(
        self, 
        label_type, 
        pairs,
        image_points=None,
        in_image_coords=False,
        tform_image_to_tool=None
        ):        
        """
        :param label_type: {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        :param image_points: 2-by-num or 3-by-num in pixel, used for computing loss
        :param in_image_coords: label points coordinates, True (in pixel) or False (in mm), default is False
        :param tform_image_to_tool: transformation from image coordinates to tool coordinates, usually obtained from calibration
        """


        self.label_type = label_type
        self.pairs = pairs
        
        
        if self.label_type=="point":        
            self.image_points = image_points
            self.tform_image_to_tool = tform_image_to_tool
            # pre-compute reference points in tool coordinates
            self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
            self.in_image_coords = in_image_coords
            if self.in_image_coords:  # pre-compute the inverse
                self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
            
            self.call_function = self.to_points
        
        elif self.label_type=="transform":  # not in use
            self.call_function = self.to_transform_t2t   

        elif self.label_type=="parameter":
            self.call_function = self.to_parameters   

        else:
            raise('Unknown label_type!')
    

    def __call__(self, *args, **kwargs):
        return self.call_function(*args, **kwargs)        
    

    def to_points(self, tforms, tforms_inv=None):
        _tforms = self.to_transform_t2t(tforms, tforms_inv)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]

    
    def to_transform_t2t(self, tforms, tforms_inv):

        if tforms_inv is None:
            tforms_inv = torch.linalg.inv(tforms)
        
        tforms_world_to_tool0 = tforms_inv[:,self.pairs[:,0],:,:]
        tforms_tool1_to_world = tforms[:,self.pairs[:,1],:,:]
        return torch.matmul(tforms_world_to_tool0, tforms_tool1_to_world)  # tform_tool1_to_tool0


    def to_parameters(self, tforms, tforms_inv):
        raise('Not implemented.')  # TODO: SVD
    


class PredictionTransform():

    def __init__(
        self, 
        pred_type, 
        label_type, 
        num_pairs=None,
        image_points=None,
        in_image_coords=False,
        tform_image_to_tool=None
        ):
        
        """
        :param pred_type = {"transform", "parameter", "point"}
        :param label_type = {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        """


        self.pred_type = pred_type
        self.label_type = label_type
        self.num_pairs = num_pairs


        if self.pred_type=="point":
            if self.label_type=="point":
                self.call_function = self.point_to_point
            elif self.label_type=="parameter":
                raise('Not implemented.')  #self.call_function = self.point_to_parameter    
            else:
                raise('Unknown label_type!')
            
        else:            
            self.image_points = image_points
            self.tform_image_to_tool = tform_image_to_tool
            self.in_image_coords = in_image_coords
            # pre-compute reference points in tool coordinates
            self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
            self.in_image_coords = in_image_coords
            if self.in_image_coords:  # pre-compute the inverse
                self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)

            if self.pred_type=="parameter":
                if self.label_type=="point":
                    self.call_function = self.parameter_to_point
                elif self.label_type=="parameter":
                    self.call_function = self.parameter_to_parameter
                elif self.label_type=="transform":
                    self.call_function = self.param_to_transform

                else:
                    raise('Unknown label_type!')

            elif self.pred_type=="transform":
                if self.label_type=="point":
                    self.call_function = self.transform_to_point
                elif self.label_type=="parameter":
                    raise('Not implemented.')  #self.call_function = self.transform_to_parameter
                else:
                    raise('Unknown label_type!')
            
            else:
                raise('Unknown pred_type!')
        

    def __call__(self, outputs):
        preds = outputs.reshape((outputs.shape[0],self.num_pairs,-1))
        return self.call_function(preds)


    def point_to_point(self,pts):
        return pts.reshape(pts.shape[0],self.num_pairs,-1,3)
    
    def transform_to_point(self,_tforms):
        last_rows = torch.cat([
            torch.zeros_like(_tforms[...,0])[...,None,None].expand(-1,-1,1,3),
            torch.ones_like(_tforms[...,0])[...,None,None]
            ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1,self.num_pairs,3,4),
            last_rows
            ), axis=2)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]

    def parameter_to_parameter(self,params):
        return params
    
    def parameter_to_point(self,params):
        _tforms = self.param_to_transform(params)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]
    
    @staticmethod
    def param_to_transform(params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
    
        cos_x = torch.cos(params[...,2])
        sin_x = torch.sin(params[...,2])
        cos_y = torch.cos(params[...,1])
        sin_y = torch.sin(params[...,1])
        cos_z = torch.cos(params[...,0])
        sin_z = torch.sin(params[...,0])
        return torch.cat((
            torch.stack([cos_y*cos_z, sin_x*sin_y*cos_z-cos_x*sin_z,  cos_x*sin_y*cos_z+sin_x*sin_z,  params[...,3]], axis=2)[:,:,None,:],
            torch.stack([cos_y*sin_z, sin_x*sin_y*sin_z+cos_x*cos_z,  cos_x*sin_y*sin_z-sin_x*cos_z,  params[...,4]], axis=2)[:,:,None,:],
            torch.stack([-sin_y,      sin_x*cos_y,                    cos_x*cos_y,                    params[...,5]], axis=2)[:,:,None,:],
            torch.cat((torch.zeros_like(params[...,0:3])[...,None,:], torch.ones_like(params[...,0])[...,None,None]), axis=3)
            ), axis=2)



class ImageTransform:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,images):
        images = (images.to(torch.float32) - self.mean) / self.std
        return images + torch.normal(mean=0, std=torch.ones_like(images)*0.01)


# ---------------------------------------------------------------------------
# 6D Rotation Representation (Zhou et al.)
# ---------------------------------------------------------------------------

def sixd_to_matrix(r6d):
    """
    Convert 6D rotation representation to 4×4 transform matrices.

    :param r6d: (..., 9) — first 6 values are the 6D rotation, last 3 are translation.
    :return: (..., 4, 4) — homogeneous transform matrices.
    """
    r1 = r6d[..., 0:3]                         # first column (..., 3)
    r2 = r6d[..., 3:6]                         # second column (..., 3)
    t  = r6d[..., 6:9]                         # translation (..., 3)

    # Gram-Schmidt orthogonalisation (eps=1e-8 for stability)
    r1 = torch.nn.functional.normalize(r1, dim=-1, eps=1e-8)
    r2 = r2 - (r1 * r2).sum(dim=-1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(r2, dim=-1, eps=1e-8)
    r3 = torch.cross(r1, r2, dim=-1)           # third column

    R = torch.stack([r1, r2, r3], dim=-1)       # (..., 3, 3)
    R = torch.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)

    last_row = torch.tensor([0, 0, 0, 1], dtype=r6d.dtype, device=r6d.device)
    last_row = last_row.expand(*r6d.shape[:-1], 1, 4)

    T = torch.cat([
        torch.cat([R, t.unsqueeze(-1)], dim=-1),  # (..., 3, 4)
        last_row,
    ], dim=-2)                                   # (..., 4, 4)
    return T


class PredictionTransform6D:
    """Prediction transform that converts 6D rotation + translation to point predictions."""

    def __init__(self, image_points, tform_image_to_tool):
        self.image_points = image_points
        self.tform_image_to_tool = tform_image_to_tool
        # Pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(
            self.tform_image_to_tool, self.image_points)

    def __call__(self, outputs):
        """
        :param outputs: (batch, num_pairs, 9) — 6D rotation + translation per pair
        :return: (batch, num_pairs, 3, num_points) — corner points in tool coords
        """
        T = sixd_to_matrix(outputs)              # (batch, num_pairs, 4, 4)
        corners = torch.matmul(T, self.image_points_in_tool)  # (batch, num_pairs, 4, num_points)
        return corners[..., :3, :]               # (batch, num_pairs, 3, num_points)


class TransformAccumulation:

    def __init__(
        self, 
        image_points=None,
        in_image_coords=False,  # TODO: accept image1->image0 instead of tool1->tool0
        tform_image_to_tool=None
        ):
        if in_image_coords:
            raise("not implemented.")
        
        self.image_points = image_points
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool) # pre-compute the inverse
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)

    def __call__(self, tform_tool1_to_tool0, tform_tool2_to_tool1):  
        # The accumulated transform is passed in and returned explicitly
        
        # last_rows = torch.cat((torch.zeros_like(tform_tool2_to_tool1[0,0,:,0]),torch.ones_like(tform_tool2_to_tool1[0,0,0:1,0])),axis=0).expand(list(tform_tool2_to_tool1.shape[0:2])[0],list(tform_tool2_to_tool1.shape[0:2])[1],1,4)
        # tform_tool2_to_tool1 = torch.cat((
        #     tform_tool2_to_tool1,
        #     last_rows
        #     ), axis=2)
        
        tform_tool2_to_tool0 = torch.matmul(tform_tool1_to_tool0, tform_tool2_to_tool1)
        # tform_tool2_to_image0 = torch.matmul(self.tform_tool_to_image, tform_tool2_to_tool0)
        return torch.matmul(tform_tool2_to_tool0, self.image_points_in_tool)[0:3,:], tform_tool2_to_tool0
