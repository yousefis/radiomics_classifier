import numpy as np
import json
import SimpleITK as sitk
import six


class SurgeryPlan:
    def transform_cutting_plane(self, pth, CT_ICF):
        '''
        This function transforms the cutting plane coordinates from the robot coordinate frame (RCF)
        to image coordinate frame (ICF)
        :param pth: path to the patient data
        :return:
            scaled_translated_cutting_plane: scaled and transformed cutting plane to image coordinate frame (ICF)
            ct_shape: shape of the CT
        '''
        with open(pth + '/ICF_to_RCF_transform.json','r') as f:
            ICF_to_RCF_transform = json.load(f)
        translated_cutting_plane = CT_ICF + np.tile(ICF_to_RCF_transform['Offsets'], (4, 1))
        scaled_translated_cutting_plane = [np.transpose(np.matmul(np.diag(np.reciprocal(ICF_to_RCF_transform['Scales'])), np.transpose(i)))
                                           for i in  translated_cutting_plane]
        scaled_translated_cutting_plane = np.int32(scaled_translated_cutting_plane)

        ct_shape = np.load(pth + '/CT_ICF.npy').shape
        return ct_shape, scaled_translated_cutting_plane


    def binary_cutting_plane(self, shape, cutting_vertices):
        '''
        This function returns a 3D binary mask of the cutting plane
        :param shape: shape of the binary mask (same as the CT shape)
        :param cutting_vertices: the vertices of the cutting plane in image coordinate frame (ICF)
        :return:
        '''
        # the equation of a hyper-plane with 3 points of A,B,C:
        #AB=B-A
        #AC=C-A
        #np.cross(AB,AC) = (a,b,c)
        #d=-(a*A_x + b*A_y + c*A_z)

        cutting_plane = np.zeros(shape)
        V12 = cutting_vertices[0] - cutting_vertices[2]
        V13 = cutting_vertices[1] - cutting_vertices[2]
        a, b, c = np.cross(V12,V13) #cross product returns the coefficents of the hyper-plane
        d = -(a * cutting_vertices[0][0] + b * cutting_vertices[0][1] + c * cutting_vertices[0][2]) # the constant of the hyper-palne

        #make the binary mask
        for x in range(0,shape[0]):
            for y in range(0,shape[1]):
                for z in range(0,shape[2]):
                    if a * x + b * y + c * z + d > 0: #the equation of the hyper-plane
                        cutting_plane[x, y, z] = 1
        return cutting_plane


    def remove_remain_bone(self, CT_ICF, Seg, pth):
        cutting_plane = sitk.ReadImage(pth + '/cutting_plane.mha')
        filter = sitk.MultiplyImageFilter()
        # mask ct with the cutting plane
        # masked_ct_cutting_plane = filter.Execute(CT_ICF, sitk.Cast(cutting_plane, sitk.sitkFloat32))
        # extract bone by masking the CT and segmentation
        bone = filter.Execute(CT_ICF, sitk.Cast(Seg, sitk.sitkFloat32))
        # removed bone
        removed_bone = sitk.Cast(filter.Execute(bone, sitk.Cast(cutting_plane, sitk.sitkFloat32)), sitk.sitkFloat32)
        # remained bone
        not_filter = sitk.BinaryNotImageFilter()
        remained_bone = filter.Execute(bone, sitk.Cast(not_filter.Execute(sitk.Cast(cutting_plane, sitk.sitkUInt8)),
                                                       sitk.sitkFloat32))
        sitk.WriteImage(removed_bone, pth + '/removed_bone.mha')
        sitk.WriteImage(remained_bone, pth + '/remained_bone.mha')
        return removed_bone, remained_bone

if __name__ == "__main__":
    # instantiate from class SurgeryPlan
    surgery_plan = SurgeryPlan()
    # make binary mask for patient A
    pth = 'SendToCandidate/PatientA'
    # A_CT_ICF = np.loadtxt(pth + '/cutting_plane_RCF.txt')
    # ct_shape, A_scaled_translated_cutting_plane = surgery_plan.transform_cutting_plane(pth=pth, CT_ICF=A_CT_ICF)
    # A_cutting_plane = surgery_plan.binary_cutting_plane(shape=ct_shape, cutting_vertices=A_scaled_translated_cutting_plane)
    # sitk.WriteImage(sitk.GetImageFromArray(A_cutting_plane), pth + '/cutting_plane.mha')

    # get removed and remained bone for patient A
    A_CT_ICF = sitk.ReadImage(pth + '/CT.mha')
    A_Seg = sitk.ReadImage(pth + '/seg.mha')
    A_removed_bone, A_remained_bone = surgery_plan.remove_remain_bone(CT_ICF=A_CT_ICF, Seg=A_Seg, pth=pth)

    # use the biomarkers to decide if the removed and remained bones are diseased and healthy


    # --------------------------------------------------------------------- Patient B

    # make binary mask for patient B
    pth = 'SendToCandidate/PatientB'
    # B_CT_ICF = np.loadtxt(pth + '/cutting_plane_RCF.txt')
    # ct_shape, B_scaled_translated_cutting_plane = surgery_plan.transform_cutting_plane(pth=pth, CT_ICF=B_CT_ICF)
    # B_cutting_plane = surgery_plan.binary_cutting_plane(shape=ct_shape, cutting_vertices=B_scaled_translated_cutting_plane)
    # sitk.WriteImage(sitk.GetImageFromArray(B_cutting_plane), pth + '/cutting_plane.mha')

    # get removed and remained bone for patient B
    B_CT_ICF = sitk.ReadImage(pth + '/CT.mha')
    B_Seg = sitk.ReadImage(pth + '/seg.mha')
    B_removed_bone, B_remained_bone = surgery_plan.remove_remain_bone(CT_ICF=B_CT_ICF, Seg=B_Seg, pth=pth)




    print(1)

