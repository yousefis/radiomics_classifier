import numpy as np
import matplotlib.pyplot as plt
import six
import json
# image handling
import SimpleITK as sitk
# feature extraction
import radiomics
from radiomics import firstorder, glcm, glrlm, glszm, shape
# statistical feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class Biomarkers:
    def print_features(self, features):
        for i in features.items():
            print(i)
        print("\n")

    def features_distance(self, features1, features2):
        for f1, f2 in zip(features1.items(), features2.items()):
            print(f1[0], f1[1] - f2[1])
        print("\n")

    def extract_features(self, image, mask, feature_list):
        features = {}
        if 'FirstOrder' in feature_list:
            first_order_features = radiomics.firstorder.RadiomicsFirstOrder(image, mask)
            first_order_features.disableAllFeatures()
            first_order_features.enableAllFeatures()  # On the feature class level, all features are disabled by default.
            _first_order_features = first_order_features.execute()
            features.update(dict((key, val) for (key, val) in six.iteritems(_first_order_features)))
        if 'Shape' in feature_list:
            shape_features = shape.RadiomicsShape(image, mask)
            shape_features.enableAllFeatures()
            _shape_features = shape_features.execute()
            features.update(dict((key, val) for (key, val) in six.iteritems(_shape_features)))
        if 'GLCM' in feature_list:
            glcm_features = glcm.RadiomicsGLCM(image, mask)
            glcm_features.enableAllFeatures()
            _glcm_features = glcm_features.execute()
            features.update(dict((key, val) for (key, val) in six.iteritems(_glcm_features)))
        if 'GLRLM' in feature_list:
            glrlm_features = glrlm.RadiomicsGLRLM(image, mask)
            glrlm_features.enableAllFeatures()
            _glrlm_features = glrlm_features.execute()
            features.update(dict((key, val) for (key, val) in six.iteritems(_glrlm_features)))
        if 'GLSZM' in feature_list:
            glszm_features = glszm.RadiomicsGLSZM(image, mask)
            glszm_features.enableAllFeatures()
            _glszm_features = glszm_features.execute()
            features.update(dict((key, val) for (key, val) in six.iteritems(_glszm_features)))
        # self.print_features(features)
        return features




def medical_image_viewer(volume):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.flip(volume[volume.shape[0] // 2, :, :].T, 0), cmap="Greys_r")
    ax1.title.set_text('Sagital')
    ax1.axis('Off')

    ax2.imshow(np.flip(volume[:, volume.shape[1] // 2, :].T, 0), cmap="Greys_r")
    ax2.title.set_text('Coronal')
    ax1.axis('Off')
    plt.axis('Off')


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


    def remove_remain_bone(self,  pth):
        CT_ICF = sitk.ReadImage(pth + '/sitk_imgs/CT.mha')
        Seg = sitk.ReadImage(pth + '/sitk_imgs/seg.mha')
        cutting_plane = sitk.ReadImage(pth + '/sitk_imgs/cutting_plane.mha')
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
        sitk.WriteImage(removed_bone, pth + '/sitk_imgs/removed_bone.mha')
        sitk.WriteImage(remained_bone, pth + '/sitk_imgs/remained_bone.mha')
        return removed_bone, remained_bone

def make_binary_make(scan):
    '''
    This function make a binary mask from an input scan
    :param scan:
    :return:
    '''
    binary_threshold_filter = sitk.BinaryThresholdImageFilter()
    binary_threshold_filter.SetLowerThreshold(0)
    binary_threshold_filter.SetUpperThreshold(0)
    binary_threshold_filter.SetOutsideValue(1)
    binary_threshold_filter.SetInsideValue(0)
    binary_mask = binary_threshold_filter.Execute(scan)
    return binary_mask


class rotate_image:
    def resample(self,image, transform):
        reference_image = image
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)
    def rotate(self,image,degrees):

        affine = sitk.AffineTransform(3)
        radians = np.pi * degrees / 180.
        affine.Rotate(axis1=0, axis2=1, angle=radians)
        resampled = self.resample(image, affine)
        return resampled

if __name__ == "__main__":
    # read the CT and segmentation
    # A_CT = np.load(os.path.join(os.getcwd(), 'PatientA\\\\CT_ICF.npy'))#sahar
    A_CT = np.load('PatientA/CT_ICF.npy')
    # A_seg = np.load(os.path.join(os.getcwd(), 'PatientA\\\\Segmentation_ICF.npy'))#sahar
    A_seg = np.load('PatientA/Segmentation_ICF.npy')
    # B_CT = np.load(os.path.join(os.getcwd(), 'PatientB\\\\CT_ICF.npy'))#sahar
    B_CT = np.load('PatientB/CT_ICF.npy')
    # B_seg = np.load(os.path.join(os.getcwd(), 'PatientB\\\\Segmentation_ICF.npy'))#sahar
    B_seg = np.load('PatientB/Segmentation_ICF.npy')

    sitk_A_ct = sitk.GetImageFromArray(A_CT.astype(np.float32))
    sitk_A_seg = sitk.GetImageFromArray(A_seg.astype(np.uint8))

    sitk_B_ct = sitk.GetImageFromArray(B_CT.astype(np.float32))
    sitk_B_seg = sitk.GetImageFromArray(B_seg.astype(np.uint8))

    # instantiate from Biomarkers
    radiomic_features = ['FirstOrder', 'Shape', 'GLCM', 'GLRLM', 'GLSZM']
    biomarkers = Biomarkers()
    # extract biomarker features for patiant A and B
    A_features = biomarkers.extract_features(sitk_A_ct, sitk_A_seg, radiomic_features)
    B_features = biomarkers.extract_features(sitk_B_ct, sitk_B_seg, radiomic_features)



    # ANOVA feature selection for numeric input and categorical output
    no_extracted_features = 5
    data = [[i.item() if type(i) == np.ndarray else i for i in list(A_features.values())]]
    data.append(list([i.item() if type(i) == np.ndarray else i for i in list(A_features.values())]))
    label = np.array([0, 1])
    fs = SelectKBest(score_func=f_classif, k=no_extracted_features)
    X_selected = fs.fit_transform(data, label)
    selected_features = []
    for i in range(no_extracted_features):
        indx = np.argwhere(data[0] == X_selected[0][i])
        selected_features.append(list(A_features.keys())[indx[0][0]])
    print(selected_features)

    # ............................ Surgery plan ....................................
    # Patient A
    # instantiate from class SurgeryPlan
    surgery_plan = SurgeryPlan()
    # make binary mask for patient A
    pth = 'PatientA'
    # A_CT_ICF = np.loadtxt(pth + '/cutting_plane_RCF.txt')
    # ct_shape, A_scaled_translated_cutting_plane = surgery_plan.transform_cutting_plane(pth=pth, CT_ICF=A_CT_ICF)
    # A_cutting_plane = surgery_plan.binary_cutting_plane(shape=ct_shape, cutting_vertices=A_scaled_translated_cutting_plane)
    # sitk.WriteImage(sitk.GetImageFromArray(A_cutting_plane), pth + '/cutting_plane.mha')

    # get removed and remained bone for patient A
    A_removed_bone, A_remained_bone = surgery_plan.remove_remain_bone(pth=pth)


    # Patient B

    # make binary mask for patient B
    pth = 'PatientB'
    # B_CT_ICF = np.loadtxt(pth + '/cutting_plane_RCF.txt')
    # ct_shape, B_scaled_translated_cutting_plane = surgery_plan.transform_cutting_plane(pth=pth, CT_ICF=B_CT_ICF)
    # B_cutting_plane = surgery_plan.binary_cutting_plane(shape=ct_shape, cutting_vertices=B_scaled_translated_cutting_plane)
    # sitk.WriteImage(sitk.GetImageFromArray(B_cutting_plane), pth + '/cutting_plane.mha')

    # get removed and remained bone for patient B
    B_removed_bone, B_remained_bone = surgery_plan.remove_remain_bone(pth=pth)
    # use the biomarkers to decide if the removed and remained bones are diseased and healthy

    # Decision
    # features for removed bone of patient A
    A_removed_bone_binary_mask = make_binary_make(A_removed_bone)
    A_removed_features = biomarkers.extract_features(A_removed_bone, A_removed_bone_binary_mask, radiomic_features)

    # features for remained bone of patient A
    A_remained_bone_binary_mask = make_binary_make(A_remained_bone)
    A_remained_features = biomarkers.extract_features(A_remained_bone, A_remained_bone_binary_mask, radiomic_features)

    # features for removed bone of patient B
    B_removed_bone_binary_mask = make_binary_make(B_removed_bone)
    B_removed_features = biomarkers.extract_features(B_removed_bone, B_removed_bone_binary_mask, radiomic_features)
    # features for remained bone of patient B
    B_remained_bone_binary_mask = make_binary_make(B_remained_bone)
    B_remained_features = biomarkers.extract_features(B_remained_bone, B_remained_bone_binary_mask, radiomic_features)

    selected_features_A = [A_removed_features[f] for f in selected_features]
    print(np.corrcoef(selected_features_A, X_selected[0, :]))

    selected_features_B = [B_removed_features[f] for f in selected_features]
    print(np.corrcoef(selected_features_B, X_selected[0, :]))

    print(1)