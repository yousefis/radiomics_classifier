import numpy as np
import matplotlib.pyplot as plt
import radiomics
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, featureextractor
import SimpleITK as sitk
import six


class Biomarkers:
    def print_features(self,features):
        for i in features.items():
            print(i)
        print("\n")


    def features_distance(self,features1, features2):
        for f1,f2 in zip(features1.items(),features2.items()):
            print(f1[0], f1[1]-f2[1])
        print("\n")


    def extract_features(self,image, mask, feature_list):
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


    def diff_features(self,features1, features2):
        for f1,f2 in zip(features1.items(),features2.items()):
            print(f1[0], f1[1]-f2[1])
        print("\n")

def medical_image_viewer(volume):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.flip(volume[volume.shape[0]//2, :, :].T, 0), cmap="Greys_r")
    ax1.title.set_text('Sagital')
    ax1.axis('Off')

    ax2.imshow(np.flip(volume[:, volume.shape[1]//2, :].T, 0), cmap="Greys_r")
    ax2.title.set_text('Coronal')
    ax1.axis('Off')
    plt.axis('Off')



if __name__ == "__main__":
    # read the CT and segmentation
    # A_CT = np.load(os.path.join(os.getcwd(), 'PatientA\\\\CT_ICF.npy'))#sahar
    A_CT = np.load('PatientA/CT_ICF.npy')
    # A_seg = np.load(os.path.join(os.getcwd(), 'PatientA\\\\Segmentation_ICF.npy'))#sahar
    A_seg = np.load( 'PatientA/Segmentation_ICF.npy')
    # B_CT = np.load(os.path.join(os.getcwd(), 'PatientB\\\\CT_ICF.npy'))#sahar
    B_CT = np.load('PatientB/CT_ICF.npy')
    # B_seg = np.load(os.path.join(os.getcwd(), 'PatientB\\\\Segmentation_ICF.npy'))#sahar
    B_seg = np.load('PatientB/Segmentation_ICF.npy')

    sitk_A_ct = sitk.GetImageFromArray(A_CT.astype(np.float32))
    sitk_A_seg = sitk.GetImageFromArray(A_seg.astype(np.uint8))

    sitk_B_ct = sitk.GetImageFromArray(B_CT.astype(np.float32))
    sitk_B_seg = sitk.GetImageFromArray(B_seg.astype(np.uint8))

    # instantiate from Biomarkers
    biomarkers = Biomarkers()
    # extract biomarker features for patiant A and B
    A_features = biomarkers.extract_features(sitk_A_ct, sitk_A_seg, ['FirstOrder', 'Shape', 'GLCM', 'GLRLM', 'GLSZM'])
    B_features = biomarkers.extract_features(sitk_B_ct, sitk_B_seg, ['FirstOrder', 'Shape', 'GLCM', 'GLRLM', 'GLSZM'])
    # calculate the distance between the features
    biomarkers.diff_features(A_features,B_features)