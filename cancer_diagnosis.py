import sys

model = tf.keras.models.load_model("Brain Cancer - MobileNetV3.h5")
class_names = ['all_benign', 'all_early', 'all_pre', 'all_pro']

def predict_cancer_type(cancer_type):
    # Check if the provided cancer_type is in the list of class names from the training dataset
    if cancer_type in class_names:
        return cancer_type
    else:
        return "Wrong Input: The provided cancer type is not valid."

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python cancer_diagnosis.py <cancer_type>")
    else:
        cancer_type = sys.argv[1]
        result = predict_cancer_type(cancer_type)
        print(result)
