import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load your trained model
my_model = tf.keras.models.load_model(
    r"C:\Users\HP\Desktop\BT Segmentation\saved_model\3D_unet_100_epochs_2_batch_patch_training.keras",
    compile=False
)
print("Model loaded successfully!")

# Initialize scaler
scaler = MinMaxScaler()

# Function to preprocess NIfTI files (4 modalities)
def preprocess_nifti(t1c_path, t1n_path, t2f_path, t2w_path, patch_size=(96, 96, 96)):
    
    def load_and_scale(path):
        img = nib.load(path).get_fdata()
        img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
        return img

    # Load all 4 modalities
    t1c = load_and_scale(t1c_path)
    t1n = load_and_scale(t1n_path)
    t2f = load_and_scale(t2f_path)
    t2w = load_and_scale(t2w_path)

    # Combine into (x, y, z, 4)
    combined = np.stack([t1n, t1c, t2f, t2w], axis=3)

    # Crop to (128, 128, 128, 4)
    combined = combined[56:184, 56:184, 13:141, :]

    # Downsample to (96, 96, 96, 4) for model input
    combined = combined[16:112, 16:112, 16:112, :]

    # Add batch dimension → (1, 96, 96, 96, 4)
    combined = np.expand_dims(combined, axis=0)

    return combined

# Function to run segmentation and save result
def run_segmentation(model, t1c, t1n, t2f, t2w, output_folder):
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print("Preprocessing input files...")
    input_image = preprocess_nifti(t1c, t1n, t2f, t2w)

    print("Running prediction (may take a few minutes)...")
    prediction = model.predict(input_image, verbose=1)
    prediction_argmax = np.argmax(prediction, axis=-1)[0, :, :, :]

    # Save output
    output_file = os.path.join(output_folder, "BraTS-SSA-00008-000_segmentation.nii.gz")
    nib.save(
        nib.Nifti1Image(prediction_argmax.astype(np.float32), np.eye(4)),
        output_file
    )
    print(f"Segmentation saved to: {output_file}")

# Your file paths - all filled in!
t1c = r"C:\Users\HP\Desktop\BT Segmentation\BraTS-Africa\95_Glioma\BraTS-SSA-00008-000\BraTS-SSA-00008-000-t1c.nii.gz"
t1n = r"C:\Users\HP\Desktop\BT Segmentation\BraTS-Africa\95_Glioma\BraTS-SSA-00008-000\BraTS-SSA-00008-000-t1n.nii.gz"
t2f = r"C:\Users\HP\Desktop\BT Segmentation\BraTS-Africa\95_Glioma\BraTS-SSA-00008-000\BraTS-SSA-00008-000-t2f.nii.gz"
t2w = r"C:\Users\HP\Desktop\BT Segmentation\BraTS-Africa\95_Glioma\BraTS-SSA-00008-000\BraTS-SSA-00008-000-t2w.nii.gz"
output_folder = r"C:\Users\HP\Desktop\BT Segmentation\output_folder"

# Run it!
run_segmentation(my_model, t1c, t1n, t2f, t2w, output_folder)