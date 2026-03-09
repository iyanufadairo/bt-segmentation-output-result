import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import gdown
from scipy.ndimage import zoom

# Force CPU-only
os.environ['TF_ENABLE_ONEDNN_OPTS=0'] 

st.set_page_config(page_title="Glioma Segmentation", layout="wide")

scaler = MinMaxScaler()

MODEL_URL = "https://drive.google.com/uc?id=15DvYjyBHo-OgI-oVPrruocNr_WUauQEk"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "3D_unet_100_epochs_2_batch_patch_training.keras")
TARGET_SHAPE = (96, 96, 96, 4)

os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model download failed")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    try:
        tf.get_logger().setLevel('ERROR')
        # FIXED: use tf.keras instead of tensorflow.keras.models
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_uploaded_files(uploaded_files):
    modalities = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        try:
            img = nib.load(tmp_path)
            img_data = img.get_fdata()
            # FIXED: correct reshape for 3D volumes
            img_data = scaler.fit_transform(img_data.reshape(-1, 1)).reshape(img_data.shape)
            if 't1n' in file_name:
                modalities['t1n'] = img_data
            elif 't1c' in file_name:
                modalities['t1c'] = img_data
            elif 't2f' in file_name:
                modalities['t2f'] = img_data
            elif 't2w' in file_name:
                modalities['t2w'] = img_data
            elif 'seg' in file_name:
                modalities['mask'] = img_data.astype(np.uint8)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return modalities

def prepare_input(modalities):
    required = ['t1n', 't1c', 't2f', 't2w']
    if not all(m in modalities for m in required):
        return None, None, None
    combined = np.stack([
        modalities['t1n'],
        modalities['t1c'],
        modalities['t2f'],
        modalities['t2w']
    ], axis=3)
    combined = combined[56:184, 56:184, 13:141, :]
    original_shape = combined.shape
    downsampled = combined[::2, ::2, ::2, :]
    return downsampled, original_shape, combined

def make_prediction(model, input_data):
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data, verbose=0)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    return prediction_argmax

def upsample_prediction(prediction, target_shape):
    zoom_factors = (
        target_shape[0] / prediction.shape[0],
        target_shape[1] / prediction.shape[1],
        target_shape[2] / prediction.shape[2]
    )
    return zoom(prediction, zoom_factors, order=0)

def visualize_results(original_data, prediction, ground_truth=None):
    image_data = original_data[:, :, :, 1]
    slice_indices = [50, 75, 90]
    cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(3, cols, figsize=(10, 6))
    for i, slice_idx in enumerate(slice_indices):
        img_slice = np.rot90(image_data[:, :, slice_idx])
        pred_slice = np.rot90(prediction[:, :, slice_idx])
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Input - Slice {slice_idx}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred_slice)
        axes[i, 1].set_title(f'Prediction - Slice {slice_idx}')
        axes[i, 1].axis('off')
        if ground_truth is not None:
            gt_slice = np.rot90(ground_truth[:, :, slice_idx])
            axes[i, 2].imshow(gt_slice)
            axes[i, 2].set_title(f'Ground Truth - Slice {slice_idx}')
            axes[i, 2].axis('off')
    plt.tight_layout()
    return fig

def main():
    st.title("3D Glioma Segmentation with U-Net")
    st.write("Upload MRI scans in NIfTI format for glioma segmentation")

    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload **all four MRI modalities** (T1n, T1c, T2f, T2w) as NIfTI files (.nii.gz)
        2. Optionally upload a segmentation mask (filename must contain 'seg')
        3. Click 'Process and Predict'
        4. View and download results
        
        **Note:** First run downloads the model (~100MB). Runs on CPU.
        """)

    model = download_and_load_model()
    if model is None:
        st.error("Failed to load model.")
        return

    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=['nii', 'nii.gz'],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) >= 4:
        if st.button("Process and Predict"):
            with st.spinner("Processing files..."):
                modalities = process_uploaded_files(uploaded_files)
                input_data, original_shape, original_data = prepare_input(modalities)

                if input_data is None:
                    st.error("Missing modalities. Please upload T1n, T1c, T2f, and T2w.")
                    return

                ground_truth = modalities.get('mask', None)
                if ground_truth is not None:
                    ground_truth = ground_truth[56:184, 56:184, 13:141]
                    ground_truth[ground_truth == 4] = 3

            with st.spinner("Making prediction (may take a few minutes on CPU)..."):
                start_time = time.time()
                prediction = make_prediction(model, input_data)
                prediction = upsample_prediction(prediction, original_shape[:3])
                prediction = prediction.astype(np.int32)
                elapsed_time = time.time() - start_time

            st.success(f"Prediction completed in {elapsed_time:.2f} seconds")
            fig = visualize_results(original_data, prediction, ground_truth)
            st.pyplot(fig)

            st.subheader("Download Prediction")
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                pred_img = nib.Nifti1Image(prediction, affine=np.eye(4))
                nib.save(pred_img, tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    pred_data = f.read()
                os.unlink(tmp_file.name)

            st.download_button(
                label="Download Segmentation (NIfTI)",
                data=pred_data,
                file_name="glioma_segmentation.nii.gz",
                mime="application/octet-stream"
            )
    elif uploaded_files and len(uploaded_files) < 4:
        st.warning("Please upload all four modalities (T1n, T1c, T2f, T2w)")

if __name__ == "__main__":
    main()