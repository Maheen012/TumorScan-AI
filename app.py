import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from utils import (
    create_attention_map,
    extract_focus_region,
    create_localization_image,
    generate_pdf_report
)

# page config
st.set_page_config(
    page_title="TumorScan AI",
    layout="wide"
)

# load model
@st.cache_resource
def load_localization_model(model_path):

    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)

    return None


MODEL_FILE = "models/best_pro_model.keras"

# load model
model = load_localization_model(MODEL_FILE)

if model is None:

    st.error(
        f"could not locate model file at: {MODEL_FILE}"
    )

    st.stop()

st.title("TumorScan AI")

st.caption(
    "AI-Powered Tumor Predictor | Radiology Assistant"
)

st.markdown("---")

st.write(
    "Upload a brain MRI scan to analyze tumor classification "
    "and localization results."
)

# disclaimer
st.warning(
    "Disclaimer: This application is for educational and research "
    "purposes only and should not be used as a medical diagnosis tool."
)

st.sidebar.title("Upload Scan")

uploaded_file = st.sidebar.file_uploader(
    "Upload Brain MRI Scan",
    type=["png", "jpg", "jpeg"]
)

# main inference pipeline
if uploaded_file is not None:

    # preprocess image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img_bgr = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    img_resized = cv2.resize(
        img_rgb,
        (224, 224)
    )

    img_display = (
        img_resized.astype(np.float32) / 255.0
    )

    # run prediction
    with st.spinner("Running AI model analysis..."):

        img_tensor = tf.expand_dims(
            tf.convert_to_tensor(
                img_resized,
                dtype=tf.float32
            ),
            axis=0
        )

        predictions = model.predict(
            img_tensor,
            verbose=0
        )[0]

        CLASSES = [
            'glioma',
            'meningioma',
            'normal',
            'pituitary'
        ]

        pred_idx = np.argmax(predictions)

        confidence = predictions[pred_idx]

        predicted_class = CLASSES[pred_idx]

    # show metrics
    st.markdown("AI Model Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            label="Prediction",
            value=predicted_class.upper()
        )

    with metric_col2:

        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.2f}%"
        )

    st.markdown("---")

    # visualization section
    st.markdown("## MRI Visualization")

    col1, col2, col3 = st.columns(3)

    # original image
    with col1:

        st.markdown("### Original MRI Scan")

        fig_original, ax_original = plt.subplots(
            figsize=(5, 5)
        )

        ax_original.imshow(img_display)

        ax_original.axis('off')

        st.pyplot(
            fig_original,
            use_container_width=True
        )

        plt.close(fig_original)

    # normal case
    if predicted_class == 'normal':

        localization_image = img_resized.copy()

        with col2:

            st.markdown("### Attention Heatmap")

            fig_norm_att, ax_norm_att = plt.subplots(
                figsize=(5, 5)
            )

            ax_norm_att.imshow(img_display)

            ax_norm_att.axis('off')

            st.pyplot(
                fig_norm_att,
                use_container_width=True
            )

            plt.close(fig_norm_att)

            st.info(
                "No significant activation regions detected."
            )

        with col3:

            st.markdown("### Localization Result")

            fig_norm_loc, ax_norm_loc = plt.subplots(
                figsize=(5, 5)
            )

            ax_norm_loc.imshow(img_display)

            ax_norm_loc.axis('off')

            st.pyplot(
                fig_norm_loc,
                use_container_width=True
            )

            plt.close(fig_norm_loc)

            st.success(
                "No abnormal tumor regions detected."
            )

    # tumor case
    else:

        # create attention map
        attention_map = create_attention_map(
            model,
            img_tensor,
            pred_idx
        )

        # extract roi
        roi_data = extract_focus_region(
            attention_map
        )

        # create localization image
        localization_image = create_localization_image(
            img_resized,
            roi_data
        )

        # show attention map
        with col2:

            st.markdown("### Attention Heatmap")

            fig_att, ax_att = plt.subplots(
                figsize=(5, 5)
            )

            ax_att.imshow(
                attention_map,
                cmap='jet'
            )

            ax_att.axis('off')

            st.pyplot(
                fig_att,
                use_container_width=True
            )

            plt.close(fig_att)

            st.info(
                "Highlighted regions influenced the AI prediction."
            )

        # show localization image
        with col3:

            st.markdown("### Tumor Localization")

            fig_loc, ax_loc = plt.subplots(
                figsize=(5, 5)
            )

            ax_loc.imshow(localization_image)

            ax_loc.axis('off')

            st.pyplot(
                fig_loc,
                use_container_width=True
            )

            plt.close(fig_loc)

            st.success(
                "Potential tumor region localized."
            )

    # generate report
    st.markdown("---")

    report_pdf = generate_pdf_report(
        predicted_class,
        confidence,
        img_resized,
        localization_image
    )

    st.download_button(
        label="Download Diagnostic Report",
        data=report_pdf,
        file_name="tumorscan_ai_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

else:

    st.info(
        "Upload a brain MRI scan from the left sidebar to begin analysis."
    )