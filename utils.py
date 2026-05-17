import io
import cv2
import numpy as np
import tensorflow as tf

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# create attention heatmap
def create_attention_map(model, img_tensor, pred_idx):

    with tf.GradientTape() as tape:

        tape.watch(img_tensor)

        loss = model(img_tensor, training=False)[:, pred_idx]

    gradients = tape.gradient(loss, img_tensor)

    saliency = tf.reduce_max(
        tf.abs(gradients),
        axis=-1
    )[0].numpy()

    saliency_blurred = cv2.GaussianBlur(
        saliency,
        (7, 7),
        0
    )

    s_min = saliency_blurred.min()

    s_max = saliency_blurred.max()

    saliency_norm = (
        saliency_blurred - s_min
    ) / (s_max - s_min + 1e-7)

    return saliency_norm


# extract strongest activation region
def extract_focus_region(attention_map):

    high_focus_thresh = np.percentile(
        attention_map,
        99.7
    )

    high_attention_pixels = np.argwhere(
        attention_map >= high_focus_thresh
    )

    if len(high_attention_pixels) == 0:
        return None

    ymin, xmin = high_attention_pixels.min(axis=0)

    ymax, xmax = high_attention_pixels.max(axis=0)

    box_width = xmax - xmin

    box_height = ymax - ymin

    padding = 3

    xmin_pad = max(0, xmin - padding)

    ymin_pad = max(0, ymin - padding)

    width_pad = min(
        224 - xmin_pad,
        box_width + (padding * 2)
    )

    height_pad = min(
        224 - ymin_pad,
        box_height + (padding * 2)
    )

    return (
        xmin_pad,
        ymin_pad,
        width_pad,
        height_pad
    )


# draw localization box
def create_localization_image(
    original_image,
    roi_data
):

    localization_image = original_image.copy()

    if roi_data is None:
        return localization_image

    xmin, ymin, width, height = roi_data

    cv2.rectangle(
        localization_image,
        (xmin, ymin),
        (xmin + width, ymin + height),
        (0, 255, 0),
        3
    )

    return localization_image


# generate pdf report
def generate_pdf_report(
    predicted_class,
    confidence,
    original_image,
    localization_image
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    # add title
    title = Paragraph(
        "<b>TumorScan AI Diagnostic Report</b>",
        styles['Title']
    )

    story.append(title)

    story.append(Spacer(1, 20))

    # add results table
    data = [
        ["AI Model Prediction", predicted_class.upper()],
        ["Confidence Score", f"{confidence * 100:.2f}%"]
    ]

    table = Table(
        data,
        colWidths=[220, 220]
    )

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))

    story.append(table)

    story.append(Spacer(1, 25))

    # add original image
    story.append(
        Paragraph(
            "<b>Original MRI Scan</b>",
            styles['Heading2']
        )
    )

    original_rgb = original_image.astype(np.uint8)

    _, original_encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(
            original_rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    original_buffer = io.BytesIO(
        original_encoded.tobytes()
    )

    story.append(
        Image(
            original_buffer,
            width=250,
            height=250
        )
    )

    story.append(Spacer(1, 25))

    # add localization image
    if localization_image is not None:

        story.append(
            Paragraph(
                "<b>Tumor Localization Result</b>",
                styles['Heading2']
            )
        )

        localization_rgb = localization_image.astype(np.uint8)

        _, localization_encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(
                localization_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        localization_buffer = io.BytesIO(
            localization_encoded.tobytes()
        )

        story.append(
            Image(
                localization_buffer,
                width=250,
                height=250
            )
        )

        story.append(Spacer(1, 25))

    # add disclaimer
    disclaimer = Paragraph(
        "<b>Disclaimer:</b> "
        "This system is for educational and research purposes only. "
        "it should not be considered a medical diagnosis.",
        styles['BodyText']
    )

    story.append(disclaimer)

    # build pdf
    doc.build(story)

    buffer.seek(0)

    return buffer