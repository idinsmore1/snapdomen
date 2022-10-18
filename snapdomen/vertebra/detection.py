import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import keras.backend as K


def normalize_zero_one(image, eps=1e-8):
    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) - np.min(image) + eps)

    return ret


def reduce_hu_intensity_range(img, minv=100, maxv=1500):
    img = np.clip(img, minv, maxv)
    img = 255 * normalize_zero_one(img)

    return img


def pad_image_to_size(image, img_size=(64, 64, 64), loc=(2, 2, 2), **kwargs):
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    to_padding = [[0, 0] for _ in range(rank)]

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i] < img_size[i]:
            if loc[i] == 0:
                to_padding[i][0] = (img_size[i] - image.shape[i])
                to_padding[i][1] = 0
            elif loc[i] == 1:
                to_padding[i][0] = 0
                to_padding[i][1] = (img_size[i] - image.shape[i])
            else:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2 + (img_size[i] - image.shape[i]) % 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            to_padding[i][0] = 0
            to_padding[i][1] = 0
    return np.pad(image, to_padding, **kwargs)


def preprocess_for_detection(image, spacing, target_spacing, min_height=512, min_width=512):
    image = zoom(image, [spacing[2] / target_spacing, spacing[0] / target_spacing])
    image = reduce_hu_intensity_range(image)

    v = min_height if image.shape[0] <= min_height else 2 * min_height
    img_size = [v, min_width]
    padded_image = pad_image_to_size(image, img_size, loc=[1, -1], mode='constant')
    padded_image = padded_image[:v, :min_width] - 128
    return padded_image[np.newaxis, :, :, np.newaxis], image


def rescale_prediction(prediction, original_spacing):
    probability = np.max(prediction[0])
    slice_n = np.argmax(prediction[0])
    slice_n = np.int16(slice_n // original_spacing[2])
    return slice_n, probability


def detect_vertebra(frontal_mip, model, vertebra_weights, vertebra_names, spacing, target_spacing):
    image, _ = preprocess_for_detection(frontal_mip, spacing, target_spacing)
    vertebra_info = {}
    for weight, name in zip(vertebra_weights, vertebra_names):
        model.load_weights(weight)
        prediction = model.predict(image)
        slice_n, probability = rescale_prediction(prediction, spacing)
        vertebra_info[f'{name}_slice'] = slice_n.tolist()
        vertebra_info[f'{name}_probability'] = probability.tolist()
    K.clear_session()
    return vertebra_info


def plot_vertebra_detection(series, vertebra_info, vertebra_names, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(series.frontal)
    for name, color in zip(vertebra_names, ['r', 'g', 'y']):
        ax.axhline(vertebra_info[f'{name}_slice'], color=color)
        ax.text(0, vertebra_info[f'{name}_slice'], f'{name} {vertebra_info[f"{name}_probability"]:.2f}', color='r')
    plt.savefig(output_dir + f'/MRN{series.mrn}_{series.accession}_{series.cut}_vertebrae_overlay.png')


def save_overlay(dicom_series, slice_loc, output_dir):
    fig = plt.imshow(dicom_series.frontal)
    plt.axhline(slice_loc, color='r')
    plt.savefig(output_dir + f'/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_l3_overlay.png')