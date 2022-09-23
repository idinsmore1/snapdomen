import argparse
import json
import os
import keras.backend as K
import matplotlib.pyplot as plt

from snapdomen.imaging.dicomseries import DicomSeries
from snapdomen.liver.measure import measure_liver_hu, measure_spleen_hu
from snapdomen.vertebra.model import load_model
from snapdomen.vertebra.detection import detect_vertebra, plot_vertebra_detection
from snapdomen.abfat.quant import quantify_abdominal_fat

parser = argparse.ArgumentParser(description='Snapdomen CLI - Metabolic Snapshot from CT Scan')
parser.add_argument('--input', type=str, help='The path to the input dicom series')
parser.add_argument('--input-pattern', type=str, help='The pattern of the input dicom series', default='*')
parser.add_argument('--output-dir', type=str, help='The path to the output directory')
parser.add_argument('--liver-weights', type=str, help='The path to the liver segmentation model weights')
parser.add_argument('--roi-area', default=5, type=int, help='The area of the roi in cm^2 (default: 5)')
parser.add_argument('--roi-alpha', default=0.55, type=float, help='The alpha value for the liver roi (default: 0.55)')
parser.add_argument('--spleen-weights', type=str, help='The path to the spleen segmentation model weights')
parser.add_argument('--vertebrae-weights', type=str, help='The path to the vertebrae segmentation model weights, comma separated')
parser.add_argument('--vertebrae-names', type=str, help='The names of the vertebrae, comma separated')
parser.add_argument('--vertebrae-threshold', default=0.1, type=float, help='The threshold for the vertebrae segmentation (default: 0.1)')
parser.add_argument('--abdomen-weights', type=str, help='The path to the abdomen segmentation model weights')
args = parser.parse_args()

def main():
    """
    Main CLI runner that combines the effects of all the applications in snapdomen\n
    :return: None
    """
    # Read in the dicom series
    dicom_series = DicomSeries(args.input, args.input_pattern)
    quant_data = dicom_series.series_info
    output_directory = f'{args.output_dir}/MRN{dicom_series.mrn}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Measure the liver
    liver_data = measure_liver_hu(dicom_series, args.liver_weights, output_directory, args.roi_area, args.roi_alpha)
    quant_data.update(liver_data)
    # Measure the spleen
    spleen_data = measure_spleen_hu(dicom_series, args.spleen_weights, roi_area=args.roi_area, alpha=args.roi_alpha)
    quant_data.update(spleen_data)

    # Detect vertebrae locations
    vertebrae_weights = args.vertebrae_weights.split(',')
    vertebrae_names = args.vertebrae_names.split(',')
    vertebra_model = load_model('snapdomen/vertebra/CNNLine.json')

    del dicom_series
    dicom_series = DicomSeries(args.input, args.input_pattern, 40, 1200)
    vertebrae_info = detect_vertebra(dicom_series.frontal, vertebra_model, vertebrae_weights, vertebrae_names, dicom_series.spacing, 1)
    # Plot the vertebrae detection
    plot_vertebra_detection(dicom_series.frontal, vertebrae_info, vertebrae_names, output_directory)
    quant_data.update(vertebrae_info)

    start, end = vertebrae_info['l1_slice'], vertebrae_info['l5_slice']
    # Detect the abdomen
    fat_measurements = quantify_abdominal_fat(dicom_series, start, end, args.abdomen_weights)
    quant_data['abdominal_fat'] = fat_measurements
    json.dump(quant_data, open(f'{output_directory}/MRN{dicom_series.mrn}_quant.json', 'w'))


if __name__ == '__main__':
    main()
