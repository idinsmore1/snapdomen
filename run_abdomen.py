import argparse
import json
import os
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from snapdomen.imaging.dicomseries import DicomSeries
# from snapdomen.liver.measure import measure_liver_hu, measure_spleen_hu
from snapdomen.vertebra.model import load_model
from snapdomen.vertebra.detection import detect_vertebra, plot_vertebra_detection
from snapdomen.abfat.quant import quantify_abdominal_fat

parser = argparse.ArgumentParser(description='Snapdomen CLI - Metabolic Snapshot from CT Scan')
parser.add_argument('--input', type=str, help='The path to the input dicom series')
parser.add_argument('--input-pattern', type=str, help='The pattern of the input dicom series', default='*')
parser.add_argument('--output-dir', type=str, help='The path to the output directory')
# parser.add_argument('--liver-weights', type=str, help='The path to the liver segmentation model weights')
# parser.add_argument('--roi-area', default=5, type=int, help='The area of the roi in cm^2 (default: 5)')
# parser.add_argument('--roi-alpha', default=0.55, type=float, help='The alpha value for the liver roi (default: 0.55)')
# parser.add_argument('--spleen-weights', type=str, help='The path to the spleen segmentation model weights')
parser.add_argument('--vertebrae-weights', type=str,
                    help='The path to the vertebrae segmentation model weights, comma separated')
parser.add_argument('--vertebrae-names', type=str, help='The names of the vertebrae, comma separated')
parser.add_argument('--vertebrae-threshold', default=0.1, type=float,
                    help='The threshold for the vertebrae segmentation (default: 0.1)')
parser.add_argument('--abdomen-weights', type=str, help='The path to the abdomen segmentation model weights')
parser.add_argument('--gpu', type=str, help='The GPU to use', default=None)
parser.add_argument('--write-file-names', action='store_true', help='Write the completed file names to a log file')
args = parser.parse_args()


def main():
    """
    Main CLI runner that combines the effects of all the applications in snapdomen\n
    :return: None
    """
    plt.ioff()
    # set the GPU to use
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Detect vertebrae locations
    vertebrae_weights = args.vertebrae_weights.split(',')
    vertebrae_names = args.vertebrae_names.split(',')
    vertebra_model = load_model('snapdomen/vertebra/CNNLine.json')

    # del dicom_series
    dicom_series = DicomSeries(args.input, args.input_pattern, 40, 1200)
    quant_data = dicom_series.series_info
    del_keys = []
    for k in quant_data.keys():
        if k not in ['mrn', 'accession', 'cut', 'scan_date']:
            del_keys.append(k)
    for k in del_keys:
        del quant_data[k]
    output_directory = f'{args.output_dir}/MRN{dicom_series.mrn}'
    print('Detecting vertebrae locations...')
    vertebrae_info = detect_vertebra(dicom_series.frontal, vertebra_model, vertebrae_weights, vertebrae_names,
                                     dicom_series.spacing, 1)
    # Plot the vertebrae detection
    plot_vertebra_detection(dicom_series, vertebrae_info, vertebrae_names, output_directory)
    quant_data.update(vertebrae_info)
    print('Vertebrae detection completed!\n')

    start, end, l3 = vertebrae_info['l1_slice'], vertebrae_info['l5_slice'], vertebrae_info['l3_slice']
    # Detect the abdomen
    print('Measuring abdominal fat...')
    try:
        dicom_series.pixel_array = np.clip(dicom_series.pixel_array, -500, 500)
        fat_measurements = quantify_abdominal_fat(dicom_series, start, end, l3, args.abdomen_weights, output_directory)
        quant_data['abdominal_fat'] = fat_measurements
    except Exception as e:
        print(f'Abdominal fat did not measure properly {e}')

    json.dump(quant_data,
              open(
                  f'{output_directory}/MRN{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_abdominal_fat.json',
                  'w'))
    if args.write_file_names:
        with open(f'{args.output_dir}/completed.txt', 'a') as f:
            f.write(f'{args.input}\n')

    print('Quantification finished!')


if __name__ == '__main__':
    main()
