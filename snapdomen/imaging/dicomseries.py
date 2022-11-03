import logging
import os
from glob import glob
from datetime import datetime
import natsort
import numpy as np
import pydicom
# from numba import njit
from .utils import transform_to_hu, window_image


class DicomSeries:
    """
    A class representing a series of dicom files in a specified directory\n

    """
    orientations = {
        'COR': [1, 0, 0, 0, 0, -1],
        'SAG': [0, 1, 0, 0, 0, -1],
        'AX': [1, 0, 0, 0, 1, 0]
    }
    important_attributes = {
        'PatientID': 'mrn',
        'AccessionNumber': 'accession',
        'SeriesDescription': 'cut',
        'ImageOrientationPatient': 'ct_direction',
        'PatientSex': 'sex',
        'PatientBirthDate': 'birthday',
        'AcquisitionDate': 'scan_date',
        'PatientAge': 'age_at_scan',
        'PixelSpacing': 'pixel_spacing',
        'SliceThickness': 'slice_thickness_mm',
        'Manufacturer': 'manufacturer',
        'ManufacturerModelName': 'manufacturer_model',
        'KVP': 'kvp',
        'ContrastBolusAgent': 'contrast_bolus_agent',
        'ContrastBolusRoute': 'contrast_bolus_route',
        'MultienergyCTAcquisitionSequence': 'multienergy_ct'
    }

    def __init__(self, directory, filepattern='*.dcm', window_center=30, window_width=150, read_images=True):
        """
        :param directory: the directory containing the series
        :param filepattern: the file pattern of the series
        :param window_center: the center of the hounsfield units to view the images (default: 30)
        :param window_width: the width of the hounsfield units to view the images (default: 150)
        :param read_images: a boolean indicating whether to read the images or not (default: True)
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f'Given directory does not exist or is not a file: {directory}')

        self.directory = os.path.abspath(directory)
        self.header = pydicom.dcmread(glob(os.path.join(directory, filepattern))[0])
        self.series_info = self._get_image_info(self.header)
        self.mrn = self.series_info['mrn']
        self.accession = self.series_info['accession']
        self.cut = self.series_info['cut']
        if read_images:
            self.pixel_array = self._read_dicom_series(directory, filepattern, self.header.RescaleSlope,
                                                       self.header.RescaleIntercept, window_center, window_width)
        self.bone_array = self._read_dicom_series(directory, filepattern, self.header.RescaleSlope,
                                                       self.header.RescaleIntercept, 1000, 3000)
        self.frontal = self._get_mip(self.bone_array, axis=1)
        self.sagittal = self._get_mip(self.bone_array, axis=2)
        self.spacing = [self.header.PixelSpacing[0], self.header.PixelSpacing[1], self.header.SliceThickness]

    # @staticmethod
    def _read_dicom_series(self, directory, filepattern, slope, intercept, window_center, window_width):
        """
        A function to read a series of dicom files into one numpy array \n
        :param directory: directory containing the files
        :param filepattern: filepattern to match dicom files
        :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
        :param intercept: the intercept value from the dicom file (depends on the machine)
        :param window_center: hounsfield window center
        :param window_width: hounsfield window width
        :return: a numpy array containing the dicom series
        """
        logging.info(f'Reading Dicom files from {directory}...')
        files = natsort.natsorted(glob(os.path.join(directory, filepattern)))
        logging.info(f'- Number of files: {len(files)}')

        # create an empty dictionary to store image and image number
        ct_scans = {}
        # read all the files
        logging.info(f'- Reading files...')
        for i, file in enumerate(files):
            ds = pydicom.dcmread(file)
            # get the image number
            if hasattr(ds, 'InstanceNumber'):
                image_number = int(ds.InstanceNumber)
            else:
                image_number = i
            image = ds.pixel_array
            # store the image and number in the dictionary
            ct_scans[image_number] = image

        # sort the images by image number
        logging.info(f'- Sorting images by image number...')
        sorted_ct = [ct_scans[key] for key in sorted(ct_scans.keys())]
        # stack the images into one array
        combined_image = np.stack(sorted_ct, axis=0)
        # Convert to hounsfield units
        logging.info(f'- Converting to hounsfield units...')
        hu_image = transform_to_hu(combined_image, slope, intercept)
        # Window the hounsfield units
        logging.info(f'- Windowing hounsfield units...')
        windowed_image = window_image(hu_image, window_center, window_width)
        # return the windowed image
        logging.info(f'- Done!\n')
        return windowed_image

    def _get_image_info(self, header):
        """
        A function to get the important information from the dicom header \n
        :param header: the dicom header object
        :return: a dictionary containing the important information
        """
        series_info = {}
        # Loop through all the important attributes
        for tag, column in self.important_attributes.items():
            try:
                value = getattr(header, tag)
                if tag == "SeriesDescription":
                    value = value.replace('/', '_').replace(' ', '_').replace('-', '_').replace(
                        ',', '.')
                elif tag == "ImageOrientationPatient":
                    orientation = np.round(getattr(header, tag))
                    for key, direction in self.orientations.items():
                        if np.array_equal(orientation, direction):
                            value = key
                            break
                # elif tag == 'PatientBirthDate':
                #     value = datetime.strptime(value, '%Y%m%d').date()
                #     value = value.strftime('%Y-%m-%d')
                elif tag == 'AcquisitionDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'PatientAge':
                    value = int(value[:-1])
                elif tag == 'PixelSpacing':
                    # Value in this case is a list
                    length, width = value
                    series_info['length_mm'] = length
                    series_info['width_mm'] = width
                    continue
                elif tag == 'PatientSex':
                    if value == 'M':
                        value = 1
                    elif value == 'F':
                        value = 0
                elif tag == 'ContrastBolusAgent' or tag == 'ContrastBolusRoute':
                    value = True
                elif tag == 'MultienergyCTAcquisitionSequence':
                    value = True

                series_info[column] = value

            except AttributeError:
                logging.info(f'{tag} not found in dicom header')
                series_info[column] = None

        return series_info

    @staticmethod
    def _get_mip(pixel_array, axis):
        """
        A method to get the Maximum Intensity Projection (MIP) of the dicom pixel array along an axis \n
        :param pixel_array: the dicom pixel array
        :param axis: the axis to project along
        :return: the frontal MIP
        """
        # Get the frontal MIP
        mip = np.amax(pixel_array, axis=axis)
        return mip
