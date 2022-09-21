# Snapdomen
python pipeline for taking a "metabolic snapshot" of a patient through a non-contrast CT scan.

## Goals
1. Segment Liver and quantify the amount of fat within the liver tissue using the deep learning library Keras
2. Automatically detect the location of various vertebrae in the spine
3. Measure visceral and abdominal fat area, as well as the waist circumference from the L1 through L5 vertebrae for each axial image slice
4. Output summary data into a json file that can be used for further analysis

## Installation
1. Clone the repository
2. Install the requirements using `pip install -r requirements.txt`
3. use run.py to run the pipeline with specified parameters