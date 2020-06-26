# TB-Prediction-App-Capstone
This is a system that predicts pulmonary tuberculosis using Image Processing and Deep Learning.This system performs thresholding, Histogram Equalization and ROI segmentation and train the data over a neural network that predicts the positive/negative case of Tuberculosis. It  indicates an improvement from the existing systems with the help of image processing which is justified in the metrics the system achieves, while studying high definition X-ray scan images. The  diagnostic system achieves 91.25% accuracy, comparable to the performance of expert radiologists. It also gains sensitivity measure of 94% and specificity rate of 88% measure.

# To use this project
All you have to do to run this project is to clone this directory and run the `predict.py` file.<br />
`python predict.py`

## File Specifications
[tbpredict.py](https://github.com/anand498/TB-Prediction-App-Capstone/blob/master/predict.py):
This is the main and the only script to run for the prediction website to start running on the localhost server.

[Tuberculosis.ipynb](https://github.com/anand498/TB-Prediction-App-Capstone/blob/master/Tuberculosis.ipynb)

To download the required libraries: 
`pip install -r requirements.txt ` <br />
### Dataset used
All implementations related to this paper were tested in the public datasets Shenzen and Montgomery in. The Montgomery dataset was composed over many years within the Tuberculosis control program of the Department of Health and Human Services of Montgomery County. This dataset set contains 138 posteroanterior CXRs, among which 80 CXRs are normal, while the remaining 58 CXRs are abnormal. The X-Rays were captured in Portable Network Graphics (PNG) format as 12-bit gray images. The size of the images provided in the dataset is either (4020 X 4892) or (4892 X 4020) pixels.

### Citation
The project was done as a Final Year Project. The paper has been sent to Elsevier Journals. The citation link and DOI will be added as soon as the paper is published.

