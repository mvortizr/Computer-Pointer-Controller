# Computer Pointer Controller

Computer Pointer Controller is an application that moves the mouse pointer according to the movement of the eyes of a person. The application can take as an input a video file or the webcam, then it uses 4 different inference models to detect the eyes and head pose of a person. 
This application is powered by the inference engine of the Intel OpenVINO Toolkit. 


## Project Set Up and Installation

__Requirements__
* OpenVINO Toolkit 2020.1.023
* Python3
* Pip


__Step 1: Download the OpenVINO Toolkit__
This project requires the OpenVINO Toolkit that is available [here](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html). The version used in this project is the 2020.1.023. This [guide](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) can guide you with the installation process in different operating systems.

__Step 2: Clone this repository__
Assuming you have git installed on your target device, you can use this command:
```
git clone https://github.com/mvortizr/Computer-Pointer-Controller
```
Otherwise, you can simply select the option 'Download Zip' in the green button on the right.

__Step 3: Initialize the OpenVINO environment__
Run the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
__Step 4: Download the required models__

This project requires 4 different models, the models should be placed in a folder called `models`. 

These model can be downloaded using the Model Downloader from the OpenVINO toolkit. For this, you need to enter the following commands:

Change the directory to the one that has the Model Downloader:
```
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
```
To download the Face Detection Model:
```
sudo ./downloader.py --name face-detection-adas-binary-0001 -o $HOME/Documents/DevProjects/Computer-Pointer-Controller/models
```
To download the Head Pose Estimation Model:
```
sudo ./downloader.py --name head-pose-estimation-adas-0001 -o $HOME/Documents/DevProjects/Computer-Pointer-Controller/models
```
To download the Facial Landmarks Detection Model:
```
sudo ./downloader.py --name landmarks-regression-retail-0009 -o $HOME/Documents/DevProjects/Computer-Pointer-Controller/models
```
To download the Gaze Estimation Model
```
sudo ./downloader.py --name gaze-estimation-adas-0002 -o $HOME/Documents/DevProjects/Computer-Pointer-Controller/models
```

__Step 5: Create a Python Environment__
This project requires a virtual environment, for this you will need to install the package `virtualenv` using the following command:

```
 pip3 install virtualenv
```
Then, you will need to create a new python3 virtualenv using the following command:
```
virtualenv -p /usr/bin/python3 env 
```
After that, you will need to source the new enviroment, using the following command:

```
source env/bin/activate
```

Then install all the dependencies of this project by running:
```
pip install -r requirements.txt
```

## Demo

Command to run the application on the CPU
```
python src/main.py -f models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -l models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d CPU 
```

Command to run the application on the CPU and show the outputs of the models
```
python src/main.py -f models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -l models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d CPU -o FLM HPEM GEM
```

## Documentation

__Arguments__
The entrypoint of the application is the file `/src/main.py` you need to pass the following arguments:


| Argument     | Required           | Usage  |
| :------------- |:-------------:| -----:|
| -f or --face_detection_model     | yes| Path to .xml file of the face detection model |
| -l or --facial_landmarks_detection_model    | yes    |   Path to .xml file of the facial landmarks detection model |
| -hp or --head_pose_estimation_model | yes  |    Path to .xml file of the head pose estimation model |
| -g or --gaze_estimation_model | yes  |    Path to .xml file of the gaze estimation model |
| -i or --input | yes  |Path to video file or enter 'cam' to work with webcam |
| -t or --threshold | no  |Probability threshold for the face detection model. |
| -o or --outputs | no  | See outputs of selected models. Valid inputs: FLM, HPEM and GEM |
| -d or --device | no  | Target device to infer. Valid inputs: CPU, GPU, FPGA or MYRIAD. |
| -ce or --cpu_extension | no  | Earlier versions of OpenVINO require the path to the CPU extension |

__Models Used__

* [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html): Used to isolate the face from the background.
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html): To detect the pose of the head. 
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html): To detect the left and right eyes.
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html): To detect the gaze of the person.

__The Pipeline__

First,  the input stream from the camera or video file goes into the face detection model.  Then, the output from this model feeds the landmark detection model and the head pose estimation model. After that, the outputs from both models feed the gaze estimation model that produces the coordinates needed to move the mouse pointer. This architecture is illustrated on the following image:

--Add image of the architecture---


__File Structure__

--Add file structure image--

- `bin` contains a video file called `demo.mp4` to test the application
- `requirements.txt` contains all the dependencies required to run the applicatio
-   `src` folder contains all the source files of the application:
    
    1.  face_detection.py
        
        -   Contains the inference engine setup to run the Face Detection model, including the preprocessing of inputs and outputs
    
    2.  facial_landmarks_detection.py
        
        -   Contains the inference engine setup to run the Facial Landmarks Detection model, including the preprocessing of inputs and outputs
       
    3.  head_pose_estimation.py
        
        -   Contains the inference engine setup to run the Head Pose Estimation model, including the preprocessing of inputs and outputs
    
    4.  gaze_estimation.py
        
        -   Contains the inference engine setup to run the Gaze Estimation model, including the preprocessing of inputs and outputs
        
    5.  input_feeder.py
  
        -   Contains InputFeeder class that handles the input from the webcam or the video file
    6.  mouse_controller.py
        
        -   Contains MouseController class that handles the movement of the mouse pointer
   
    7.  main.py
  
        -   The entrypoint of the application and the file that handles all the workflow



## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

