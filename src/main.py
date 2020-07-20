import cv2
import os
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to .xml file of the face detection model")

    parser.add_argument("-l", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Path to .xml file of the facial landmarks detection model")

    parser.add_argument("-hp", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to .xml file of the head pose estimation model")

    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to .xml file of the gaze estimation model")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter 'cam' to work with webcam")

    parser.add_argument("-t", "--threshold", required=False, type=float, default=0.6,
                        help="Probability threshold for the face detection model.")

    parser.add_argument("-o", "--output", required=False, type=str,nargs='+', default=[],
                        help="See outputs of selected models")
    
    parser.add_argument("-d", "--device", required=False,type=str, default="CPU",
                        help="Target device to infer on: "
                             "Valid inputs: CPU, GPU, FPGA or MYRIAD.")

    parser.add_argument("-ce", "--cpu_extension", required=False,type=str, default=None,
                        help="Path to the CPU extension")
    return parser



def check_source(filepath):
	
	feeder = None

	if filepath.lower()=='cam':
		feeder = InputFeeder('cam')
	else:
		if not os.path.isfile(filepath):
			print('[ERROR] Video file does not exist')
			exit(1)

		feeder = InputFeeder('video',filepath)

	return feeder 


def init_models(args):

	face_detec=None
	fac_land=None
	head_pose=None
	gaze_est=None
	
	models_info = {'face_detection_model': args.face_detection_model, 
	'facial_landmarks_detection_model': args.facial_landmarks_detection_model, 
    'head_pose_estimation_model': args.head_pose_estimation_model, 
    'gaze_estimation_model': args.gaze_estimation_model}

	for model_name in models_info.keys():
		if not os.path.isfile(models_info[model_name]):
			print(f'[ERROR] {model_name} file does not exist')
			exit(1)

	#Init classes
	face_detec = FaceDetectionModel(models_info['face_detection_model'], args.device, args.cpu_extension, args.threshold)
	fac_land = FacialLandmarksDetectionModel(models_info['facial_landmarks_detection_model'], args.device, args.cpu_extension)
	head_pose = HeadPoseEstimationModel(models_info['head_pose_estimation_model'], args.device, args.cpu_extension)
	gaze_est = GazeEstimationModel(models_info['gaze_estimation_model'], args.device, args.cpu_extension)
    
	
	return face_detec, fac_land, head_pose, gaze_est



def main():
    
    # Grab command line args
    args = build_argparser().parse_args()

    #Init video feeder
    feeder = check_source(args.input)

    #Init mouse controller
    mouse_controller = MouseController('medium','fast')

    #Init models 
    face_detec, fac_land, head_pose, gaze_est = init_models(args)

    #Get data from source
    feeder.load_data()

    #Load models
    print('Loading models...')
    face_detec.load_model()
    fac_land.load_model()
    head_pose.load_model()
    gaze_est.load_model()
    print('Models Loaded')










if __name__ == '__main__':
    main()


