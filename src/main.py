import cv2
import os
import numpy as np
import logging
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import time



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
 
    parser.add_argument("-fl", "--flags", required=False, type=str,nargs='+', default=[],
                        help="See outputs of selected models. "
                        "Valid inputs: 'FLM', 'HPEM' and 'GEM' ")
    
    parser.add_argument("-d", "--device", required=False,type=str, default="CPU",
                        help="Target device to infer on: "
                             "Valid inputs: CPU, GPU, FPGA or MYRIAD.")

    parser.add_argument("-ce", "--cpu_extension", required=False,type=str, default=None,
                        help="Path to the CPU extension")

    parser.add_argument("-op", "--output_path", required=False,type=str, default=None,
                        help="Output path of the benchmark file")

    parser.add_argument("-b", "--benchmarking", required=False,type=str2bool, nargs='?',const=True,default=False,
                        help="Activates benchmarking mode and saves stats")
    return parser


def str2bool(v):
	#convert arg string into boolean
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_source(filepath,logger):
	
	feeder = None

	if filepath.lower()=='cam':
		feeder = InputFeeder('cam')
	else:
		if not os.path.isfile(filepath):
			logger.error("Unable to find specified video file")
			exit(1)

		feeder = InputFeeder('video',filepath)

	return feeder 


def init_models(args, logger):

	face_detec=None
	fac_land=None
	head_pose=None
	gaze_est=None
	
	models_info = {
	'face_detection_model': args.face_detection_model, 
	'facial_landmarks_detection_model': args.facial_landmarks_detection_model, 
    'head_pose_estimation_model': args.head_pose_estimation_model, 
    'gaze_estimation_model': args.gaze_estimation_model
    }

	for model_name in models_info.keys():
		if not os.path.isfile(models_info[model_name]):          
			logger.error("Unable to find the model file of " + str(model_name))
			exit(1)

	#Init classes
	face_detec = FaceDetectionModel(models_info['face_detection_model'], args.device, args.cpu_extension, args.threshold)
	fac_land = FacialLandmarksDetectionModel(models_info['facial_landmarks_detection_model'], args.device, args.cpu_extension)
	head_pose = HeadPoseEstimationModel(models_info['head_pose_estimation_model'], args.device, args.cpu_extension)
	gaze_est = GazeEstimationModel(models_info['gaze_estimation_model'], args.device, args.cpu_extension)
    
	
	return face_detec, fac_land, head_pose, gaze_est


def show_flm_outputs(face,eyes_coords):
	
	le_coords = eyes_coords[0]
	re_coords = eyes_coords[1]
	eye_area=10

	cv2.rectangle(face, (le_coords[0]- eye_area, le_coords[1]- eye_area), (le_coords[2]+ eye_area, le_coords[3]+eye_area), (0,0,255), 3)
	cv2.rectangle(face, (re_coords[0]- eye_area, re_coords[1]-eye_area), (re_coords[2] + eye_area, re_coords[3]+eye_area), (0,0,255), 3)
	      

def show_gem_outputs(gaze,left_eye,right_eye,face,eyes_coords):

	x, y, w = int(gaze[0]*12), int(gaze[1]*12), 160

	le_coords = eyes_coords[0]
	re_coords = eyes_coords[1]
    
	le_line =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
	cv2.line(le_line, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
    
	re_line = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
	cv2.line(re_line, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
    
	face[le_coords[1]:le_coords[3],le_coords[0]:le_coords[2]] = le_line
	face[re_coords[1]:re_coords[3],re_coords[0]:re_coords[2]] = re_line


def main():
    

    # Grab command line args
    args = build_argparser().parse_args()

    #Create logger obj
    logger = logging.getLogger('main')

    # Flags to show outputs of the model
    show_outputs = args.flags

    #Check if we are in benchmark mode
    benchmarking_mode= args.benchmarking


    output_path = args.output_path

    #Init video feeder
    feeder = check_source(args.input, logger)

    #Init mouse controller
    mouse_controller = MouseController('medium','fast')

    #Init models 
    face_detec, fac_land, head_pose, gaze_est = init_models(args,logger)

    #Get data from source
    feeder.load_data()

	#Load models
    logger.info("Loading Models")
    start_model_load_time = time.time()
    face_detec.load_model()
    fac_land.load_model()
    head_pose.load_model()
    gaze_est.load_model()
    model_load_time = time.time() - start_model_load_time
    logger.info('Models Loaded')

    #Init frame counter
    count = 0

    start_inference_time = time.time()

    for r, frame in feeder.next_batch():
       
        if not r:
        	break

        count+=1

        #To do: delete this
        if count%5==0 and len(show_outputs)==0 and (not benchmarking_mode):
           cv2.imshow('Computer Pointer Controller',cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        
        face, face_coords = face_detec.predict(frame.copy())

        if type(face) == int:
        	logger.error("Can't detect face")
        	if key==27:
        		break
        	continue

        hp_pred = head_pose.predict(face.copy())

        left_eye, right_eye, eyes_coords = fac_land.predict(face.copy())
        
        mouse_coords, gaze = gaze_est.predict(left_eye, right_eye, hp_pred)

        #Flags
        if ((len(show_outputs)>0) and (not benchmarking_mode)):

	        new_frame = face.copy()
	        
	        if 'FLM' in show_outputs:
	        	show_flm_outputs(new_frame,eyes_coords)

	        if 'HPEM' in show_outputs:
	            cv2.putText(new_frame, "Head > y:{:.2f} | p:{:.2f} | r:{:.2f}".format(hp_pred[0],hp_pred[1],hp_pred[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 255), 1)
	        
	        if 'GEM' in show_outputs:
	        	show_gem_outputs(gaze,left_eye,right_eye,new_frame,eyes_coords)

	        cv2.imshow('Outputs',cv2.resize(new_frame,(500,500)))


        if (count%5==0) and not benchmarking_mode:
        	mouse_controller.move(mouse_coords[0],mouse_coords[1])

        if key==27:
                break  

    inference_time = time.time() - start_inference_time
	#Stats for benchmark
    if(benchmarking_mode):
        with open(output_path+'benchmark.txt', 'w') as f:
            f.write(str(inference_time) + '\n')
            f.write(str(model_load_time) + '\n')

        logger.info('Model load time: ' + str(model_load_time))
        logger.info('Inference time: ' + str(inference_time))

    logger.info('Video stream ended')

    cv2.destroyAllWindows()
    feeder.close()


if __name__ == '__main__':
    main()


