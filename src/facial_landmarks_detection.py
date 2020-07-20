'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore

class FacialLandmarksDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        #Initialize class
        self.model_name = model_name
        self.device = device
        self.extensions= extensions
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        # Getting the reference of the model
        model_structure = self.model_name
        model_weights = self.model_name.split('.')[0]+'.bin'

        # Initialize the plugin for the device
        if not self.plugin:
            self.plugin = IECore()
        else:
            self.plugin = plugin

        #Add CPU extension if applicable
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions,self.device)

        # Read the IR
        #self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        self.network = IENetwork(model=model_structure, weights=model_weights)

        #Check for unsuppported layers
        self.check_model()

        #Get the exec network       
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        #Handling inputs and outputs 
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def predict(self, image):
        processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed})
        coords = self.preprocess_output(outputs)
        return self.preprocess_coords(coords,image)

    def preprocess_coords(self,coords,image):
        h,w = image.shape
        eye_area=10

        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        le_xmin=coords[0]-eye_area
        le_ymin=coords[1]-eye_area
        le_xmax=coords[0]+eye_area
        le_ymax=coords[1]+eye_area
        
        re_xmin=coords[2]-eye_area
        re_ymin=coords[3]-eye_area
        re_xmax=coords[2]+eye_area
        re_ymax=coords[3]+eye_area
        
        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        
        return left_eye, right_eye, eye_coords
    

    def check_model(self):
        #Check for unsupported layers 
        if self.device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)  
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("[ERROR] Unsupported layers found: {}".format(unsupported_layers))
                sys.exit(1)

    def preprocess_input(self, image):
        # Resize and change channels 
        processed = cv2.resize(image,(self.input_shape[3], self.input_shape[2]))
        processed = processed.transpose(2, 0, 1)
        processed = processed.reshape(1, *processed.shape)
        return processed

    def preprocess_output(self, outputs):
        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)
        
        