'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import logging



class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,prob_threshold=0.6):
        #Initialize class
        self.model_name = model_name
        self.device = device
        self.extensions= extensions
        self.prob_threshold=prob_threshold
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
        #Make inference
        processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed})
        coords = self.preprocess_output(outputs)
        return self.crop_face(coords,image)

    
    def check_model(self): 
        #Check for unsupported layers 
        log = logging.getLogger()
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

    
    def crop_face(self,coords,image):
        
        if (len(coords)==0):
            return 0,0

        coords = coords[0] 

        h = image.shape[0]
        w = image.shape[1]
        
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)   
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face,coords


    def preprocess_output(self, outputs):
        coords =[]
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf>self.prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return coords
