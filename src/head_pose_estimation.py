'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import logging

class HeadPoseEstimationModel:
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
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, image):
        processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed})
        return self.preprocess_output(outputs)

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
        #Resize and change channels
        processed = cv2.resize(image,(self.input_shape[3], self.input_shape[2]))
        processed = processed.transpose(2, 0, 1)
        processed = processed.reshape(1, *processed.shape)
        return processed

    def preprocess_output(self, outputs):
        out = []
        out.append(outputs['angle_y_fc'].tolist()[0][0])
        out.append(outputs['angle_p_fc'].tolist()[0][0])
        out.append(outputs['angle_r_fc'].tolist()[0][0])
        return out
