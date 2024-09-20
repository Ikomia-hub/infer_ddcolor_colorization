import copy
import os
import torch
from ikomia import utils, core, dataprocess

from infer_ddcolor_colorization.ddcolor.infer import MODEL_NAMES
from infer_ddcolor_colorization.ddcolor.infer import InferDDColor


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDdcolorColorizationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = MODEL_NAMES[0]
        self.input_size = 512
        self.cuda = torch.cuda.is_available()

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = params["model_name"]
        self.input_size = int(params["input_size"])
        self.cuda = utils.strtobool(params["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {
            "model_name": self.model_name,
            "input_size": str(self.input_size),
            "cuda": str(self.cuda),
        }
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDdcolorColorization(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here

        self.ddcolor = InferDDColor(os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights"))

        # Create parameters object
        if param is None:
            self.set_param_object(InferDdcolorColorizationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        param = self.get_param_object()
        self.ddcolor.set_parameters(param.model_name, param.input_size, param.cuda)

        # Get input
        img_input = self.get_input(0)

        # Run ddcolor inference
        result = self.ddcolor.run(img_input.get_image())

        # Set output
        img_output = self.get_output(0)
        img_output.set_image(result)

        # Step progress bar (Ikomia Studio)
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDdcolorColorizationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_ddcolor_colorization"
        self.info.short_description = "Algorithm to colorize grayscale image"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Colorization"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong"
        self.info.article = "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders"
        self.info.journal = "ICCV"
        self.info.year = 2023
        self.info.license = "Apache-2.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_ddcolor_colorization"
        self.info.original_repository = "https://github.com/piddnad/DDColor/tree/master"
        # Keywords used for search
        self.info.keywords = "color, restoration, colorisation"
        # General type
        self.info.algo_type = core.AlgoType.INFER
        # Algorithms tasks
        self.info.algo_tasks = "COLORIZATION"

    def create(self, param=None):
        # Create algorithm object
        return InferDdcolorColorization(self.info.name, param)
