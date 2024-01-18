import copy
from ikomia import core, dataprocess
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDdcolorColorizationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.window_size = 25
        self.update = True


    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.window_size = int(params["window_size"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        # Example : paramMap["window_size"] = str(self.window_size)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDdcolorColorization(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())
        self.pipeline = None

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

        # Examples :
        # Get input :
        img_input = self.get_input(0)

        # Get output :
        img_output = self.get_output(0)

        param = self.get_param_object()

        if param.update or self.pipeline is None:
            plugin_folder = os.path.dirname(os.path.abspath(__file__))
            os.environ['MODELSCOPE_CACHE'] = os.path.join(plugin_folder, "cached_models")
            self.pipeline = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
            param.update = False

        result = self.pipeline(img_input.get_image())

        img_output.set_image(result['output_img'][:, :, ::-1])

        # Step progress bar (Ikomia Studio):
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
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong"
        self.info.article = "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders"
        self.info.journal = "Proceedings of the IEEE/CVF International Conference on Computer Vision"
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
