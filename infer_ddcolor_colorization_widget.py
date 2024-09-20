import torch
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_ddcolor_colorization.infer_ddcolor_colorization_process import InferDdcolorColorizationParam
from infer_ddcolor_colorization.ddcolor.infer import MODEL_NAMES

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDdcolorColorizationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDdcolorColorizationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # GPU computing
        self.check_cuda = pyqtutils.append_check(self.grid_layout,
                                                 "Cuda",
                                                 self.parameters.cuda and torch.cuda.is_available())

        # Model name
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")
        for model_name in MODEL_NAMES:
            self.combo_model_name.addItem(model_name)

        self.combo_model_name.setCurrentText(self.parameters.model_name)

        # Model input size
        self.spin_input_size = pyqtutils.append_spin(self.grid_layout,
                                                     "Input size",
                                                     self.parameters.input_size,
                                                     min=128, max=4096)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.input_size = self.spin_input_size.value()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDdcolorColorizationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_ddcolor_colorization"

    def create(self, param):
        # Create widget object
        return InferDdcolorColorizationWidget(param, None)
