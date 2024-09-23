<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_ddcolor_colorization</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_ddcolor_colorization">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_ddcolor_colorization">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_ddcolor_colorization/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_ddcolor_colorization.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

![computed](https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/result.jpg)

![original](https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/original.jpg)
Original picture made by <a href="https://unsplash.com/fr/@adamlittmandavis?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Adam Littman Davis</a> on <a href="https://unsplash.com/fr/photos/photo-en-niveaux-de-gris-de-montagnes-et-darbres-CIian0EjHAU?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  
## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow("Colorization workflow")

# Add algorithm
algo = wf.add_task(name="infer_ddcolor_colorization", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/original.jpg")

display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- cuda (bool): enable/disable cuda acceleration (if available)
- model_name (str): ddcolor models
  - ddcolor_paper: original model from scientific paper.
  - ddcolor_paper_tiny: lightweight version of ddcolor model, using the same training scheme as ddcolor_paper.
  - ddcolor_modelscope: model trained using the same data cleaning scheme as BigColor, it can get the best qualitative results with little degrading FID performance.
  - ddcolor_artistic: model trained with an extended dataset containing many high-quality artistic images. Also, colorfulness loss is not used during training, so there may be fewer unreasonable color artifacts.
- input_size (int): image input resolution in pixels


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow("Colorization workflow")

# Add algorithm
algo = wf.add_task(name="infer_ddcolor_colorization", auto_connect=True)
algo.set_parameters({
    "cuda": "True",
    "model_name": "ddcolor_paper_tiny",
    "input_size": "1024"
})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/original.jpg")

display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_ddcolor_colorization", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_ddcolor_colorization/main/images/original.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```