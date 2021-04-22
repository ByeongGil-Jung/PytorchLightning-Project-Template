<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Pytorch project template for experiments and application (Description ...)    

## How to run   
First, install dependencies   
```bash
# Clone project   
git clone https://github.com/ByeongGil-Jung/PytorchLightning-Project-Template.git

# Install project   
cd PytorchLightning-Project-Template
pip install -e .   
pip install -r requirements.txt
 ```    
If you want to modify configurations, move below directory and modify it. 
 ```bash
# Move configuration directory        
cd config/hyperparameters
```    
Next, navigate to any file and run it.   
 ```bash
# Run module (example: mnist as your main contribution)   
python main.py --model "fc" --data "mnist" --stage "fit" --tqdm_env "script"    
```

## Example code
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from config.factory import HyperparameterFactory
from dataset.factory import DataModuleFactory
from domain.metadata import ModelMetadata
from model.factory import ModelFactory
from trainer.base import TrainerBase


# Arguments
model_name = "fc"
data_name = "mnist"
stage = "fit"
tqdm_env = "script"

model_metadata = ModelMetadata(model_name=model_name, information=None)

# Arguments controller
hyperparameter_factory = HyperparameterFactory.create(data_name=data_name, model_name=model_name)
datamodule_params = hyperparameter_factory.datamodule_params.to_dict()
trainer_params = hyperparameter_factory.trainer_params.to_dict()
model_params = hyperparameter_factory.model_params.to_dict()

# DataModule controller
datamodule = DataModuleFactory.create(data_name=data_name)
datamodule = datamodule(**datamodule_params)

datamodule.prepare_data()
datamodule.setup(stage=stage)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# Trainer controller
trainer = TrainerBase(model_metadata=model_metadata, **trainer_params)

# Model controller
model = ModelFactory.create(model_name=model_name, model_params=model_params)

# Find the optimal learning rate
if trainer.is_auto_lr_find:
    trainer.lr_find(model=model, train_loader=train_loader, val_loader=val_loader)

# Training & Validation
if stage == "fit" or stage == "whole":
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

# Testing
if stage == "test" or stage == "whole":
    trainer.test(model=model, test_dataloaders=test_loader)

# Save figures
if trainer.is_auto_lr_find:
    fig = trainer.lr_finder.plot(suggest=True)
    fig.savefig(fname=model_metadata.model_file_metadata.optimal_lr_plot_path)    
```

### Citation   
```
@article{Byeonggil Jung,
  title={Title},
  author={Team},
  journal={Location},
  year={Year}
}
```   
