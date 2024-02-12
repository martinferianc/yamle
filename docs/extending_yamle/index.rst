.. _extending_yamle:

*******************
Extending YAMLE
*******************

This section covers the extension of YAMLE for :py:mod:`BaseModel <yamle.models.model.BaseModel>`, :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` and :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>`.

The :py:mod:`BaseModel <yamle.models.model.BaseModel>` is the base class for all models in YAMLE. It provides the architecture of the model and its forward pass. It defines several components, such as the input and output layers, which can be modified by the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>`. The goal is to write general and configurable implementations of a model that can be used across different datasets and tasks. For example, if defining a multi-layer perceptron, the :py:mod:`BaseModel <yamle.models.model.BaseModel>` should be configurable to different widths, depths, and activation functions.

Please see the :ref:`extending_model` section for more details on how to extend YAMLE for new models.

The :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>` is the base class for all data modules in YAMLE. It is responsible for downloading, loading, and preprocessing data. It defines the task, e.g., classification or regression, to be solved by the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` and :py:mod:`BaseModel <yamle.models.model.BaseModel>` and handles data splitting into training, validation, and test sets. It also defines the data input and output dimensions, which can be used to modify the :py:mod:`BaseModel <yamle.models.model.BaseModel>` by the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>`.

Please see the :ref:`extending_datamodule` section for more details on how to extend YAMLE for new data modules.

The :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` is the base class for all methods in YAMLE. It defines the interface that can optionally change the model and specifies the training, validation, and test steps by reusing PyTorch Lightning's functionality. For instance, it can be used to implement a new training algorithm by overloading the :py:meth:`_training_step <yamle.methods.method.BaseMethod._training_step>`, :py:meth:`_validation_step <yamle.methods.method.BaseMethod._validation_step>`, and :py:meth:`_test_step <yamle.methods.method.BaseMethod._test_step>` methods. Depending on the provided :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>` it decides automatically which are relevant algorithmic metrics to log and automatically logs them through the use of callbacks provided by PyTorch Lightning at the end of each epoch. The validation metrics are automatically passed to syne-tyne for hyperparameter optimization if it is desired. The :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` also considers the loss function, optimizer, and regularization during training; and can incorporate :py:mod:`BasePruner <yamle.methods.pruning.pruner.BasePruner>` and :py:mod:`BaseQuantizer <yamle.methods.quantization.quantizer.BaseQuantizer>` during evaluation. 

Please see the :ref:`extending_method` section for more details on how to extend YAMLE for new methods.

All the components—:py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>`, :py:mod:`BaseModel <yamle.models.model.BaseModel>`, and :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>`—enable customization through defining their own arguments that can be triggered via argparse. These components are orchestrated by the :py:mod:`Trainer/Tester <yamle.trainers.trainer.Trainer>`, responsible for querying the :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>`, :py:mod:`BaseModel <yamle.models.model.BaseModel>`, and :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>`, and executing training and evaluation loops through the step methods, as well as running on a specific device platform. These classes are connected and facilitate end-to-end experiments from data preprocessing to model training and evaluation. It only requires subclassing the appropriate classes, registering them in the framework for selection via argparse, and executing training or evaluation using the methods defined in :py:mod:`yamle.cli <yamle.cli>`.


.. toctree::
    :maxdepth: 2

    datamodule
    method
    model
