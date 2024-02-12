YAMLE (Yet Another Machine Learning Environment)
================================================

.. image:: https://img.shields.io/badge/license-GPLv3-blue
  :target: https://opensource.org/licenses/GPL-3.0
  :alt: License: GPLv3
.. image:: https://img.shields.io/badge/Python-3.9-blue.svg
  :target: https://www.python.org/downloads/release/python-390/
  :alt: Python: 3.9
.. image:: https://img.shields.io/badge/PyTorch-2.0.0-blue.svg
  :target: https://pytorch.org/
  :alt: PyTorch: 2.0.0
.. image:: https://img.shields.io/badge/PyTorch%20Lightning-2.0.8-blue.svg
  :target: https://www.pytorchlightning.ai/
  :alt: PyTorch Lightning: 2.0.8

Overview
--------

YAMLE: Yet Another Machine Learning Environment is an open-source framework that facilitates rapid prototyping and experimentation with machine learning models and methods. The key motivation is to reduce repetitive work when implementing new approaches and improve reproducibility in ML research. YAMLE includes a command-line interface and integrations with popular and well-maintained PyTorch-based libraries to streamline training, hyperparameter optimisation, and logging. The ambition for YAMLE is to grow into a shared ecosystem where researchers and practitioners can quickly build on and compare existing implementations.

You can find the YAMLE repository on GitHub: `YAMLE Repository <https://github.com/martinferianc/yamle>`_

You can fint the short paper describing YAMLE: `arXiv paper <https://arxiv.org/abs/2402.06268>`_

You can find the documentation at: `yamle.readthedocs.io <https://yamle.readthedocs.io/en/latest/>`_

Table of Contents
-----------------

- `Introduction`_
- `Core Components and Modules`_
- `Use Cases and Applications`_
- `Future Development Roadmap`_
- `Citation`_

Introduction
------------

This repository introduces -- an open-source generalist customisable experiment environment with boilerplate code already implemented for rapid prototyping with ML models and methods. The main features of the environment are summarised as follows:

- **Modular Design**: The environment is divided into three main components - data, models, and methods - which are infrastructurally connected but can be independently modified and extended. The goal is to write a method or a model and then seamlessly use it across different models or methods across different datasets and tasks.

- **Command-line Interface**: The environment includes a command-line interface for easy configuration of all hyperparameters and training of models.

- **Hyperparameter Optimisation**: YAMLE is integrated with syne-tune for hyperparameter optimisation.

- **Logging**: YAMLE is integrated with TensorBoard for logging and visualization of training, validation, and test metrics.

- **End-to-End Experiments**: YAMLE enables end-to-end experiments, covering data preprocessing, model training, and evaluation. All settings are recorded for reproducibility.

Core Components and Modules
---------------------------

YAMLE is built on PyTorch and PyTorch Lightning and relies on torchmetrics for evaluation metrics and syne-tune for hyperparameter optimisation. The framework is designed to provide an ecosystem for rapid prototyping and experimentation. The core components and modules of YAMLE include:

- `BaseDataModule <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.data.datamodule.html>`_: Responsible for downloading, loading, and preprocessing data. It defines the task, data splitting, input/output dimensions, and more.

- `BaseModel <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.models.model.html>`_: Defines the architecture of the model and its forward pass. It can be configured for different widths, depths, and activation functions.

- `BaseMethod <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.methods.method.html>`_: Defines the training, validation, and test steps, as well as the loss function, optimiser, and regularization. It can also incorporate pruning and quantisation during evaluation.

These components are orchestrated by the `BaseTrainer <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.trainers.trainer.html>`_ class, which is responsible for executing training and evaluation loops and running on a specific device platform. YAMLE facilitates end-to-end experiments, from data preprocessing to model training and evaluation, by allowing users to customise these components through command-line arguments.

Use Cases and Applications
---------------------------

YAMLE is designed to serve as the template for the main project itself, allowing researchers and practitioners to conduct experiments, compare their models and methods, and easily extend the framework. The typical workflow for using YAMLE includes:

1. Clone the YAMLE repository and install dependencies.
2. Experiment with new methods or models by subclassing the `BaseModel <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.models.model.html>`_ or `BaseMethod <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.methods.method.html>`_ on the chosen `BaseDataModule <https://yamle.readthedocs.io/en/latest/_apidoc/yamle.data.datamodule.html>`_ or any other customisable component.
3. When satisfied with your additions, contribute them to the repository via a pull request.
4. New additions will be reviewed and categorised as staple or experimental features, and YAMLE will be updated accordingly.

YAMLE currently supports three primary use cases:

- **Training**: Initiate model training using the command-line interface, specifying hyperparameters, datasets, and other settings.

e.g. ``python3 yamle/cli/train.py --method base --trainer_devices "[0]" --datamodule mnist --datamodule_batch_size 256 --method_optimizer adam --method_learning_rate 3e-4 --regularizer l2 --method_regularizer_weight 1e-5 --loss crossentropy  --save_path ./experiments  --trainer_epochs 3 --model_hidden_dim 32 --model_depth 3 --datamodule_validation_portion 0.1 --save_path ./experiments --datamodule_pad_to_32 1``

- **Testing**: Conduct testing to evaluate the performance of your models or methods.

e.g. ``python3 yamle/cli/test.py --method base --trainer_devices "[0]" --datamodule mnist --datamodule_batch_size 256 --loss crossentropy  --save_path ./experiments --model_hidden_dim 32 --model_depth 3 --datamodule_validation_portion 0.1 --save_path ./experiments --datamodule_pad_to_32 1 --load_path ./experiments/<FOLDER>``

- **Hyperparameter Optimisation**: Optimise hyperparameters using syne-tune, a framework integrated into YAMLE for this purpose.

YAMLE allows users to quickly set up experiments, perform training, testing, and hyperparameter optimisation, covering the entire machine learning pipeline from data preprocessing to model evaluation.

e.g. ``python3 yamle/cli/tune.py --config_file <FILE_NAME> --optimiser "Grid Search" --save_path ./experiments/hpo/ --max_wallclock_time 420 --optimisation_metric "validation_nll"``


Future Development Roadmap
---------------------------

YAMLE is an evolving project, and there are several areas for future development and improvement:

- **Documentation**: Prioritising the creation of comprehensive documentation to make YAMLE more accessible to users.

- **Additional Tasks**: Expanding the range of problems supported by YAMLE, including unsupervised, self-supervised learning, and reinforcement learning tasks.

- **Expanding the Model Zoo**: Increasing the collection of models and methods for easy comparison with existing implementations.

- **Testing**: Implementing unit tests to ensure the reliability of the framework.

- **Multi-device Runs**: Extending support for multi-device training and testing.

- **Other Hyperparameter Optimisation Methods**: Including support for additional hyperparameter optimisation methods like Optuna and Ray Tune.

These improvements and extensions will enhance YAMLE's capabilities and make it an even more valuable tool for machine learning researchers and practitioners.

Citation
--------

If you use YAMLE in your research, please cite the following paper:

.. code-block:: bibtex

    @article{ferianc2024yamle,
      title={YAMLE: Yet Another Machine Learning Environment},
      author={Ferianc, Martin and Rodrigues, Miguel},
      journal={arXiv preprint arXiv:2402.06268},
      year={2024}
    }