***************
Getting Started
***************

This section covers the usage of YAMLE from the installation to the training of a model. 

YAMLE is not available as a package on PyPI yet. However, it is possible to install it directly from the git repository.
It is recommended to install YAMLE in a virtual environment.
For example, follow
`these instructions <https://docs.python.org/3/library/venv.html>`_ to use :code:`venv`.

.. code-block:: bash

    git clone https://github.com/martinferianc/yamle.git
    cd yamle
    pip install -e .
    # Subsequently also install syne-tune - unfortunatelly installation in one go breaks for package tqdm
    pip install 'syne-tune[extra]==0.10.0'
    

Afterwards you can try to run the example script:

.. code-block:: bash

    python yamle/cli/train.py --method base --trainer_devices "[0]" --datamodule mnist --datamodule_batch_size 256 --method_optimizer adam --method_learning_rate 3e-4 --regularizer l2 --method_regularizer_weight 1e-5 --loss crossentropy  --save_path ./experiments --trainer_epochs 3 --model_hidden_dim 32 --model_depth 3 --datamodule_validation_portion 0.1 --save_path ./experiments --model fc --datamodule_pad_to_32 1

This script trains a simple fully connected network :py:mod:`FC <yamle.models.fc.FC>` on the :py:mod:`MNIST <yamle.data.classification.TorchvisionClassificationDataModuleMNIST>` dataset. It uses L2 regularization defined by :py:mod:`L2 <yamle.regularizers.weight.L2>` and cross-entropy loss defined by :py:mod:`CrossEntropyLoss <yamle.losses.classification.CrossEntropyLoss>`. The model is trained for 3 epochs and the validation set is 10% of the training set. The model is saved to the :code:`./experiments` directory. All of this is grouped together through a trainer class :py:mod:`Trainer <yamle.trainers.trainer.Trainer>` which executes the training, validation or testing loops. The metrics are logged automatically and the base algorithmic metrics are supplied by the function: :py:meth:`metrics_factory <yamle.evaluation.metrics.algorithmic.__init__.metrics_factory>`. The logging is done through a PyTorch Lightning callback in the :py:mod:`LoggingCallback <yamle.utils.trainer_utils.LoggingCallback>`.

In general, YAMLE operates through the CLI where the user specifies the configuration of the experiment. The configuration is then parsed and the experiment is run. The configuration is specified through the command line arguments. The arguments are grouped into several categories. The most important ones are:

* :code:`--method` which specified the method and its parameters
* :code:`--model` which specifies the model and its parameters
* :code:`--loss` which specifies the loss and its parameters
* :code:`--regularizer` which specifies the regularizer and its parameters
* :code:`--datamodule` which specifies the datamodule and its parameters
* :code:`--trainer` which specifies the trainer and its parameters

When adding a new method, datamodule, model, regularizer etc. you will be able to define your own arguments. 

When a model was trained we can evaluate it using:

.. code-block:: bash

    python yamle/cli/evaluate.py --method base --trainer_devices "[0]" --datamodule mnist --datamodule_batch_size 256 --loss crossentropy  --save_path ./experiments --model_hidden_dim 32 --model_depth 3 --datamodule_validation_portion 0.1 --save_path ./experiments --model fc --datamodule_pad_to_32 1 --load_path ./experiments/2023-10-23-13-11-33-546652-train-fc-mnist-base

This script evaluates the model trained in the previous step. The evaluation is done on any data split specified by the datamodule. 

The last main feature of YAMLE is hyperparameter-optimisation. It is done through the syne-tune library. The hyperparameters and their range are specified in a config file e.g.:

.. code-block:: python 
    
    # config.py
    from syne_tune.config_space import randint, rand

    def config_space() -> Dict[str, Any]:
        return {
            "model_hidden_dim": randint(16, 128),
            "model_depth": randint(1, 5),
            "method": "base",
            "method_learning_rate": 3e-4,
            "method_optimizer": "adam",
            "method_regularizer_weight": 1e-5,
            "regularizer": "l2",
            "loss": "crossentropy",
            "datamodule": "mnist",
            "datamodule_batch_size": 256,
            "datamodule_validation_portion": 0.1,
            "datamodule_pad_to_32": 1,
            "trainer_epochs": 3,
            "save_path": "./experiments",
        }

The config file is then passed to the hyperparameter optimisation script:

.. code-block:: bash

    python yamle/cli/tune.py --config_file config.py --optimizer "Grid Search" --save_path ./experiments --max_wallclock_time 420 --optimization_metric "validation_accuracy"

The script will run the hyperparameter optimisation and save the best model to the :code:`./experiments` directory. We encourage you to look into the tune script to see how the hyperparameter optimisation is done.

In order to generate documentation from the docstrings, run:

.. code-block:: bash

    cd docs
    make html

The documentation will be generated in the :code:`docs/build/html` directory.

