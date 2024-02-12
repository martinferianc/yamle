.. _extending_datamodule:

************************
Extending DataModule
************************

In this Tutorial we will demonstrate how to extend the :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>` class to create a custom DataModule.

We will be adding or looking at how to add the MNIST dataset to YAMLE through a custom DataModule. MNIST is a dataset of handwritten digits, which is a popular dataset for testing image classification models. The dataset is available through the `torchvision <https://pytorch.org/vision/stable/datasets.html#mnist>`_ package.

To start an implementation of any datamodule we recommend to look at the :py:mod:`BaseDataModule <yamle.data.datamodule.BaseDataModule>` class. It has many arguments which can be used to customize the datamodule.

.. literalinclude:: ../../yamle/data/datamodule.py
    :language: python
    :lines: 36-60

This class also does already cointain a lot of useful functionality e.g. to do automatic splitting of the dataset to training, validation and calibration portions e.g. through the :py:meth:`setup <yamle.data.datamodule.BaseDataModule.setup>` method.

.. literalinclude:: ../../yamle/data/datamodule.py
    :language: python
    :pyobject: BaseDataModule.setup

Note that the :py:meth:`setup <yamle.data.datamodule.BaseDataModule.setup>` method wraps the datasets into a `SurrogateDataset <yamle.data.dataset_wrappers.SurrogateDataset>` which is a wrapper around the `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_ class. This wrapper allows to manually control the data or the target transformations. 

The transformations are generally managed through a :py:meth:`get_transform <yamle.data.datamodule.BaseDataModule.get_transform>` method which is being called for each dataset split: training, validation, calibration and testing. 

Then there is the :py:meth:`prepare_data <yamle.data.datamodule.BaseDataModule.prepare_data>` method which is used to download the dataset. This method is only called once per machine and not per GPU. This is important to know if you want to download the dataset multiple times. The :py:meth:`prepare_data <yamle.data.datamodule.BaseDataModule.prepare_data>` method is called before the :py:meth:`setup <yamle.data.datamodule.BaseDataModule.setup>` method.

Now let's start with the implementation of the MNIST datamodule. In fact, many of the torchvision datasets can be processed in a similar way hence we will create two classes. One for general torchvision classification datasets and one concretely for MNIST.

The torchvision classification datamodule is implemented in :py:mod:`TorchvisionClassificationDataModule <yamle.data.classification.TorchvisionClassificationDataModule>`. 

.. literalinclude:: ../../yamle/data/classification.py
    :language: python
    :pyobject: TorchvisionClassificationDataModule

It inherits from a :py:mod:`VisionClassificationDataModule <yamle.data.datamodule.VisionClassificationDataModule>` which implements useful methods for debugging and plotting of the predictions or the applied augmentations.

Any datamodule also allows specification of custom arguments e.g. the :code:`datamodule_pad_to_32` argument through :py:meth:`add_specific_args <yamle.data.datamodule.BaseDataModule.add_specific_args>`. 

.. literalinclude:: ../../yamle/data/classification.py
    :language: python
    :pyobject: TorchvisionClassificationDataModule.add_specific_args

Note the :code:`datamodule_` prefix which is used to avoid name clashes with other arguments and separate the datamodule arguments from any other arguments.

The module can accept custom arguments such as :code:`pad_to_32` which can pad the image to a size of 32x32 pixels. This is useful if you want to use a model which requires a certain input size or to be used to apply out-ouf-distribution augmentations common in the field of out-of-distribution detection. Notice that, in practice the user only needs to fill in the :py:meth:`prepare_data <yamle.data.datamodule.BaseDataModule.prepare_data>` method which downloads the training or the test datasets and places them at the :py:attr:`_data_dir <yamle.data.datamodule.BaseDataModule._data_dir>` location. The :py:meth:`setup <yamle.data.datamodule.BaseDataModule.setup>` method is then used to wrap the datasets into a :py:class:`SurrogateDataset <yamle.data.dataset_wrappers.SurrogateDataset>` and to split the training dataset into training, validation and calibration portions. 

Finally we create a concrete MNIST datamodule :py:mod:`TorchvisionClassificationDataModuleMNIST <yamle.data.classification.TorchvisionClassificationDataModuleMNIST>` which inherits from the :py:mod:`TorchvisionClassificationDataModule <yamle.data.classification.TorchvisionClassificationDataModule>`

.. literalinclude:: ../../yamle/data/classification.py
    :language: python
    :pyobject: TorchvisionClassificationDataModuleMNIST

Note that each end datamodule which implements a concrete dataset needs to specify the :py:attr:`inputs_dim <yamle.data.datamodule.BaseDataModule.inputs_dim>`,  :py:attr:`outputs_dim <yamle.data.datamodule.BaseDataModule.outputs_dim>`, :py:attr:`targets_dim <yamle.data.datamodule.BaseDataModule.targets_dim>` and optionally :py:attr:`mean <yamle.data.datamodule.BaseDataModule.mean>` and :py:attr:`std <yamle.data.datamodule.BaseDataModule.std>` attributes. These attributes are used to normalize the data and to calculate the input and output dimensions of the model.

The last step is to register the new datamodule in the :py:mod:`__init__ <yamle.data.__init__>` module along all the other available datamodules.

.. literalinclude:: ../../yamle/data/__init__.py
    :language: python
