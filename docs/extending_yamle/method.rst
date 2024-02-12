.. _extending_method:

====================
Extending Method
====================

In this Tutorial we will demonstrate how to extend the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` class to create a new model.

More concretely, we will be implemeting the Monte Carlo Dropout method from the paper `Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning <https://arxiv.org/abs/1506.02142>`_.

The method is based on inserting a dropout layer before each learnable layer in the network. The dropout layer is then used to sample from the posterior distribution of the weights. The method is implemented in the :py:mod:`MCDropoutMethod <yamle.methods.mcdropout.MCDropoutMethod>` class.

To start an implementation from scratch, we first need to import the :py:mod:`MCSamplingMethod <yamle.methods.uncertain_method.MCSamplingMethod>` class. This class implements the basic functionality of the Monte Carlo sampling methods. It is a subclass of the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` class, which implements the basic functionality of all methods. The Monte Carlo sampling runs the network multiple times with different dropout masks. The output of the network is then averaged over all runs.

For a new method you would ideally create a new file in the :code:`yamle/methods` folder e.g. :code:`yamle/methods/mcdropout.py`. In this file you would then create a new class e.g. :py:mod:`MCDropoutMethod <yamle.methods.mcdropout.MCDropoutMethod>` that inherits from the :py:mod:`MCSamplingMethod <yamle.methods.uncertain_method.MCSamplingMethod>` class.

.. literalinclude:: ../../yamle/methods/mcdropout.py
   :language: python
   :pyobject: MCDropoutMethod

In the :py:meth:`__init__() <yamle.methods.mcdropout.MCDropoutMethod.__init__>` method we first call the :code:`super().__init__()` method to initialize the :py:mod:`MCSamplingMethod <yamle.methods.uncertain_method.MCSamplingMethod>` class with any parent arguments. These include for example the number of samples to take or the number of epochs to train for.

In the :code:`def __init__` we also define any arguments that are specific to Monte Carlo Dropout e.g. :code:`p` the dropout probability or :code:`mode` whether to use insert the dropout layer in the entire network or only before the last layer.

To make a model compatible with the Monte Carlo Dropout method, we need to insert dropout layers into the network and make sure that they are always active. We implement a custom :py:mod:`Dropout <yamle.models.specific.mcdropout>` layer that has the dropout turned always on. Do implement custom modules under the :code:`yamle/models/specific` folder for consistency.

.. literalinclude:: ../../yamle/models/specific/mcdropout.py
   :language: python
   :pyobject: Dropout1d

The `def __init__` also gives us the space to modify the :py:mod:`Model <yamle.models.model.BaseModel>` by inserting the dropout layers via the :py:meth:`replace_with_dropout <yamle.models.specific.mcdropout.replace_with_dropout>` method. Also for any support methods, we can add them to the :code:`yamle/models/specific/mcdropout.py` file for consistency.

.. literalinclude:: ../../yamle/models/specific/mcdropout.py
   :language: python
   :pyobject: replace_with_dropout

Now let's talk about how to customise the training, validation and test steps of a method. These are generally defined in the :py:meth:`_training_step <yamle.methods.method.BaseMethod._training_step>` method, the :py:meth:`_validation_step <yamle.methods.method.BaseMethod._validation_step>` method and the :py:meth:`_test_step <yamle.methods.method.BaseMethod._test_step>` method. In general, these functions call a default :py:meth:`_step <yamle.methods.method.BaseMethod._step>` method that is defined in the :py:mod:`BaseMethod <yamle.methods.method.BaseMethod>` class. 

.. literalinclude:: ../../yamle/methods/method.py
   :language: python
   :pyobject: BaseMethod._step

This method is responsible for running the network and calculating the loss. The training, validation or test steps can define custom behaviour by overriding them. In this case, it is not necessary to modify any of these methods, since :py:mod:`MCSamplingMethod <yamle.methods.uncertain_method.MCSamplingMethod>` already implements the correct behaviour through overriding the `def _predict <yamle.methods.uncertain_method.MCSamplingMethod._predict>` method.

.. literalinclude:: ../../yamle/methods/uncertain_method.py
   :language: python
   :pyobject: MCSamplingMethod._predict

Lastly, we need to be able to provide the arguments of the method to the :py:mod:`MCDropoutMethod <yamle.methods.MCDropoutMethod>` class. This is done by overriding the :py:meth:`add_specific_args <yamle.methods.mcdropout.MCDropoutMethod.add_specific_args>` method. This method is called by the :py:mod:`BaseMethod <yamle.methods.method>` class when the arguments are parsed.

.. literalinclude:: ../../yamle/methods/mcdropout.py
   :language: python
   :pyobject: MCDropoutMethod.add_specific_args

Notice the `--method_` prefix in the argument names. This is necessary to avoid conflicts with other arguments that can be parsed via the command line.

The last step would be to add the new method to the :py:mod:`__init__ <yamle.methods.__init__>` file in the :code:`yamle/methods` folder. This is necessary to make the method available via the command line.

.. literalinclude:: ../../yamle/methods/__init__.py
   :language: python

Afterwards you can run the method via the command line. For example:

.. code-block:: bash

   python3 yamle/cli/train.py --method mcdropout --trainer_devices "[0]" --datamodule mnist --datamodule_batch_size 256 --method_optimizer adam --method_learning_rate 3e-4 --regularizer l2 --method_regularizer_weight 1e-5 --loss crossentropy  --save_path ./experiments  --trainer_epochs 3 --model_hidden_dim 32 --model_depth 3 --datamodule_validation_portion 0.1 --save_path ./experiments --datamodule_pad_to_32 1 --method_p 0.3 --method_mode all --method_num_members 10

