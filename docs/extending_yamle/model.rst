.. _extending_model:

**********************
Extending Model
**********************

In this Tutorial we will demonstrate how to extend the :py:mod: `BaseModel <yamle.models.model.BaseModel>` class to create a new model.

.. literalinclude:: ../../yamle/models/model.py
   :language: python
   :pyobject: BaseModel

Each model which is added to YAMLE needs to inherit from the :py:mod: `BaseModel <yamle.models.model.BaseModel>` class. The :py:mod:`BaseModel <yamle.models.model.BaseModel>` class provides a number of methods which are used to cross-interact the model with a method `BaseMethod <yamle.methods.method.BaseMethod>` and a datamodule `BaseDataModule <yamle.data.datamodule.BaseDataModule>`.

Note that each model needs to be able to accept the :code:`inputs_dim`, :code:`outputs_dim` and :code:`task` which automatically decides the number of inputs and outputs for the model. The :code:`task` is a string which is used to determine the type of task the model is being used for. The task usually determines the output activation, for example softmax for classification and exponential applied to one of the outputs in regression to model the variance. 

It is expected that the very first learnable layer will be in the :py:attr:`_input <yamle.models.model.BaseModel._input>` attribute and the very last learnable layer will be in the :py:attr:`_output <yamle.models.model.BaseModel._output>` attribute. The output activation is expected to be in the :py:attr:`_output_activation <yamle.models.model.BaseModel._output_activation>` attribute. This is such that it is possible to easily extract the model's input and output layers and the output activation if needed by the underlying `BaseMethod <yamle.methods.method.BaseMethod>`.

There are also other functions which can be used to define the exact behaviour when quantising the model, reset the model each training epoch or to add method-specific layer to the model while keeping the backbone of the model the same. These are all optional and can be overridden if needed.

The most important methods are the :py:meth:`forward <yamle.models.model.forward>` or :py:meth:`final_layer <yamle.models.model.final_layer>` which specify the forward pass of the model or the processing of the last hidden features with respect to the output layer and the output activation.

A concrete example is a fully connected network with multiple hidden layers :py:mod:`FC <yamle.models.fc_model.FCModel>`.

.. literalinclude:: ../../yamle/models/fc.py
   :language: python
   :pyobject: FCModel

Notice the implementation of the :code:`_input` and :code:`_output` layers which automatically take into the account the input and output dimensions passed down by the datamodule that has been chosen to run the experiment with. The :code:`_output_activation` is also automatically chosen based on the task.

Notice that the :py:meth:`forward <yamle.models.model.forward>` method takes in extra keyword arguments e.g. to output the hidden representation by each hidden layer, this is used by certain specific methods along with the function to add extra layers for some specfic methods. 

To specify the arguments of the model there is the fucntion :py:meth:`add_specific_args <yamle.models.model.BaseModel.add_specific_args>` which is used to add the arguments of the model to the :py:mod: `ArgumentParser <argparse.ArgumentParser>` of the experiment in the command line. This is used to specify the number of hidden layers, the activation or the width of the network.

The model also uses some general layers such as :py:mod:`LinearNormActivation <yamle.models.operations.LinearNormActivation>` which is a linear layer followed by a normalisation layer and an activation layer. This class is used also in other models since it is quite general. If you feel that you will be using/implementing a general layer, place it in the :py:mod:`operations <yamle.models.operations>` module. For a method-specific layer, place it in the :py:mod:`specific <yamle.models.specific>` folder.

The last step is to register the new model in the :py:mod:`__init__ <yamle.models.__init__>` file of the :py:mod:`models <yamle.models>` module. This is done by adding the model to the following list:

.. literalinclude:: ../../yamle/models/__init__.py
   :language: python
