# CIFAR10 Claffier with Two Concatanated Images
In this implementation the a CNN is trained with two concatanated images from CIFAR10 dataset. The images are constructed by randomly shuffling the input batch and the corresponding labels and then the original input batch and the shuffled batched are concanated on the dimension 3. Therefore, the input_wide becomes `Bx3x32x64`. The final labels are also calculated by adding the original labels with the shuffled labels with weights of 0.5. Therefore, the output label has 0.5 for each class and it has 1.0 if both images are from the same class. 
# Dependencies
The dependencies of the code are 
Numpy
Pytorch 1.1.0
Matplotlib
# Running the Code
The traning and evaluation is performed inside the `training_evalution.py` file. It can be run on the command line by just typing
`python training_evalution.py [--epochs N] [-b N] [--lr LR] [--momentum M] [-j NW]`
Where N is the number of traning epochs, N is the batch size, LR is the learning rate, M is the momentum of the stochastic gradient descent and NW is the number of workers used for loading the training and testing datasets. 
# Evaluation
The accuracy calculation using the test data is performed inside the `training_evalution.py` and the calculated accuracy is printed with message like
`Accuracy of the network on the 10000 test images: X %` 
# Jupyter Notebook
Moreover, it is also possible to perform all these operations using  the jupyter notebook `cifar10_notebook.ipynb`. It is also possible to visualize the dataset and see the corresponding labels inside the dataset. 
![A sample of the visualization](./assets/dataset_visualize.png?raw=true "Title")
