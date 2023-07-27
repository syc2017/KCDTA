# Resources:
## Source codes:
- **create_davis_kiba_pdb.py** is to convert Davis dataset, KIBA dataset and PDBBind Refined dataset into pytorch format.
- **create_metz.py** is to convert Metz dataset into pytorch format.
- **test.py** is used to test the accuracy of this experiment.
- **training.py** is used to train the model of the dataset.
- **utils.py** is a tool class for metrics and transforming data formats.
- **models/cnn.py** is the neural network defined in this experiment.
# Steps to execute the program
## 1. Convert data to pytorch format
If you want to convert the Metz dataset to pytorch format, you need to do the following 
``` python
python create_metz.py
```
If you want to convert the Davis, KIBA, PDBBind Refined dataset into pytorch format, you need to do the following
``` python
python create_davis_kiba_pdb.py
```
## 2.Train a prediction model
You need to do the following
``` python
python training.py 0 0 0
```
The first argument has four values, *0/1/2/3*, representing the ***Davis,KIBA,PDBBind Refined,Metz*** datasets.The second argument has only one value, 
*0*. It represents the use of the convolutional neural network provided in this experiment.The value of the third argument is the value of your cuda index, which has two values of *0/1*. The real CUDA name may vary from these, so please change the following code according to the actual situation.
``` python
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
```
The best model of MSE will be obtained through training, and two files that return model arguments and performance results. If you run 
``` python
python training.py 0 0 0
````
it will return ***model_cnn_davis.model*** and ***result_cnn_davis.csv***.
## 3.Test the performance of the trained model
You need to run the following 
``` python
python test.py 0
```
arguments *0/1/2/3* represent ***Davis, KIBA, PDBBind Renfined, Metz*** datasets respectively.This operation will test the model generated during the training phase with the selected dataset to obtain the final result.
