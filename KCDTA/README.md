# Resources:
## Source codes:
- **create_davis_kiba_pdb.py** is to convert Davis dataset, KIBA dataset and PDBBind Refined dataset into pytorch format.
- **create_metz.py** is to convert Metz dataset into pytorch format.
- **test.py** is used to test the accuracy of Davis, KIBA and PDBBind Refined datasets.
- **metz_test.py** is used to test the accuracy of Metz dataset.
- **training.py** is used to train the model of the dataset.
- **utils.py** is a tool class for metrics and transforming data formats.
- **models/cnn.py** is the neural network defined in this experiment.
- In the **data** folder, we store the .csv files of our processed Metz, PDBBind Refined datasets and the files of unprocessed Davis and KIBA datasets.
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
It is worth noting that the code we provide is the fifth fold in a five-fold crossover experiment to create the Davis and KIBA datasets.
## 2.Train a prediction model
You need to do the following
``` python
python training.py 0 0 0
```
The first argument has four values, *0/1/2/3/4/5/6*, representing the ***Davis,KIBA,PDBBind Refined,Metz1/Metz2/Metz3*** datasets.The second argument has only one value, 
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
## 3.Test the performance of the trained model on Davis,KIBA,PDBBind Refined datasets
You need to run the following 
``` python
python test.py 0
```
arguments *0/1/2* represent ***Davis, KIBA, PDBBind Refined*** datasets respectively.This operation will test the model generated during the training phase with the selected dataset to obtain the final result.

It is worth noting that if you want to reproduce our experimental results, regarding the Davis, KIBA data set, only the fifth of the five folds was created in the **create_davis_kiba_pdb.py** we provided.

If you want to reproduce the first fold data, please modify the 17th to 18th lines in **create_davis_kiba_pdb.py** as follows
``` python
train_fold = [ee for e in train_fold1[0:4] for ee in e]
test_fold = [ee for e in train_fold1[4:5] for ee in e]
```
If you want to reproduce the second fold data, please modify the 17th to 18th lines in **create_davis_kiba_pdb.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:1] for ee in e]
train_fold3 = [ee for e in train_fold1[2:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[1:2] for ee in e]
```

If you want to reproduce the third fold data, please modify the 17th to 18th lines in **create_davis_kiba_pdb.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:2] for ee in e]
train_fold3 = [ee for e in train_fold1[3:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[2:3] for ee in e]
```
If you want to reproduce the fourth fold data, please modify the 17th to 18th lines in **create_davis_kiba_pdb.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:3] for ee in e]
train_fold3 = [ee for e in train_fold1[4:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[3:4] for ee in e]
```
After modifying the code according to the above operations, execute **create_davis_kiba_pdb.py** to create a dataset (every time you create a dataset, you need to check whether there is a processd directory under the data folder, and delete it if it exists), and copy the model parameters from the cloud disk we provided to Under the KC-DTA directory, note that the names of the model parameters need to be changed to ***model_cnn_davis*** and ***model_cnn_kiba*** according to the Davis dataset and the KIBA dataset respectively(For example, modify the ***first fold*** in the davisvest directory to ***model_cnn_davis***).

For the results of reproducing the **PDBBind Refined** dataset, no code modification is required, just copy the model parameters from the network disk to the KCDTA directory after creating the dataset, and execute
``` python
python test.py 2
```
## 3.Test the performance of the trained model on Metz dataset
You need to run the following 
``` python
python metztest.py 0
```
arguments *0/1/2* represent ***metz1,metz2,metz3*** datasets respectively.This operation will test the model generated during the training phase with the selected dataset to obtain the final result.

If you want to reproduce our results, you need to copy the model parameters from the cloud disk to the directory of KCDTA , and execute the commands corresponding to the **Metz** dataset without modifying any code
