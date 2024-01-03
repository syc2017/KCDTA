# Resources:
## Source codes:
- **create_davis_kiba.py** is to convert Davis dataset, KIBA dataset into pytorch format. Lines 121 to 154 involve processing protein sequences into three-dimensional matrices using the k-mers method and storing them in the **pro_dic** dictionary. Lines 177 to 202 involve processing protein sequences into two-dimensional matrices using the Cartesian product method and storing them in the **dpro_dic** dictionary. Small molecules are converted into graphs using the **smile_to_graph** function, and lines 170 to 173 store these graphs in the **smile_graph** dictionary.
- **test.py** is used to test the accuracy of Davis, KIBA datasets.
- **training.py** is used to train the model of the dataset.
- **utils.py** is a tool class for metrics and transforming data formats.
- **models/cnn.py** is the neural network defined in this experiment.
- In the **data** folder, we store the .csv files of our processed Metz and the files of unprocessed Davis and KIBA datasets.
# Steps to execute the program
## 1. Convert data to pytorch format
If you want to convert the Davis, KIBA into pytorch format, you need to do the following
``` python
python create_davis_kiba.py
```
It is worth noting that the code we provide is the fifth fold in a five-fold crossover experiment to create the Davis and KIBA datasets.
## 2.Train a prediction model
You need to do the following
``` python
python training.py 0 0 0
```
The first argument has five values, *0/1*, representing the ***Davis,KIBA*** datasets.The second argument has only one value, 
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
## 3.Test the performance of the trained model on Davis,KIBA datasets
You need to run the following 
``` python
python test.py 0
```
arguments *0/1* represent ***Davis, KIBA*** datasets respectively.This operation will test the model generated during the training phase with the selected dataset to obtain the final result.

It is worth noting that if you want to reproduce our experimental results, regarding the **Davis, KIBA** dataset, only the fifth of the five folds was created in the **create_davis_kiba.py** we provided.

If you want to reproduce the first fold data, please modify the 18th to 19th lines in **create_davis_kiba.py** as follows
``` python
train_fold = [ee for e in train_fold1[0:4] for ee in e]
test_fold = [ee for e in train_fold1[4:5] for ee in e]
```
If you want to reproduce the second fold data, please modify the 18th to 19th lines in **create_davis_kiba.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:1] for ee in e]
train_fold3 = [ee for e in train_fold1[2:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[1:2] for ee in e]
```

If you want to reproduce the third fold data, please modify the 18th to 19th lines in **create_davis_kiba.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:2] for ee in e]
train_fold3 = [ee for e in train_fold1[3:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[2:3] for ee in e]
```
If you want to reproduce the fourth fold data, please modify the 18th to 19th lines in **create_davis_kiba.py** as follows
``` python
train_fold2 = [ee for e in train_fold1[0:3] for ee in e]
train_fold3 = [ee for e in train_fold1[4:5] for ee in e]
train_fold =train_fold2+train_fold3
test_fold = [ee for e in train_fold1[3:4] for ee in e]
```
After modifying the code according to the above operations, execute **create_davis_kiba.py** to create a dataset (every time you create a dataset, you need to check whether there is a processd directory under the data folder, and delete it if it exists), and copy the model parameters from the cloud disk(https://drive.google.com/file/d/1nsL6QR0xlY_JRVyUuLI47dRAFjRvvOyF/view?usp=drive_link) we provided to Under the KC-DTA directory, note that the names of the model parameters need to be changed to ***model_cnn_davis*** and ***model_cnn_kiba*** according to the **Davis** dataset and the **KIBA** dataset respectively(For example, modify the ***first fold*** in the **davisbest** directory to ***model_cnn_davis***).
