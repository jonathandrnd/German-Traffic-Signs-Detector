# German-Traffic-Signs-Detector
Kiwi Deep Learning Challenge
This project consist in **build a German Traffic Sign Classifier**, This is a  multi-classification task on data from the link http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip
We have to implement 3 models to solve this problem:  Logistic Regression from sklearn, Logistic Regression from Tensorflow and build a Lenet Architecture from Tensorflow.

Details of this challengue can be found [here](https://github.com/KiwiCampusChallenge/Kiwi-Campus-Challenge/blob/master/Deep-Learning-Challenge.md).

A guide step by step will be found in **app.ipynb,  reports/model1.ipynb , reports/model2.ipynb , reports/model3.ipynb**

To download the data automatically and split the data  in **images/train** and **images/test** use the command
```
python app.py download
```

To train the data
```
python app.py train -m [choose the model] -d [choose the folder]
```
model could be (model1, model2 or model3)

To test the data
```
python app.py test -m [choose the model] -d [choose the folder]
```
To infer the data
If you have your own data
```
python app.py test -m [choose the model] -d [choose the folder]
```




