__author__ = "Jonathan Durand"
__email__ = "jonathan.drnd@gmail.com"

import cv2
import time,random
import os,shutil
import urllib.request
import zipfile
import click
import matplotlib.pyplot as plt
import numpy as np
import pickle,sys
from models import *
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from PIL import Image
from tensorflow.contrib.layers import flatten

os.getcwd()
########################################################################
ROWS=32
COLS=32
NUM_CLASSES=43


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .10 + np.random.uniform()/3.0
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def transform_image(img):
    ang_range = 5
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = augment_brightness_camera_images(img)
    return img


def data_augmentation(X, Y):
    x1=[]
    y1=[]
    for i in range(X.shape[0]):
        x1.append(X[i])
        y1.append(Y[i])
        for num in range(3):
            x1.append(transform_image(X[i]))
            y1.append(Y[i])
    return np.array(x1), np.array(y1)


def print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in download().
    """
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size
    msg = "\r- File downloading: {0:.1%}".format(pct_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

########################################################################


@click.group()
def cli():
    pass

@cli.command('download')
def download():
    url = "http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip"
    download_dir = "data/"
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a zip file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip"
    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/"
    :return:
        Nothing.
    """
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        print("Extracting from: ",url )
        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,filename=file_path,reporthook=print_download_progress)

        print("Download finished in folder data/")
        print("Done.")
    else:
        print("Data has apparently already been downloaded")
        print("Getting new, testing and training images  images/train, images/test")

    if file_path.endswith(".zip"):
        # Unpack the zip-file.
        print("We will extract only classification files ")
        download_file = download_dir + "FullIJCNN2013.zip"
        with zipfile.ZipFile(download_file, 'r') as zip:
            # printing all the contents of the zip file
            filelist = zip.filelist
            # extracting all the folders and ReadMe.txt
            for detail in filelist:
                name = detail.filename
                if name.count("/") == 2:
                    zip.extract(name, "data/")
                if name[-10:] == 'ReadMe.txt':
                    zip.extract(name, "data/")

    print("We split the data 80% training and 20% testing in /images/train and /images/test")
    X,Y= read_images()
    Xtrain,Xtest,Ytrain,Ytest = split(X,Y)
    save_images(Xtrain,Xtest,Ytrain,Ytest)


@cli.command()
@click.option('-m', default='model1', help='model1 (logistic scikit),model2 (logistic tensorflow),model3 (lenet tensorflow) , example model1')
@click.option('-d', default='/images/train/',help='Path to directory with training data ,example /images/train')
def train(m, d):
    print("Training Phase")
    if d[-1]!='/':
        d=d+"/"

    if not os.path.exists("data/FullIJCNN2013/"):
        print("Please use first the next command (python app.py download)")
        return

    if m=="model1":
        print("Task3: Logistic Regresion - Scikit")
        logistic_regression_scikit(d,True)

    if m=="model2":
        print("Task4: Logistic Regresion - Tensorflow")
        logistic_regression_tensorflow(d,True)

    if m=="model3":
        print("Task5: LeNet Architecture - Tensorflow")
        lenet_tensorflow(d,True)

@cli.command()
@click.option('-m', default='model1', help='model1 (logistic scikit),model2 (logistic tensorflow),model3 (lenet tensorflow) ,example model1')
@click.option('-d', default='/images/test/',help='Path to directory with training data ,example /images/test')
def test(m, d):
    print("Test Phase")
    if d[-1]!='/':
        d=d+"/"

    if not os.path.exists("data/FullIJCNN2013/"):
        print("Please use first the next command (python app.py download)")
        return

    if m=="model1":
        if not os.path.isfile("models/model1/saved/model1.sav"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model1 -d images/test)")
            return
        print("Task3: Logistic Regresion - Scikit")
        logistic_regression_scikit(d,False)

    if m=="model2":
        if not os.path.isfile("models/model2/saved/checkpoint"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model2 -d images/test)")
            return
        print("Task4: Logistic Regresion - Tensorflow")
        logistic_regression_tensorflow(d,False)

    if m=="model3":
        if not os.path.isfile("models/model3/saved/checkpoint"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model3 -d images/test)")
            return
        print("Task5: LeNet Architecture - Tensorflow")
        lenet_tensorflow(d,False)

@cli.command()
@click.option('-m', default='model1', help='model1 (logistic scikit),model2 (logistic tensorflow),model3 (lenet tensorflow) ,example model1')
@click.option('-d', default='/images/user/',help='Path to directory with training data ,example /images/user')
def infer(m, d):
    print("Inference Phase - Path ",d)
    if d[-1]!='/':
        d=d+"/"

    if not os.path.exists("data/FullIJCNN2013/"):
        print("Please use first the next command (python app.py download)")
        return

    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("images/user"):
        os.makedirs("images/user")

    #get label of every class  (extracted from data/ReadMe.txt)
    label = get_label()
    label = np.array(label)
    Xinfer = []
    path_dir=d

    for file in os.listdir(path_dir):
        if os.path.isfile(os.path.join(path_dir, file)):
            if not file.endswith(".txt") and not file.endswith(".zip") \
                    and not file.endswith(".gzip") and not file.endswith(".md"):
                Xinfer.append(np.array(Image.open(os.path.join(path_dir, file))))
    Xinfer = np.array(Xinfer)
    _ = np.zeros(1)

    Xinfer, _ = preprocess_data(Xinfer, _)
    predictions = []

    if m == "model1":
        if not os.path.isfile("models/model1/saved/model1.sav"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model1 -d images/test)")
            return
        Xinfer = Xinfer.reshape([Xinfer.shape[0], -1])
        print("Task6: Inference in Logistic Regresion - Scikit")
        predictions = logistic_regression_scikit(d,False, 1, Xinfer)
        print(predictions)

    if m == "model2":
        if not os.path.isfile("models/model2/saved/checkpoint"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model2 -d images/test)")
            return

        print("Task6: Inference in Logistic Regresion - Tensorflow")
        Xinfer = Xinfer.reshape([Xinfer.shape[0], -1])
        predictions = logistic_regression_tensorflow(d,False, 2, Xinfer)
        print(predictions)

    if m == "model3":
        if not os.path.isfile("models/model3/saved/checkpoint"):
            print("Does not exist, model trained")
            print("Please train first with the next command (python app.py test -m model3 -d images/test)")
            return
        print("Task6: Inference in LeNet Architecture - Tensorflow")
        predictions = lenet_tensorflow(d,False, 3, Xinfer)
        print(predictions)

    predictions = np.array(predictions)
    cont = 0

    for file in os.listdir(path_dir):
        if os.path.isfile(os.path.join(path_dir, file)):
            if not file.endswith(".txt") and not file.endswith(".zip") \
                    and not file.endswith(".gzip") and not file.endswith(".md"):
                filepath = path_dir + file
                txt = str("Class ")
                txt = txt + str(predictions[cont])
                txt = txt + str(": ")
                txt = txt + label[predictions[cont]]
                print("label: ", txt)
                x = plt.imread(filepath)
                plt.imshow(x)
                plt.title(txt)
                plt.show()
                cont = cont + 1


def save_images(Xtrain,Xtest,Ytrain,Ytest):
    """
        Save images in directorys
        images/test -> testing data   -- format png
        images/train ->training data  -- format png
    """
    if not os.path.exists("images"):
        os.makedirs("images")

    if os.path.exists("images/test"):
        shutil.rmtree("images/test")
        time.sleep(5)
    os.makedirs("images/test")

    it=0
    for i in range(NUM_CLASSES):
        folder_class="images/test/"+str(i)+"/"

        if not os.path.exists(folder_class):
            os.makedirs(folder_class)

        count=(Ytest==i).sum()
        for j in range(count):
            file_name="00"
            if j<10:
                file_name = file_name + "00" + str(j) + ".png"
            else:
                file_name = file_name + "0" + str(j) + ".png"
            Image.fromarray(Xtest[it]).save(folder_class+file_name)
            it=it+1

    if os.path.exists("images/train"):
        shutil.rmtree("images/train")
        time.sleep(5)
    os.makedirs("images/train")

    it = 0
    for i in range(NUM_CLASSES):
        folder_class = "images/train/" + str(i) + "/"

        if not os.path.exists(folder_class):
            os.makedirs(folder_class)

        count = (Ytrain == i).sum()
        for j in range(count):
            file_name = "00"
            if j < 10:
                file_name = file_name + "00" + str(j) + ".png"
            else:
                file_name = file_name + "0" + str(j) + ".png"
            Image.fromarray(Xtrain[it]).save(folder_class + file_name)
            it = it + 1


def split(X,Y, thres=0.8):
    """
        Split dataset
        Arguments:
            - X: Image data
            - Y: Labels
            - thres: Percentage of split in training
        Returns:
            - Xtrain: Image Training data
            - Ytrain: Labels Training data
            - Xtest: Image Test data
            - Ytest: Labels Test data

    """

    n=X.shape[0]
    order=np.argsort(Y)
    X2=X
    Y2=Y

    for i in range(n):
        X2[i]=X[order[i]]
        Y2[i]=Y[order[i]]

    X=X2
    Y=Y2
    Xtrain=[]
    Xtest=[]
    Ytrain=[]
    Ytest=[]

    start=0
    for i in range(NUM_CLASSES):
        # We choose percentage (thres) to training for every class.
        count= (Y==i).sum()
        select= int(count*thres+0.6)
        #We have to leave at least one image to testing
        if select==count:
            select=select-1
        #random
        choice= np.random.choice(count,select,replace=False)
        for j in range(count):
            if j in choice:
                Xtrain.append(X[j+start])
                Ytrain.append(Y[j+start])
            else:
                Xtest.append(X[j+start])
                Ytest.append(Y[j+start])
        start=start+count

    return np.array(Xtrain),np.array(Xtest),np.array(Ytrain),np.array(Ytest)

def get_label(pathreadme="data/FullIJCNN2013/ReadMe.txt"):
    """ Read ReadMe.txt and get label of the classes
        return description of classes
    """
    label=[]
    startline=0
    with open(pathreadme, mode='rb') as f:
        for line in f:
            x=str(line.strip())
            strline = str(startline) + " ="
            if strline in x:
                label.append(x[6:(len(x)-1)].strip())
                startline = startline + 1
    f.close()
    return np.array(label)

def read_train_test_from_directory(path_dir):
    #Read images in directory (TRAIN or TEST)
    #return numpy of images and label class
    print("Reading data from: ",path_dir)
    Xtrain_test=[]
    Ytrain_test=[]

    for i in range(NUM_CLASSES):
        pathfile=path_dir+str(i)+"/"
        for file in os.listdir(pathfile):
            if os.path.isfile(os.path.join(pathfile, file)):
                Xtrain_test.append(np.array(Image.open(os.path.join(pathfile, file))))
                Ytrain_test.append(i)
    return np.array(Xtrain_test),np.array(Ytrain_test)

def preprocess_data(X,Y):
    """
    Preprocess image, (resize 32x32  then converts RGB images into grayscale)   and convert labels into one-hot
    Arguments:
        - X: Image data
        - Y: Labels

    Returns:
        - Preprocessed X, one-hot version of Y
    """
    X_preprocess=[]
    num_of_images=X.shape[0]

    for i in range(num_of_images):
        X_preprocess.append(np.array(Image.fromarray(X[i]).resize((32, 32))))
    X_preprocess = np.array(X_preprocess)
    X_preprocess = X_preprocess.astype('float64')
    X_preprocess = (X_preprocess - 128.) / 128.

    images_gray = np.average(X_preprocess, axis=3)
    images_gray = np.expand_dims(images_gray, axis=3)

    y_onehot = np.zeros([Y.shape[0], NUM_CLASSES])

    for i in range(Y.shape[0]):
        y_onehot[i][int(Y[i])] = 1
    Y = y_onehot
    return images_gray,Y

def read_images(path_folder="data/FullIJCNN2013/"):
    """ Read all the folder (images)
        Preprocess images - resize (32x32)
        return images and classes
    """

    X=[]
    Y=[]

    for name in os.listdir(path_folder):
        # Classes are represented by folders
        if os.path.isdir(os.path.join(path_folder, name)):
            idclass=int(name)
            path_class=os.path.join(path_folder, name)
            #read files (.ppm) for every folder
            for file in os.listdir(path_class):
                if os.path.isfile(os.path.join(path_class, file)):
                    img = Image.open(os.path.join(path_class, file))
                    img_array=np.array(img)
                    X.append(img_array)
                    Y.append(idclass)
    return np.array(X),np.array(Y)

if __name__ == '__main__':
    cli()