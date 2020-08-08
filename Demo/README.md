# Requirements
`python 3.6`

`pandas`: pip install pandas

`numpy` : pip install numpy

`opencv` : pip install opencv-python

`dlib` : pip install dlib

`pandas` : pip install pandas

`sklearn` : pip install sklearn

`keras` : pip install Keras

`tensorflow` : pip install --upgrade tensorflow

`openface`: For Windows and Anaconda
        
        + Open the command line from within anaconda.
        + If required, install git (conda install -c anaconda git)
        + git clone https://github.com/cmusatyalab/openface.git
        + cd openface
        + pip install -r requirements.txt
        + IF there's a problem with pandas, delete it from requirements.txt and install it manually
        + python setup.py install
        + Refresh

 
## Before running:

1- [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    download this file and place it at `./Demo/`, note it's 90 MB. It's responsible for Detecting faces, Predicting the 68 facial points, and apply affine tranformation on face images.

2- Download the [dataset](https://drive.google.com/file/d/12Uik_DInDR9YfnHS1rwcLE9QchoeiPLZ/view?usp=sharing) and extract it here `./Demo/dataset`, note it's 70 MB.

3- Prepare `open_face.h5`, found in this [repo](https://github.com/krsatyam1996/Face-recognition-and-identification), clone or download it, and place the file `open_face.h5` in `./Demo/src/`. It's responsible for Extracting facial features from images that's been transformed.



# What to run?
+ Go to Demo/src and type `python .\main.py` and follow the instructions.




# Folders Discription:


# [dataset folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/dataset)
Obviously!

# [script folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/sctipt)
contains any used scripts to pre/post-process the data
or any other bash files that we might use.

# [csv_files](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/csv_files)
contains all of the csv files used during the process.


# [src folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/src)
contains the core functional python scripts used.

Most important files:

`main.py`

The file responsible for comparing images, showing facial landmarks...etc

`facial_landmarks.py`

The main file that detect the face key points and extract each part individually.

`auxilary.py`

Where repeatedly used functions exist
 

## Facial Analysis
Use the permDict(dataframe) to analyze a DataFrame of faces. 
Each face has 6 length values that determine similarity,
as compared to the x scale of the face.