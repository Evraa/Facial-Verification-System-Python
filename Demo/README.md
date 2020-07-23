# Requirements
`python 3.6`

`pandas`: pip install pandas

`numpy` : pip install numpy

`opencv` : pip install opencv-python

`dlib` : pip install dlib

`pandas` : pip install pandas

# How to run:

1- [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    download this file and place it at `./Demo/`

2- place your data of images at `./Demo/dataset`

3- go to `./Demo/src`, open the terminal and type `python facial_landmarks.py`, to create the face key points at the `csv_example.csv` file.

4- to find similar faces: TODO


## Folders Discription:

shape_predictor_68_face_landmarks

# [dataset folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/dataset)
contains a very small set of face images.

# [script folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/sctipt)
contains any used scripts to pre/post-process the data
or any other bash files that we might use.

# [src folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/src)
contains the core functional python scripts used.

`main.py`

The file responsible for comparing images, showing facial landmarks...etc

`facial_landmarks.py`

The main file that detect the face key points and calculate the distances

`auxilary.py`

Where repeatedly used functions exist
 

## Facial Analysis
Use the permDict(dataframe) to analyze a DataFrame of faces. 
Each face has 6 length values that determine similarity,
as compared to the x scale of the face.