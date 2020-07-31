# Requirements
`python 3.6`

`pandas`: pip install pandas

`numpy` : pip install numpy

`opencv` : pip install opencv-python

`dlib` : pip install dlib

`pandas` : pip install pandas

`sklearn` : pip install sklearn
 
## Before running:

1- [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    download this file and place it at `./Demo/`, note it's 90 MB.

2- download the [dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and extract it here `./Demo/dataset`, note it's 170 MB.


## To run the main file:

1- go to `./Demo/src`, open the terminal and type `python .\main.py`


# Folders Discription:


# [dataset folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/dataset)
contains a very small set of face images.

# [script folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/sctipt)
contains any used scripts to pre/post-process the data
or any other bash files that we might use.

# [src folder](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/src)
contains the core functional python scripts used.

# [csv_files](https://github.com/Evraa/Facial-Verification-System-Python/tree/master/Demo/csv_files)
contains all of the csv files used during the process.


`main.py`

The file responsible for comparing images, showing facial landmarks...etc

`facial_landmarks.py`

The main file that detect the face key points and extract each part individually.

`get_length`

Calculate the distances between key points.

`calc_weights`

Figures wich key point shall take more wieght in the identification process.

`identify_faces`

Responsible for the logic of identifying the faces.

`auxilary.py`

Where repeatedly used functions exist
 

## Facial Analysis
Use the permDict(dataframe) to analyze a DataFrame of faces. 
Each face has 6 length values that determine similarity,
as compared to the x scale of the face.