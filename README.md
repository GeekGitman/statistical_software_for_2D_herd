### Analyse herd behaviors
---
The folder contains Python source code for this sofware. Compatible with Linux/MacOS/Windows.

#### Environment
You need these packages to build it successgully:

* Python2.7
* Numpy >=1.6.0
* Scipy >=0.15.0
* Matplotlib (Please use 'pip install')
* wxpython >=2.8

By the way, it is recomended to install Anaconda in advance.

#### Run
Run in terminal:

```
cd sta
python run.py
```

#### Other issues that you should consider :
* In 'data' directory there are two '.csv' files to tell you the format of input data. Other files that want
to be loaded must strictly obey the format. Please note that the first row of input file is the number of frames
that the data contains.
You can also refer to 'produce_sampledata.py' for details.

* The menu selection 'saveAllFigure' will save figures calculated from all Quantitative. 
* Quantitative Descriptions:
    - All quantitatives are listed in sta/config/name.txt file.
    - The calculating methods are expounded in 'Calculating Methods of Quantitative in Sheep Herds.pdf' file.
