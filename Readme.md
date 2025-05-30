# 3D Printing Slicers Software Tests:
This Repository contains the source code to test 2 already built different slicing software ([Slic3r](https://github.com/slic3r/Slic3r) - [PrusaSclicer](https://github.com/prusa3d/PrusaSlicer)) and 1 slicing library for SLM printing technology built on Python ([PYSLM](https://github.com/drlukeparry/pyslm)).

The goal for these tests is to showcase the differences between the different slicing software and be able to generate g-code that represents the printing for a 3D model for metal 3D printing.

## Before you start:
1. These tests were carried out on the **Windows** operating system, so please use Windows if you want to run the scripts and tests on the slicing software.
2. Make sure that you have Python 3.10 or higher installed on your machine to be able to run the scripts [Link to install Python](https://www.python.org/downloads/release/python-3116/).
3.  After you install Python, please clone the repository and then navigate from the terminal to the repo directory: `cd 'PATH_TO_REPO_DIR'`
4. Create a virtual environment in Python using the command: `python -m venv venv`.
5. Activate the virtual environment using: `.\venv\Scripts\activate`.
6. Then, install required libraries: `pip install -r requirement.txt`.
7. Now you can start your tests.

## Structure:
- Inside the `src` directory, you can find three different scripts to run all the tests needed:
  - slic3r_slicing.py
  - prusa_slicing.py
  - pyslm_lib_slicing.py
  
  Each one uses specific slicing software/library to slice a 3D model and then generates an output G-code file. Also, there are other helper modules that the scripts use, located inside the `src` folder.

- Inside the `3D_models` folder. There are some 3D models to use as examples for the testing process.

- The `gcode` folder you can store your gcode files generated from the tests.

## How to run:
Please navigate to the `src` directory to apply to run the test directly: `cd src`.

### Slic3r:

![slic3r_bash](https://github.com/user-attachments/assets/460de02f-e6bc-4d68-af49-619be608fec5)

To show all the attributes that you can use to run the Slic3r test, please type: `python slic3r_slicing.py -h`

An example of using slic3r:

`python slic3r_slicing.py --model-path ../3D_models/mini-top.stl --layer-height 0.2 --scale 3 --layer-num 10 --gcode-output-path ../gcode/test.nc --infill-image-pixel-size 2 2 --infill-noise 5 --contour-power 100 --infill-power 20 --rotate 90`

### PrusaSlicer:

![prusa_bash](https://github.com/user-attachments/assets/d67660e2-5af1-4a62-8b44-7fa2bb6881d2)

To show all the attributes that you can use to run the PrusaSlicer test, please type: `python prusa_slicing.py -h`

An example of using PrusaSlicer:

`python prusa_slicing.py --model-path ../3D_models/mini-top.stl --layer-height 10 --gcode-output-path ../gcode/test.gcode --infill-image-pixel-size 2 2 --infill-noise 20 --contour-power 100 --hollowing-enable`

### PYSLM:

![pyslm_lib_bash](https://github.com/user-attachments/assets/91073204-03d1-4fc2-80a7-a7440ca59582)

To show all the attributes that you can use to run the PYSLM test, please type: `python pyslm_lib_slicing.py -h`

An example of using PYSLM:

`python pyslm_lib_slicing.py --model-path ../3D_models/mini-top.stl --layer-height 0.1 --gcode-output-path ../gcode/test.nc --infill-image-pixel-size 1 1 --infill-noise 10 --contour-power 100 --contour-power 100 --infill-outer-contour 1 --infill-volume-offset 0.1 --infill-resolution 10`
