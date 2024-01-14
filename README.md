# Visual Odometry

## Overview
Visual Odometry (VO) is a computer vision technique that estimates the motion of a camera by analyzing consecutive frames from a camera. This project implements monocular visual odometry, a subfield of VO, using Python and OpenCV.
![Data](https://github.com/ekrrems/Visual-Odometry/blob/main/NTSD-complete-v1.0.1/NewTsukubaStereoDataset/illumination/lamps/L_00003.png) | ![MatLab Image](https://de.mathworks.com/help/examples/vision/win64/VisualOdometryExample_08.png)
:-------------------------:|:-------------------------:
Tsukuba Dataset Example         | Estimated Camera Trajectory


## Features
- Monocular Visual Odometry
- [To be added] Stereo Visual Odometry

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/ekrrems/Visual-Odometry
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Set the data directory as the folder contains sequence images
2. Run the main script
   ```
   python monocular_odometry.py
   ```
## Results
Your Visual Odometry path can be seen in the pop-up browser 

## Licence 
This project is licenced under the [MIT Licence](https://github.com/ekrrems/Visual-Odometry/blob/main/LICENSE)

## Acknowledgements
- [MatLab Monocular Visual Odometry](https://de.mathworks.com/help/vision/ug/monocular-visual-odometry.html)
- [CVLAB](https://home.cvlab.cs.tsukuba.ac.jp/home)
