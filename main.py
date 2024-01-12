print('ekrem')
import pandas as pd
import numpy as np
import cv2 
from pathlib import Path
import os

# use images until 370
dir_path = Path(r'NTSD-complete-v1.0.1\NewTsukubaStereoDataset\illumination\lamps')
left_imgs = [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith('L')]
print(left_imgs[:10])

