# File Theft Security Using 2D Object Displacement Detection

## Overview

This project introduces an efficient and robust methodology for detecting the translation and rotation of 2D objects, such as documents or sheets of paper, using only two images. The primary focus is on estimating the transformation matrix that encapsulates both positional and orientation changes, a crucial aspect of modern computer vision tasks. The approach is engineered to overcome challenges such as image quality variations, partial occlusions, and significant transformations while maintaining computational efficiency.

## Features

- **Homography Matrix Estimation**: Uses advanced feature matching techniques combined with homography estimation.
- **Real-Time Performance**: Ensures resilience against noise and distortions, enabling its application in real-time scenarios.
- **Document Mishandling Detection**: Addresses document mishandling detection in environments such as office spaces.
- **GUI Interface**: Provides a user-friendly graphical interface for selecting images and viewing results.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Tkinter

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/your-username/file-theft-security.git
   cd file-theft-security
   ```
