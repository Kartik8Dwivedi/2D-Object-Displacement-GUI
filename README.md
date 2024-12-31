# File Theft Security Using 2D Object Displacement Detection

## Overview

This project introduces an efficient and robust methodology for detecting the translation and rotation of 2D objects, such as documents or sheets of paper, using only two images. The primary focus is on estimating the transformation matrix that encapsulates both positional and orientation changes, a crucial aspect of modern computer vision tasks. The approach is engineered to overcome challenges such as image quality variations, partial occlusions, and significant transformations while maintaining computational efficiency.

## Features

- **Homography Matrix Estimation**: Uses advanced feature matching techniques combined with homography estimation.
- **Real-Time Performance**: Ensures resilience against noise and distortions, enabling its application in real-time scenarios.
- **Document Mishandling Detection**: Addresses document mishandling detection in environments such as office spaces.
- **GUI Interface**: Provides a user-friendly graphical interface for selecting images and viewing results.

## Requirements

- Python
- OpenCV
- NumPy
- Matplotlib
- Tkinter

## Installation

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/Kartik8Dwivedi/File-Theft-Security
   cd file-theft-security
   ```

2. Download the required packages.

```bash
    pip install opencv-python numpy
```

Note: Tkinter and Matplotlib are included in the Python standard library.

3. **Run the Application**:

   ```bash
   python GUI_2D_OBJECT_DISPLACEMENT.py
   ```

4. **Using the GUI**:
   - Click on the **Browse** buttons to select the two images.
   - Click **Get Transformation Matrix** to calculate and display the results.
   - The transformation matrix will be displayed in the text area below.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for providing powerful image processing tools.
- Tkinter for making GUI development easy.
- The open-source community for contributing to libraries and resources used in this project.
