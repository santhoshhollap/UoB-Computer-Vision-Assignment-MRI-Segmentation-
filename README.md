# UoB-Computer-Vision-Assignment-MRI-Segmentation-

The objective is to design and develop an 2D and 3D image processing algorithm that will segment a consecutive MRI axial cross section. The classes that are needed to segment are : Air (class0), Skin/Scalp(class1), Skull(class2), CSF(class3), Gray Matter(class4) and White Matter(class5). Analyze the outcome between 2D and 3D, and conclude the observations with appropriate reason.

# Usage :
## Command to run the final code (configuration can be done within Final.py(init) to set 2D or 3D processing)
```python3 Final.py <input_path> <output_path>```
### Command to run the 2d segmentation code standalone (configuration can be done within Final2D.py(init) to set which experimented methods to use)
```python3 Final2D.py <input_path> <output_path>```
### Command to run the 3d segmentation code standalone (configuration can be done within Final3D.py(init) to set which experimented methods to use)
```python3 Final3D.py <input_path> <output_path>```
### Command to run the validation(configuration can be done within validation.py to set color image or gray image evaluation)
```python3 validation.py <Groundtruth_path> <output_path>```
### Command to create .png from Brain.mat(path1 - original image, path2 - segmented groundtruth image)
```python3 imageExtraction.py <Path1> <Path2>```

# Note:
1. Standalone code will output the images in color whereas the Final code (has only optimal method) will output the gray image. 
2. Validation code can be executed on both kinds of images.
3. For different methods that can be experimented, please refer to CV.pdf
4. 2DFuzzyMultiOtsu folder has 2D gray segmented output images (obtained from Final.py under 2D configuration)
5. 3DFuzzyMultiOtsu folder has 3D gray segmented output images saved slices by slices (obtained from Final.py under 3D configuration)
