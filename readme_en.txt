[Required Libraries]
numpy

scipy

scikit-learn

scikit-image

tqdm

av

timm

PySide6

OpenCV

PyTorch
v1.6 or later
If using a GPU, please install the compatible version.

torchvision
v0.7.0 or later

FrEIA Flows
https://github.com/vislearn/FrEIA/archive/cc5cf5ebee08f9bb762bab5a6535c11d19ccb026.zip
After unzipping, rename the folder to something easy to understand.
Open the folder location from a command prompt (e.g., cd c:\FrEIA).
Run python setup.py develop.

Install other libraries as needed.

[Executable Files]
Training Executable File
MAIN_TRAINER.py

Detection Executable File
MAIN_DETECTOR_MOV.py

[Operation Flow]
- Assuming high-accuracy anomaly detection.
1. Fix the camera in a fixed position.
2. Keep the target object's position and orientation constant (slight misalignment is acceptable).
3. It is very important to provide lighting and maintain a consistent environment brightness.
4. Make the training image and detection image the same (keep the detection size, video size, and trim constant).

・Collecting Images for Training
1. Run MAIN_DETECTOR_MOV.py.
2. Determine [DETECTION SIZE] and [VIDEO SIZE].
3. Press the [TAKE PICTURE] button and select a folder to save the training images.
4. To improve detection accuracy, left-click the mouse to select an area as needed (if you make a mistake, press the [STOP] button and then the [CLEAR] button).
5. Press the spacebar to save the images.
6. Once you have saved enough images, press the [STOP] button to exit (we recommend having around 250 images).
7. Press the [SAVE] button to save the settings so that you can perform actual detection in the same environment.

・Training
1. Run MAIN_TRAINER.py.
2. If the training images are not sequentially numbered, press [RENUMBER], select the folder where the images are saved, and rename the files sequentially.
3. Set the [DETECTION SIZE] in MAIN_DETECTOR_MOV.py to the same as the [TRAIN SIZE] in MAIN_TRAINER.py.
4. Press the [OPEN] button and select the folder containing the training images.
5. Press the [START] button to start training (training should complete in approximately 1-3 hours).

Preparing for Detection
1.1. Run MAIN_DETECTOR_MOV.py.
1. Press the [LOAD] button to load the settings used when collecting images.
2. Press the [OPEN] button and select the latest training data file with the pt extension.
3. Set the [KERNEL SIZE] (the detection unit for anomaly detection; the larger the value, the less likely small anomalies will be detected).
4. Set the [THRESHOLD FOR DETECTION]. This is the threshold for determining an anomaly. Use the [CHECK] button to select the folder containing photos of good products and obtain and set the recommended value (select the image size used during training).
5. Set the [THRESHOLD FOR AREA SIZE]. This is the threshold for determining the size of an area detected as an anomaly.
6. Set the [NON SIMILER THRESHOLD]. This is the additional value used to determine whether an image is abnormal when it is completely different from the trained image (normally, no changes are required).
7. Press the [SAVE] button to save the settings.

・Detection
1. Press the [LOAD] button to load the settings.
2. Press the [START] button to begin detection.
3. After executing DETETED_PICTURE = DETECTION(frameB) in MAIN_LOOP(), the number of defects will be stored in the DETECT_COUNTER variable.
4. If necessary, add external output or other processing immediately after DETETED_PICTURE = DETECTION(frameB).
