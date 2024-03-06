# Drawing-Trainer

Environment Setup
```
conda create -n myenv python=3.7
conda activate myenv
pip install opencv-python==4.6.0.66
pip install mediapipe==0.8.8.1
pip install protobuf==3.20.3
```

Run program
```
python project.py
```

1. Only Index finger should be raised to draw
2. Only Index & Middle fingers should be raised to select/click
3. Pose estimator only detects curls and only for left arm
