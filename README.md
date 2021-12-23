# Lip_detection_under_surgical_mask

The following repository contains the code for Lip movement detection under surgical mask. Which can be used by various online platforms to focus on the person speaking in the crowd wearing mask just by using video feeds.

## Description 

Since, the pandemic hit us in 2020 it was exceedingly difficult for people to go out like they used to before the pandemic. Nowadays, everyone must carry a mask. It's mandatory to wear a mask every time when the person is outside or talking to someone.

It's easy for a human to detect who is speaking by combing the sound and movement of the person where as it is difficult for a machine to detect who is speaking just by judging the movement by camera feed. 

This is the crux of our project to detect Lip movement just by the camera feed under a surgical face mask using computer vision techniques. 

## Getting Started

### Dependencies

* OS version: Windows 10/11 or Ubantu or Mac OS
* Coding Environment: VS Code or PyCharm or anyother suitable coding playform to run python
* Python Version 3.8.10
* TensorFlow Version 2.4.1
* Mrcnn Version 0.1
* Cuda Version 11.5

For more necessary packages please refer to packages.txt in the repository
```
pip install -r packages.txt
```

### Installing

* Download all the files in one folder along with the model provided in the drive link above.
* Change the location of the model in fullpipelineV2.py.
* Before running install all the necessary dependencies in your coding environment using terminal.

### Executing program

* Simply run the program using compile and run.
* Before executing wear a face mask.
* The camera will start and it will be able to detect facemask and key features. Then would keep tracking the feature points in the frame.
* It will show No Mask in the terminal if the person is not wearing the mask. 

## Authors

Contributors names and contact info


Abhay Karade
[GitHub](https://github.com/AbhayKarade)

Ashwij Kumbla
[GitHub](https://github.com/Ashwij3)

Himanshu Gautam
[GitHub](https://github.com/Himanshu12328)

Sumukh Sreenivasarao Balakrishna
[Mail](sbalakrishna@wpi.edu)


## Acknowledgments
[Prof. Michael Gennert](https://www.wpi.edu/people/faculty/michaelg)

[Evaluation of Lucas-Kanade based optical flow algorithm](https://ieeexplore.ieee.org/abstract/document/9018982)

[ORB (Oriented FAST and Rotated BRIEF)](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)

[Mask R-CNN](https://ieeexplore.ieee.org/document/8237584)

