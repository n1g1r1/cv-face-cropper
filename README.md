# CV module: Face Extractor
Crops faces out of the webcam image and saves them into a directory as training set. At the moment this uses a naive way which needs two iterations though the image to recognise a person - one to gather the training set, one to extract the features.

## Algorithm description:
1. Asks for a name
2. Makes new dir
3. Saves a bunch of images into that folder. Advanced way: Saves keypoints into that folder.

## Knowm challenges:
[x] Record images
[ ] Replace face extractor with keypoint extractor.
