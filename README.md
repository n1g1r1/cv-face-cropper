# CV module: Face Extractor
Crops faces out of the webcam image and saves them into a directory to train a CNN.

## Algorithm description

1. The algorithm asks for a label (name) and 
2. makes a new directory in `data/training/` with the scheme `NAME_HASH` which contains an unique hash value to distinguish similar or equal labels from each other. 
3. Then it activates the webcam and makes 50 pictures every 200ms of the users face in front of it and saves them into the newly created folder with the pattern `NAME_TIMEHASH.jpg`. When no face is detected, no picture will be saved.

## Requirements

Face detector algorithm needed, that returns `faces`, `eyes` and `image`. 

List of possible detectors:

- [Face Detector module by @n1g1r1](https://github.com/n1g1r1/cv-module-face-detector)

## Usage

Import the module, as well as the detector in your file and call the method like the following:

```
face_extractor.build_training_set(detector, classifier = "lbp")
```

- `detector` is the detector module you've chosen.
- `classifier` is the chosen classifier. Default: `lbp`.

### Example

In this example the module structure is like the following:
```
└── modules
    ├── face_extractor
    └── face_detector
``` 

First, import the modules:

```
from modules.face_extractor import face_extractor as extractor
from modules.face_detector import face_detector as detector
```

Then, call the `build_training_set` method and set the `detector` as the used detector method:

```
extractor.build_training_set(detector)
```
