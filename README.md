A simple facial recognition model designed to be run on a Sony Spresense dev board. Trained on the widerface dataset on an Apple M1 Max (32 gpu cores).

For a given image, the model should return "left", "right", "center", or "none" depending on what part of the image has a face (or if there isn't a face at all).

main.py contains the model itself if you want to train/run/test the model yourself. compressor.py uses the tflite library to compress the model into a .h file the Spresense board can use.
