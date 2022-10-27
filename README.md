hi bois


First, get the images in their carrying, normal, threat folders
Then run the following:
```
# sorts them into training, validation, and testing sets
$ python3 data_classifier.py

# creates the datasets that pytorch uses
$ python3 dataset.py

# trains the model
$ python3 main.py

# evaluates the model against the test set, where X is the model number
$ python3 inference.py --m ./model_X

# compares your inference with the actual test set
$ python3 compare.py
```

If you want to test your model on one image or on some other directory, can do
```
$ python3 inference.py --m <path to model> --t <path to image>
```
