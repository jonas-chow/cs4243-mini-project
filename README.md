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
```
