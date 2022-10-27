import os
import json

pwd = os.path.dirname(__file__)

with open(os.path.join(pwd, "data", "annotations.json")) as json_data:
    annotations = json.load(json_data)
with open(os.path.join(pwd, "predictions.json")) as json_data:
    predictions = json.load(json_data)

ok = 0
fail = 0
stats = {
    "normal": {"normal": 0, "carrying": 0, "threat": 0}, 
    "carrying": {"normal": 0, "carrying": 0, "threat": 0}, 
    "threat": {"normal": 0, "carrying": 0, "threat": 0}}

for file_name in predictions:
    predicted = predictions[file_name]
    actual = annotations[file_name]
    if (predicted == actual):
        print("PASS", file_name, predicted, actual)
        ok += 1
    else:
        print("FAIL", file_name, predicted, actual)
        fail += 1

    stats[actual][predicted] += 1


print("ok:", ok, "fail:", fail)
print("success rate:", ok / (ok + fail))
print(json.dumps(stats, indent=4, sort_keys=True))
