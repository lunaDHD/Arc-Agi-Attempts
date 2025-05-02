import json
from torch import tensor

data = json.load(open("arc-prize-2025/arc-agi_evaluation_challenges.json"))
solutions = json.load(open("arc-prize-2025/arc-agi_evaluation_solutions.json"))
newdata = []

for i in data:
    train_data = [(i["input"], i["output"]) for i in data[i]["train"]]
    test_data = (data[i]["test"][0]['input'], solutions[i][0])
    newdata.append((train_data, test_data))

json.dump(newdata, open("arc-prize-2025/arc-agi_evaluation_challenges_resampled.json", "w"))
