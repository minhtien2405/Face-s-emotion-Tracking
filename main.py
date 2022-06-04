import json
from tracking import tracking
with open("D:\\emotion_tracking\\dataset\\product.json") as file_object:
    data = json.load(file_object)

print(data)

for i in data["data_saling"]:
    print(tracking.emotion_tracking(i["product_navigation"]))
