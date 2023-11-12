from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

@app.route("/predictPrice")
def predictPrice():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = nn.Linear(4, 64)
            self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
            self.out = nn.Linear(64, 1)


        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.out(x)
            return x

    net = Net()

    checkpoint_path = "theEnd.pth"
    custom_data = pd.DataFrame({
        'bed': [10.0],
        'bath': [10.0],
        'acre_lot': [0.08],
        'house_size': [50.0]
    })
    torch_custom_data = torch.tensor(custom_data.values, dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])


    net.eval()

    with torch.no_grad():
        predicted_price = net(torch_custom_data)

    
    return str(predicted_price.item())
price = 0
@app.route("/getprice", methods=["POST", "GET"])
def getprice():
    data = request.json['data'] 
    bedroom,bathroom,area,houseSize = data[0],data[1],data[2],data[3]
    bedroom = float(bedroom)
    bathroom = float(bathroom)
    area = float(area)
    houseSize = float(houseSize)
    print(bedroom)
    price = predictPrice(bedroom,bathroom,area,houseSize)
    print(area)
    return jsonify(price)

def predictPrice(bed,bath,area,houseS):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = nn.Linear(4, 64)
            self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
            self.out = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.out(x)
            return x

    net = Net()

    checkpoint_path = "theEnd.pth"
    custom_data = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'acre_lot': [area],
        'house_size': [houseS]
    })
    torch_custom_data = torch.tensor(custom_data.values, dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])


    net.eval()

    with torch.no_grad():
        predicted_price = net(torch_custom_data)

    
    return str(predicted_price.item())





if __name__ == "__main__":
    app.run(debug=True)
