# setup flask
from re import I
import torch
from flask import Flask
from flask_restful import Resource, Api, reqparse

from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import requests

app = Flask(__name__)
api = Api(app)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
  
net = torch.load('model.pt', map_location='cpu')
COMPARING_API = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
headers = {"Authorization": "Bearer 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"}

class Health(Resource):
  def post(self):
    parser = reqparse.RequestParser()
    parser.add_argument('groceries', required=True)
    args = parser.parse_args()
    column_names = ["Alcohol", "Meat", "Cereals", "Fruits", "Milk", "Starchy Roots", "Vegetable"]
    weight_percentages = np.zeros(len(column_names))
    groceries = json.loads(args['groceries'])
    total_g = 0
    for grocery in groceries:
      total_g += grocery['amount_g']
    def convertToDatabaseAppropriate(amount_g):
      return (amount_g/total_g) * 100

    # categorize groceries
    for grocery in groceries:
      response = requests.post(COMPARING_API, headers=headers, json={
        "inputs": {
          "source_sentence": grocery["category"],
          "sentences": column_names
        }
      })
      response = response.json()
      if type(response) != list: 
        if response["error"]:
          return {'status': -1, 'msg': "HuggingFace: "+response["error"]}, 400
        return {'status': -1, 'msg': 'Wating for HuggingFace API'}, 400
      index_of_match = response.index(max(response))
      weight_percentages[index_of_match] = weight_percentages[index_of_match] + convertToDatabaseAppropriate(grocery["amount_g"])

    print("weights: ", weight_percentages)

    # preprocess data
    weight_percentages = np.array(weight_percentages).astype(np.float32)
    weight_percentages = torch.from_numpy(weight_percentages)

    # make prediction
    prediction = net(weight_percentages)
    # get the number inside a tensor
    prediction = prediction.item()
    print("prediction: ", prediction)

    
    
    return {'status': 0, 'msg': prediction}

# Test that tesseract is running correctly
class Test(Resource):
  def get(self):
    return{
      'status': 'success',
      'message': '🚀 All systems go!'
    }, 200
        
api.add_resource(Health, '/health')
api.add_resource(Test, '/test')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)