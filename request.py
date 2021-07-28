import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'TV':2, 'radio':9, 'newspaper':6})

print(r.json())