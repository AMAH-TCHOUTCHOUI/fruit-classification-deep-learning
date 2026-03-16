
import requests

url = "http://127.0.0.1:8000/predict"

# Mets ici le bon chemin vers ton image
files = {"file": open(r"C:\Users\AUSARE\Desktop\projet_fruits_api\samples\r1_7.jpg", "rb")}

response = requests.post(url, files=files)
print("Réponse API :", response.json())
