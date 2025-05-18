import requests
import shutil

headers = {'User-Agent': 'Mozilla'}
urlfile = "https://sndlib.put.poznan.pl/download/sndlib-networks-xml/polska.xml"

request = requests.get(urlfile, headers=headers)
response = request.status_code

with open("../../data/raw/polska.xml", 'wb') as outfile:
    shutil.copyfileobj(response, outfile)