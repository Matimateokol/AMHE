import requests
import shutil

datasets = [ 'polska', 'germany50', 'janos-us-ca']

default_xml_url_path = "https://sndlib.put.poznan.pl/download/sndlib-networks-xml/"
default_txt_url_path = "https://sndlib.put.poznan.pl/download/sndlib-networks-native/"

headers = {'User-Agent': 'Mozilla'}

def fetch_data(data_format):
    base_url = ''
    write_mode = 'wb'
    if data_format == 'xml':
        base_url = default_xml_url_path
    elif data_format == 'txt':
        base_url = default_txt_url_path
    else:
        print("Unrecognized data format - supported formats are 'xml' and 'txt' (ascii)!")
        return
    
    for dataset in datasets:
        full_url = f"{base_url}{dataset}.{data_format}"
        response = requests.get(full_url, headers=headers, stream=True)
        with open(f"./data/raw/{dataset}.{data_format}", write_mode) as outfile:
            shutil.copyfileobj(response.raw, outfile)
    
fetch_data('txt')