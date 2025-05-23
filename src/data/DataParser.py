import os
import xml.etree.ElementTree as ET

file_location = "../../data/raw/polska.xml"
root = ET.fromstring(countrydata)


def parse_data(file_path, src_file_format):
    root = ET.parse(file_path)
    root.
    