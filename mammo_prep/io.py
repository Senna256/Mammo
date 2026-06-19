import pydicom
import numpy as no

def load_dicom(path):
    """
    Reads a DICOM file and returns a pixel array and ds object. 
    Returns both because other function (like flip_to_standard) need the ds metadata.
    """
    ds =pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(no.float32)
    return img, ds