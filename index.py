#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from sys import argv
from typing import List

import numpy as np

def get_files() -> List[str]:
    return glob(f'{argv[1]}/**/*.jpg', recursive=True)

def point2box(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v, v), axis=None)

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    for file in get_files():
        for face in face_encodings(load_image_file(file)):
            print(face)
            # TODO

if __name__ == "__main__":
    main()
