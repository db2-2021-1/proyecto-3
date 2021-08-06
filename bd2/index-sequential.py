#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from pickle import dump
from sys import argv
from typing import List, Tuple

import numpy as np

def get_files(directory: str) -> List[str]:
    return glob(f"{directory}/**/*.jpg", recursive=True)

def point2box(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v, v), axis=None)

def create_index(files: List) -> List[Tuple[np.ndarray, str]]:
    v: List[Tuple[np.ndarray, str]] = []

    for file in files:
        for face in face_encodings(load_image_file(file)):
            v.append((face, file.split("/")[-2]))

    return v

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    files = get_files(argv[1])
    with open("index", "wb") as w:
        dump(create_index(files), w)

if __name__ == "__main__":
    main()
