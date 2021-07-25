#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from pickle import dump
from rtree.index import Index, Property
from sys import argv
from typing import List

import numpy as np

def get_files(directory: str) -> List[str]:
    return glob(f'{directory}/**/*.jpg', recursive=True)

def point2box(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v, v), axis=None)

def create_index(name: str, dimensions: int, files: List[str]) -> Index:
    p = Property()
    p.dimension = dimensions
    p.dat_extension = 'data'
    p.idx_extension = 'index'

    rtindex = Index(name, properties=p)

    i = 0
    for file in files:
        for face in face_encodings(load_image_file(file)):
            rtindex.insert(i, point2box(face))
        i += 1

    return rtindex

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    files = get_files(argv[1])
    create_index('rtree', 128, files)

    with open("files", "wb") as w:
        dump(files, w)

if __name__ == "__main__":
    main()
