#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from sys import argv
from typing import List
from rtree.index import Index, Property

import numpy as np

def get_files() -> List[str]:
    return glob(f'{argv[1]}/**/*.jpg', recursive=True)

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
            # TODO i -> file map
        i += 1

    return rtindex

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    create_index('rtree', 128, get_files())

if __name__ == "__main__":
    main()
