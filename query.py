#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from pickle import load
from rtree.index import Index, Property
from sys import argv
from typing import List

import numpy as np

def point2box(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v, v), axis=None)

def load_index(name: str, dimensions: int) -> Index:
    p = Property()
    p.dimension = dimensions
    p.dat_extension = "data"
    p.idx_extension = "index"

    rtindex = Index(name, properties=p)

    return rtindex

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    with open("files", "rb") as r:
        files:List[str] = load(r)

    index = load_index("rtree", 128)

    face = face_encodings(load_image_file(argv[1]))[0]

    for id in index.nearest(point2box(face), 1):
        if isinstance(id, int):
            print(files[id])


if __name__ == "__main__":
    main()
