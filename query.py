#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from pickle import load
from rtree.index import Index, Property
from sys import argv
from typing import List

import numpy as np

def point2box(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v, v), axis=None)

def bounding_box(v: np.ndarray, r: float) -> np.ndarray:
    return np.concatenate((v-(r/2), v+(r/2)), axis=None)

def load_index(name: str, dimensions: int) -> Index:
    p = Property()
    p.dimension = dimensions
    p.dat_extension = "data"
    p.idx_extension = "index"

    rtindex = Index(name, properties=p)

    return rtindex

def knn_search(index: Index, face: np.ndarray, k: int) -> List[int]:
    return [
        id
        for id in index.nearest(point2box(face), k)
        if isinstance(id, int)
    ]

def range_search(index: Index, face: np.ndarray, r: float) -> List[int]:
    return [
        id
        for id in index.intersection(bounding_box(face, r))
        if isinstance(id, int)
    ]

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    with open("files", "rb") as r:
        files:List[str] = load(r)

    index = load_index("rtree", 128)

    face = face_encodings(load_image_file(argv[1]))[0]

    for id in knn_search(index, face, 2):
        print(files[id])

if __name__ == "__main__":
    main()
