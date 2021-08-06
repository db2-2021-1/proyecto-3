#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from pickle import load
from sys import argv
from typing import List, Tuple

import numpy as np

def knn_search(index: List[Tuple[np.ndarray, str]], face: np.ndarray, k: int) -> List[str]:
    k_vector: List[str] = []
    best_distance = np.inf

    for point in index:
        d = np.linalg.norm(face-point[0])
        if(d < best_distance):
            best_distance = d

            if(len(k_vector) >= k):
                k_vector.pop()

            k_vector.insert(0, point[1])

    return k_vector

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    with open("index", "rb") as r:
        index: List[Tuple[np.ndarray, str]] = load(r)

    face = face_encodings(load_image_file(argv[1]))[0]
    lista_knnsearch = knn_search(index, face, 3)

    for id in lista_knnsearch:
        print(id)

if __name__ == "__main__":
    main()
