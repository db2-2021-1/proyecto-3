#!/usr/bin/env python3

from face_recognition.api import load_image_file, face_encodings
from glob import glob
from sys import argv
from typing import List

def get_files() -> List[str]:
    return glob(f'{argv[1]}/**/*.jpg', recursive=True)

def main() -> None:
    if(len(argv) < 2):
        exit(1)

    for file in get_files():
        f = load_image_file(file)

        # TODO
        print(face_encodings(f))

if __name__ == "__main__":
    main()
