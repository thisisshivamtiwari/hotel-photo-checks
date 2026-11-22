import sys

from . import dataset

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build-data":
        labels = dataset.build()
        print(len(labels), "images")
    else:
        print("usage: python -m photo_checks build-data")
