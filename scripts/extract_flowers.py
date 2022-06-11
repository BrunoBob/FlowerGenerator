from typing import Optional, Set
import sys
import os
import ujson
from zipfile import ZipFile
from pathlib import Path

# path to plantnet zip
PLANTNET_ZIP_PATH = "plantnet_300K.zip"

# internal path to plantnet (in the zip), shouldn't need to be changed
BASEPATH_IN_ZIP = Path("plantnet_300K")

# path were images will be saved
DESTINATION_PATH = Path("flowers")


"""
Gets an iterator over paths of flowers (in the zip)

specify a list of species ids in `ignore_species` to remove some species from the list
"""


def get_flowers_path(metadata, ignore_species: Optional[Set[str]] = None):

    # filter on organ, to get flower pictures
    filter_flowers = lambda kv: True if kv[1]["organ"] == "flower" else False
    filtered = filter(filter_flowers, metadata.items())

    # if ignore species is set, ignore supplied species to ignore.
    if ignore_species is not None:
        remove_ignore_species = (
            lambda kv: False if kv[1]["species_id"] in ignore_species else True
        )
        filtered = filter(remove_ignore_species, filtered)

    # forge paths. They are species_id/key.jpg
    paths = map(
        lambda kv: BASEPATH_IN_ZIP
        / f"images_{kv[1]['split']}"
        / kv[1]["species_id"]
        / f"{kv[0]}.jpg",
        filtered,
    )
    return paths


if __name__ == "__main__":

    # open zipfile
    with ZipFile(PLANTNET_ZIP_PATH) as archive:

        # open json file in zipfile
        with archive.open("plantnet_300K/plantnet300K_metadata.json") as metadata:
            metadata = ujson.load(metadata)

            # get flower paths
            flower_paths = list(get_flowers_path(metadata))

        # iterate over flower paths and write them to the destination dir
        try:
            os.mkdir(DESTINATION_PATH)
        except FileExistsError:
            if DESTINATION_PATH.is_dir() and not os.listdir(DESTINATION_PATH):
                pass
            else:
                print("Directory is file or is not empty!")
                sys.exit(1)
                
        for (idx, p) in enumerate(flower_paths):
            if idx % 1000 == 0:
                print(f"done {idx} images")
            with archive.open(str(p)) as image:
                with open(DESTINATION_PATH / f"{p.name}", "wb") as f:
                    f.write(image.read())
