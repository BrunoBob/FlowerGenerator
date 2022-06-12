import ujson
from zipfile import ZipFile
from pathlib import Path

PLANTNET_ZIP_PATH = "plantnet_300K.zip"
BASEPATH_IN_ZIP = Path("plantnet_300K")
DESTINATION_PATH = "flowers"

# def get_flowers(metadata):
#     filter_flowers = lambda kv: True if kv[1]["organ"] == "flower" else False
#     return filter(filter_flowers, metadata)


def get_flowers_path(metadata):

    # filter on organ
    filter_flowers = lambda kv: True if kv[1]["organ"] == "flower" else False
    filtered = filter(filter_flowers, metadata.items())
    # for f in list(filtered)[:3]:
    #     print(f)

    # TODO use pathlib?
    # forge paths. They are species_id/key.jpg
    # paths = map(lambda kv: f"{kv[1]['species_id']}/{kv[0]}.jpg", filtered)
    paths = map(
        lambda kv: BASEPATH_IN_ZIP
        / f"images_{kv[1]['split']}"
        / kv[1]["species_id"]
        / f"{kv[0]}.jpg",
        filtered,
    )
    return paths


if __name__ == "__main__":
    with ZipFile(PLANTNET_ZIP_PATH) as archive:
        with archive.open("plantnet_300K/plantnet300K_metadata.json") as metadata:
            metadata = ujson.load(metadata)
            flower_paths = list(get_flowers_path(metadata))
        # with archive.open("plantnet_300K/") as image_path:
        for (idx, p) in enumerate(flower_paths):
            if idx % 1000 == 0:
                print(f"done {idx} images")
            archive.extract(str(p), "flowers")
