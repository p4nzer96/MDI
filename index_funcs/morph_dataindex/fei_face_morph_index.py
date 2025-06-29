from collections import defaultdict
from pathlib import Path

import pandas as pd

from parse_environment import get_environment

current_folder = Path(__file__).parent
base_ds_path = get_environment()


def make_idx_ff_morphs():
    # Bonafide images
    bonafide_folder = base_ds_path / "FEI Face Database" / "original"

    # Morphed images
    morphed_folder = base_ds_path / "FEIFaceMorph"

    # Output path
    output_path = current_folder / "fei_face_morphs"

    true_label = "bonafide"
    fake_label = "morphed"

    image_dict = defaultdict(list)

    for file in bonafide_folder.iterdir():

        if file.stem.split("-")[1] != "11":
            continue

        image_dict["Path"].append(file.resolve())
        image_dict["Label"].append(true_label)
        image_dict["ID1"].append(file.stem.split("-")[0])
        image_dict["ID2"].append(None)
        image_dict["Morph"].append(False)
        image_dict["Subdivision"].append(None)

    for file in morphed_folder.iterdir():
        image_splits = file.stem.split("_")

        image_dict["Path"].append(file.resolve())
        image_dict["Label"].append(fake_label)
        image_dict["ID1"].append(image_splits[1].split("-")[0])
        image_dict["ID2"].append(image_splits[2].split("-")[0])
        image_dict["Morph"].append(True)
        image_dict["Subdivision"].append("_".join(image_splits[3:]))

    fei_df = pd.DataFrame(image_dict, columns=["Path", "ID1", "ID2", "Morph", "Subdivision"])
    fei_df.to_csv(output_path / "fei_face_morphs_idx.csv", index=False)
