import numpy as np
import pandas as pd

from collections import defaultdict
from parse_environment import get_environment, available_datasets, get_available_datasets
from pathlib import Path

current_folder = Path(__file__).parent
base_ds_path = get_environment()


def make_idx_frll_morphs(genuine_dataset="frll") -> pd.DataFrame:
    """
    Generates a dataframe containing the indexed data of FRLL-Morphs dataframe

    :param genuine_dataset: Selects the source of the bona fide; by default, the source is taken from FRLL dataset.
    Possible options are (amsl and frll)
    :return: The DataFrame containing the indexed data of FRLL-Morphs
    :rtype: pd.DataFrame
    """
    get_available_datasets()

    base_morph_dir = Path(available_datasets["morph"]["FRLL-Morphs"])
    if genuine_dataset == "frll":
        genuine_frll_dir = Path(available_datasets["face"]["FRLL"])

    elif genuine_dataset == "amsl":
        genuine_frll_dir = Path(available_datasets["morph"]["AMSL"])
    else:
        print("Genuine must be frll or amsl. Falling back to FRLL")
        genuine_frll_dir = Path(available_datasets["morph"]["FRLL"])
        genuine_dataset = "frll"

    frll_dict = defaultdict(list)

    # Creating the labels for each subdivision
    # Morphed face
    morphed_label_dict = {
        "morph_amsl": "amsl",
        "morph_facemorpher": "facemorpher",
        "morph_opencv": "opencv",
        "morph_stylegan": "stylegan",
        "morph_webmorph": "webmorph",
    }

    # Morphed faces
    for subdivision in morphed_label_dict.keys():
        local_path = base_morph_dir / subdivision
        for file in local_path.iterdir():
            frll_dict["Path"].append(file)
            frll_dict["ID1"].append(float(file.stem[:3]))
            frll_dict["ID2"].append(float(file.stem[4:7]))
            frll_dict["Morph"].append(True)
            frll_dict["Dataset"].append("FRLL-Morphs")
            frll_dict["Subdivision"].append(morphed_label_dict[subdivision])

    # Genuine faces
    if genuine_dataset == "frll":
        genuine_label_dict = {
            "neutral": "neutral",
            "smiling": "smiling",
        }
    else:
        genuine_label_dict = {
            "neutral_front": "neutral",
            "smiling_front": "smiling",
        }

    for subdivision in genuine_label_dict.keys():
        local_path = genuine_frll_dir / subdivision
        for file in local_path.glob("*.jpg"):
            frll_dict["Path"].append(file)
            frll_dict["ID1"].append(float(file.stem[:3]))
            frll_dict["ID2"].append(np.nan)
            frll_dict["Morph"].append(False)
            frll_dict["Dataset"].append("FRLL-Morphs" if genuine_dataset == "frll" else "AMSL")
            frll_dict["Subdivision"].append(genuine_label_dict[subdivision])

    data = pd.DataFrame(frll_dict, columns=["Path", "ID1", "ID2", "Morph", "Dataset", "Subdivision"])

    return data


if __name__ == "__main__":
    frll_df = make_idx_frll_morphs()
    print(frll_df.head().to_string())
