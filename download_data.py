"""Download the landing gear datasets from UWE Bristol Research Data Repository."""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASETS = {
    "LGD_S_DATASET_1_BogiePitchTrimmer_PressureDrop_OverFlightCycles.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/1/LGD_S_DATASET_1_BogiePitchTrimmer_PressureDrop_OverFlightCycles.csv",
    "LGD_S_DATASET_2_TyrePressureIndicator_PressureDrop_OverFlightCycles.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/2/LGD_S_DATASET_2_TyrePressureIndicator_PressureDrop_OverFlightCycles.csv",
    "LGD_S_DATASET_3_TyrePressureIndicator_TemperatureRise_DuringLanding.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/3/LGD_S_DATASET_3_TyrePressureIndicator_TemperatureRise_DuringLanding.csv",
    "LGD_S_DATASET_4_LandingGearBrakes_MaxTemperature_OverFlightCycles.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/4/LGD_S_DATASET_4_LandingGearBrakes_MaxTemperature_OverFlightCycles.csv",
    "LGD_S_DATASET_5_LandingGearBrakes_Deceleration_DuringTakeoff.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/5/LGD_S_DATASET_5_LandingGearBrakes_Deceleration_DuringTakeoff.csv",
    "LGD_S_DATASET_6_LandingGearBrakes_Deceleration_OverFlightCycles.csv":
        "https://researchdata.uwe.ac.uk/id/eprint/717/6/LGD_S_DATASET_6_LandingGearBrakes_Deceleration_OverFlightCycles.csv",
}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in DATASETS.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            print(f"[skip] {filename} already exists")
            continue
        print(f"[download] {filename} ...")
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  -> {size_mb:.1f} MB")
    print("Done. All datasets are in the data/ directory.")


if __name__ == "__main__":
    main()
