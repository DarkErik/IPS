import DataLoader

PREPROCESSED_DATASET_FOLDER = "data\\preprocessed"
UTK_FACE_FOLDER = "data\\UTKFace"
UTK_DATASET_NAME = "utk"

PREPROCESS_DATA = True


def main():
    if PREPROCESS_DATA:
        DataLoader.preprocess_UTK_images(UTK_FACE_FOLDER, PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)

    dataset = DataLoader.load_dataset_from_preprocessed(PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)

if __name__ == '__main__':
    main()