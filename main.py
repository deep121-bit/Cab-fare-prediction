from src.data_loader import load_data


def main():

    file_path = "data/raw/train.csv"

    df = load_data(file_path)

    print("\n Dataset Loaded Successfully\n")

    print(df.head())


if __name__ == "__main__":
    main()