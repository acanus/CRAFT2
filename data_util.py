from lib import *

def load_data(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    return data

# if __name__ == "__main__":
#     data = load_data(r"D:\tr_vip\datasets\synthtext\SynthText\train_gt.pkl")
#     print(len(data))