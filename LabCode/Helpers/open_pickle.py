import pickle

filepath = r"Z:\t1Team\Data\2025-09-19_cooldown\OPX\TATQ01_bTa_Sapphire_02\q3\q3_2025_10_02\q3_18_51_08_SS_Cal.pkl"

with open(filepath, "rb") as file:
    data = pickle.load(file)
    print(data.keys())
    print(data["confusion"])