from celebA import random_ff_mask
import pickle

maskList = [random_ff_mask() for i in range(16)]

pickle.dump(maskList, open("test_mask.pkl", "wb+"))