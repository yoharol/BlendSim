import igl
import numpy as np
import pandas as pd

df = pd.read_csv("Weights.csv", header=None)

rows, cols = df.shape

weights = df.to_numpy()

igl.write_dmat("Weights.dmat", weights)

test = igl.read_dmat("Weights.dmat")
print(test - weights)