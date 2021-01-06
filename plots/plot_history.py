import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("saved_positions.csv")

print(df.keys())

fig = plt.figure()

for particleId in range(0, 2):
  particleId = 1
  historyFiltered = df[df['id'] == particleId]
  plt.plot(historyFiltered['x'], historyFiltered['z'], 'x')

plt.savefig("plot.pdf")