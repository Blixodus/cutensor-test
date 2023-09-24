import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('dim_data_numones.csv', delimiter=';')
plt.xscale('log')
for i in range(0, 40):
    plt.plot(data['Dimension length'], data['Number of dimensions (' + str(i) + ')'])
plt.xlabel("Dimension length")
plt.ylabel("Number of dimensions")
plt.title("Number of dimensions vs. dimension length")
plt.gcf().set_size_inches(12.8, 9.6)
plt.savefig("dim_plot_numones.png")
