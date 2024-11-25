import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

samples = 10000

val=np.random.multivariate_normal((4, 4), [[5, 3.3],[3.3, 5]], samples)
draw_1, draw_2 = np.zeros(samples), np.zeros(samples)
for i in tqdm(range(samples)):
    draw_1[i] = np.argmax(np.random.multinomial(1, [6 / 25, 8/25, 3/25, 3/25, 2/25, 3/25, 0, 0, 0, 0]))  
    draw_2[i] = np.argmax(np.random.multinomial(1, [6 / 25, 8/25, 3/25, 3/25, 2/25, 3/25, 0, 0, 0, 0]))  

print(np.sum(draw_1 == draw_2)) # 0 20.6% chance, and the chance of RC being reshuffled is equal to 6/29 = 20.7%!!!

rank1 = [1, 7, 4, 2, 4, 3, 2, 7, 4, 2, 7, 3, 2]
rank2 = [1, 6, 6, 1, 3, 4, 2, 8, 9, 1, 6, 3, 1]
plt.plot(np.arange(len(rank1)), rank1, "o-", label = "Rank before reshuffling")
plt.plot(np.arange(len(rank1)), rank2,"o-",  label = "Rank after reshuffling")
plt.xlabel("Mouse index (at least once in RC)")
plt.ylabel("Chasing rank")
plt.legend()
plt.show()


# rc_1 = np.round(val[:, 0]) == draw_1
# rc_2 = np.round(val[:, 1]) == draw_2
# conserved_rc = np.sum(np.logical_and(rc_1, rc_2))/np.sum(rc_1)
# print(conserved_rc)



# plt.scatter(np.round(val[:, 0]), np.round(val[:, 1]), alpha=0.07, c = 'blue', s = 30)
# plt.plot(np.round(val[:, 0]), alpha = 0.5)
# plt.plot(np.round(val[:, 1]), alpha = 0.5)

# plt.show()