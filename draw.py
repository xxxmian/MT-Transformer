import matplotlib.pyplot as plt
import json
with open('draw_loss.json','r') as f:
    loss = json.load(f)
plt.plot(loss)
plt.show()