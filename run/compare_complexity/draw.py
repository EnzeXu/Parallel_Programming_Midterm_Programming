import matplotlib.pyplot as plt

FONTSIZE = 17

plt.figure(figsize=(10,6), tight_layout=True)
#plotting
with open("result", "r") as f:
    lines = list(line.split() for line in f.readlines())

y_val = [(float(x[2]), float(x[3])) for x in lines]
x_val = [int(x[0]) for x in lines]

plt.plot(x_val, y_val, 'o-', linewidth=3)
plt.plot(x_val, [float(y_val[8][0])] * len(x_val),  '--', linewidth=2)
plt.plot(x_val, [float(y_val[15][0])] * len(x_val), '--', linewidth=2)
#customization
plt.xticks(range(0, len(x_val), 5))
plt.xlabel('Complexity of the Equation', fontsize=FONTSIZE)
plt.ylabel('Average of log(Search Space)', fontsize=FONTSIZE)
plt.title('The Reduction of Search Space after Apply Control Vairable', fontsize=FONTSIZE)
plt.legend(labels=['before control variable', 'after control variable', 'brute force can discover', 'state-of-art SR can discover'], fontsize=FONTSIZE)
plt.savefig('compare.png', dpi=300, bbox_inches='tight')