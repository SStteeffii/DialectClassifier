import matplotlib.pyplot as plt
import numpy as np

# Daten
data_wikipedia = {
    'precision': [0.51, 0.77, 0.97, 0.75, 0.74],
    'recall': [0.97, 0.26, 0.59, 0.61, 0.63],
    'f1-score': [0.67, 0.39, 0.74, 0.60, 0.60]
}

data_tatoeba = {
    'precision': [0.81, 0.85, 0.99, 0.88, 0.89],
    'recall': [0.91, 0.73, 0.96, 0.87, 0.88],
    'f1-score': [0.85, 0.78, 0.97, 0.87, 0.88]
}

# Labels für die x-Achse
labels = ['de', 'bar', 'nds', 'macro avg', 'weighted avg']

# Breite der Balken
barWidth = 0.35

# Setze die Position der Balken auf der x-Achse
r1 = np.arange(len(data_wikipedia['precision']))
r2 = [x + barWidth for x in r1]

# Erstellen der Balkendiagramme
plt.figure(figsize=(12, 10))

plt.bar(r1, data_wikipedia['f1-score'], color='skyblue', width=barWidth, edgecolor='grey', label='Wikipedia F1-Score')
plt.bar(r2, data_tatoeba['f1-score'], color='lightgreen', width=barWidth, edgecolor='grey', label='Tatoeba F1-Score')

# Allgemeine Layouteinstellungen
plt.xlabel('F1-Scores', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(data_wikipedia['precision']))], labels)
plt.ylabel('Percentage')
plt.title('Vergleich der F1-Scores für die Baseline Wörterbücher - Wikipedia vs. Tatoeba')
plt.legend(loc='upper left', bbox_to_anchor=(-0.05, -0.1), fancybox=True, shadow=True, ncol=1)

# Tabelle
table_data = [data_wikipedia['f1-score'], data_tatoeba['f1-score']]
columns = ['Wikipedia', 'Tatoeba']
plt.subplots_adjust(bottom=0.3)
table_data = np.array(table_data) * 100
table_data = np.round(table_data, 2)
table_data = [[f"{value}%" for value in row] for row in table_data]
table = plt.table(cellText=table_data, rowLabels=['Wikipedia', 'Tatoeba'], colLabels=labels, loc='bottom', bbox=[0.1, -0.6, 0.8, 0.3], cellLoc='center')

plt.subplots_adjust(left=0.2, bottom=0.5)

plt.savefig(str('./' + 'Result_Plot_new.png'), dpi=300, bbox_inches='tight')
plt.show()

