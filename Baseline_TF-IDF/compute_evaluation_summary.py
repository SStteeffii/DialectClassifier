import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


def read_classification_report(data_path: str) -> dict:
    df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    output_dict = dict()
    metrics = ['Precision', 'Recall', 'F1-Score']
    for t in df.itertuples():
        values = [x for x in t[1].split(' ') if len(x) > 0 and x != 'avg']
        cls = values[0]
        output_dict[cls] = dict()
        for value, metric in zip(values[1:4], metrics):
            output_dict[cls][metric] = value
    return output_dict


def store_runs(base_path: str, num_samples: int, report_name: str):
    
    # get the runs
    runs = os.listdir(base_path)
    reports = dict()
    
    for run in runs:
        
        # compute the total path
        total_path = base_path + '/' + run + '/' + str(num_samples) + '/' + report_name
        
        # load the current classification report
        report_act = read_classification_report(total_path)
        
        for cls, metrics in report_act.items():
            
            if cls not in reports:
                reports[cls] = dict()
            
            for metric, value in metrics.items():
                if metric not in reports[cls]:
                    reports[cls][metric] = [value]
                else:
                    reports[cls][metric].append(value)
    return reports


def compute_statistics(reports: dict) -> tuple:
    
    mean_report, std_report = dict(), dict()
    
    for cls, values in reports.items():
        
        if cls not in mean_report:
            mean_report[cls] = dict()
            std_report[cls] = dict()
        
        for metric, value in values.items():
            mean_report[cls][metric] = np.mean([float(x) for x in value])
            std_report[cls][metric] = np.std([float(x) for x in value])
    return mean_report, std_report

    
if __name__ == '__main__':
    data = []
    number_samples = [20, 40, 60, 80, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    # path to the file which should be read
    base_path = './Result_SupportVectorMachine/wikipedia_half'
    
    # name of the report, which we are interested in
    report_name = 'tatoeba.tsv'

    for num in number_samples:
        # the number of samples, which are interesting
        num_samples = num

        # compute the runs
        reports = store_runs(base_path, num_samples, report_name)

        mean_report, std_report = compute_statistics(reports)
        #print(str(num_samples))
        #print(mean_report['macro'])
        #print(std_report['macro'])

        data.append([num, mean_report['macro'], std_report['macro']])

    print(data)

    df = pd.DataFrame({
        'Samples': [item[0] for item in data],
        'Precision': [item[1]['Precision'] for item in data],
        'Recall': [item[1]['Recall'] for item in data],
        'F1-Score': [item[1]['F1-Score'] for item in data],
        'StdDev Precision': [item[2]['Precision'] for item in data],
        'StdDev Recall': [item[2]['Recall'] for item in data],
        'StdDev F1-Score': [item[2]['F1-Score'] for item in data]
    })

    # Erstellen des Plots
    plt.figure(figsize=(12, 8))

    for metric in ['Precision', 'Recall', 'F1-Score']:
        sns.lineplot(data=df, x='Samples', y=metric, label=metric)
        plt.fill_between(df['Samples'], df[metric] - df[f'StdDev {metric}'], df[metric] + df[f'StdDev {metric}'],
                         alpha=0.2)

    plt.xscale('log')
    ticks = df['Samples']
    plt.xticks(ticks, labels=ticks)

    # Werte auf der Linie anzeigen
    #for i in range(len(df)):
     #   plt.annotate(f"{df['F1-Score'][i]:.2f}", (df['Samples'][i], df['F1-Score'][i]), textcoords="offset points",
    #                 xytext=(0, 10), ha='center')

    plt.title(str('Metrik Vergleich für Macro Average für verschiedene Sample-Größen für die Trainingsdaten ' + base_path[base_path.rfind('/')+1:] + ' und den Testdaten ' + report_name[:-4]))
    plt.xlabel('Samples')
    plt.ylabel('Metrics')
    plt.legend(loc='right')

    # Extrahieren und Anpassen der Daten für die Tabelle
    df['Precision'] = (df['Precision'] * 100).round(2).astype(str) + '%'
    df['Recall'] = (df['Recall'] * 100).round(2).astype(str) + '%'
    df['F1-Score'] = (df['F1-Score'] * 100).round(2).astype(str) + '%'

    table_df = df[['Samples', 'Precision', 'Recall', 'F1-Score']]
    table_df['Samples'] = table_df['Samples'].astype(int)  # Samples als ganze Zahlen
    table_data = table_df.values  # Runden auf zwei Dezimalstellen

    # Anpassen des Layouts für den Plot und Hinzufügen der Tabelle
    plt.subplots_adjust(bottom=0.4)  # Erhöhen Sie den unteren Rand, um Platz für die Tabelle zu schaffen
    table = plt.table(cellText=table_data, colLabels=table_df.columns, loc='bottom', bbox=[0, -0.8, 1, 0.65], cellLoc='center')  # Platzierung und Größe der Tabelle

    plt.savefig(str(base_path[:base_path.rfind('/')] + '/train_' + base_path[base_path.rfind('/')+1:] + '_test_' + report_name[:-4] + '_plot.png'), dpi=300)
    plt.show()

