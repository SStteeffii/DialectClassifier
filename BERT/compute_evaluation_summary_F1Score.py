import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

# macro F1-Score for all test files for each training file - all samples mean with standard deviation

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
    # path to the file which should be read
    base_path = './Result_Bert/'
    train_files = ['wikipedia_half', 'dialektversum', 'mr_dialect']


    for train_data in train_files:
        base_path_long = base_path + train_data
        data = []
        number_samples = [20, 40, 60, 80, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]



        # name of the report, which we are interested in
        #report_name = 'tatoeba.tsv'
        reports_names = ['tatoeba.tsv', 'wikipedia_test.tsv', 'wikipedia_validation.tsv']

        for report_name in reports_names:
            for num in number_samples:
                # the number of samples, which are interesting
                num_samples = num

                # compute the runs
                reports = store_runs(base_path_long, num_samples, report_name)

                mean_report, std_report = compute_statistics(reports)
                #print(str(num_samples))
                #print(mean_report['macro'])
                #print(std_report['macro'])

                data.append([report_name, num, mean_report['macro'], std_report['macro']])

             #print(data)
        #print(data)

        # Umwandeln in ein DataFrame
        df = pd.DataFrame({
            'Test-Data': [item[0] for item in data],
            'Samples': [item[1] for item in data],
            'F1-Score': [item[2]['F1-Score'] for item in data],
            'StdDev F1-Score': [item[3]['F1-Score'] for item in data]
        })

        # Erstellen des Plots
        plt.figure(figsize=(12, 8))
        unique_files = df['Test-Data'].unique()

        for file in unique_files:
            file_df = df[df['Test-Data'] == file]
            sns.lineplot(data=file_df, x='Samples', y='F1-Score', label=file)
            plt.fill_between(file_df['Samples'], file_df['F1-Score'] - file_df['StdDev F1-Score'],
                             file_df['F1-Score'] + file_df['StdDev F1-Score'], alpha=0.2)

        plt.xscale('log')
        plt.title('F1-Score Vergleich für ' + base_path_long[base_path_long.rfind('/')+1:] + ' mit unterschiedlichen Testdateien')
        plt.xlabel('Samples')
        ticks = df['Samples']
        plt.xticks(ticks, labels=ticks)
        plt.ylabel('F1-Score')
        plt.legend(loc='right')

        # Erstellen der Tabelle
        pivot_df = df.pivot(index='Samples', columns='Test-Data', values='F1-Score').reset_index()
        for col in pivot_df.columns[1:]:  # Überspringt die erste Spalte 'Samples'
            pivot_df[col] = (pivot_df[col] * 100).round(2).astype(str) + '%'
        pivot_df = pivot_df.round(2)
        pivot_df['Samples'] = pivot_df['Samples'].astype(int)  # Samples als ganze Zahlen
        pivot_df = pivot_df.sort_values('Samples')

        # Anpassen des Layouts für den Plot und Hinzufügen der Tabelle
        plt.subplots_adjust(bottom=0.42)
        table = plt.table(cellText=pivot_df.values, colLabels=pivot_df.columns, loc='bottom', bbox=[0, -0.85, 1, 0.70],
                          cellLoc='center')

        plt.savefig(str(base_path_long[:base_path_long.rfind('/')] + '/train_' + base_path_long[base_path_long.rfind('/')+1:] + '_plot.png'), dpi=300)
        plt.show()
