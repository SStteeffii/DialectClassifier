import importlib.util
import os
import sys

import add_splitlabel


# import methods
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

path_Wiki = "C:/Users/Stefa/OneDrive/Dokumente/Bachelorarbeit_Shared/Data_DialectClassifier/Real_Data/Wikipedia/"
files_Wiki = [[str(path_Wiki + 'data_bar_cleaned.tsv')], [str(path_Wiki + 'data_nds_cleaned.tsv')], [str(path_Wiki + 'data_de_cleaned.tsv')]]

path_Tatoeba = "C:/Users/Stefa/OneDrive/Dokumente/Bachelorarbeit_Shared/Data_DialectClassifier/Real_Data/Tatoeba/"
files_Tatoeba = [[str(path_Tatoeba + 'bar_Tatoeba_labeled.tsv')], [str(path_Tatoeba + 'nds_Tatoeba_labeled.tsv')], [str(path_Tatoeba + 'de_Tatoeba_labeled.tsv')]]

path_MrDialect = "C:/Users/Stefa/OneDrive/Dokumente/Bachelorarbeit_Shared/Data_DialectClassifier/Translated_Data/MrDialect/"
files_MrDialect = [[str(path_MrDialect + 'bar_MrDialekt.tsv')], [str(path_MrDialect + 'nds_MrDialekt.tsv')], [str(path_MrDialect + 'de_Wiki_cleaned_resized_NewLine.tsv')]]

path_Dialektversum = "C:/Users/Stefa/OneDrive/Dokumente/Bachelorarbeit_Shared/Data_DialectClassifier/Translated_Data/Dialektversum/"
files_Dialektversum = [[str(path_Dialektversum + 'bar_Dialektversum-RespektEmpire.tsv')], [str(path_Dialektversum + 'nds_Dialektversum-Oeverstetter.tsv')], [str(path_Dialektversum + 'de_Wiki_cleaned_resized_NewLine.tsv')]]


target_size = 230  # size of the data set per dialect

# create folders:
if not os.path.exists(str(target_size)):
    os.makedirs(str(target_size))
if not os.path.exists(str(target_size) + '_final'):
    os.makedirs(str(target_size) + '_final')

# resize random:
tatoeba_resize = load_module("resize", "Real_Data/Tatoeba/2-Tatoeba_resize.py")
tatoeba_filenames_resized = tatoeba_resize.resize(target_size, files_Tatoeba)
wikipedia_filenames_resized = tatoeba_resize.resize(target_size, files_Wiki)

# resize cut file:
mrDialect_resize = load_module("resize", "Translated_Data/MrDialect/2-MrDialect-bar+nds_resize.py")
mrDialect_filenames_resized = mrDialect_resize.resize(target_size, files_MrDialect)
dialektversum_filenames_resized = mrDialect_resize.resize(target_size, files_Dialektversum)

# preprocess wikipedia:
wikipedia_preprocess = load_module("preprocess", "Real_Data/Wikipedia/4-Wikipedia_preprocess+label.py")
wikipedia_filenames_preprocessed = wikipedia_preprocess.preprocess(wikipedia_filenames_resized)

# mix data without labeling:
tatoeba_mix = load_module("mix", "Real_Data/Tatoeba/3-Tatoeba_mix_data_randomly.py")
tatoeba_mix.mix(tatoeba_filenames_resized, str(str(target_size) + '/Tatoeba_de+nds+bar_mixed.tsv'))
tatoeba_mix.mix(wikipedia_filenames_preprocessed, str(str(target_size) + '/Wikipedia_de+nds+bar_mixed.tsv'))
tatoeba_mix.mix(wikipedia_filenames_preprocessed, str(str(target_size) + '_final/' + str(target_size) + '_Wikipedia.tsv'))

# mix and label data:
dialektversum_mix = load_module("mix", "Translated_Data/Dialektversum/3-Dialektversum-bar+nds_label+mix_data_randomly.py")
dialektversum_mix.mix(dialektversum_filenames_resized, str(str(target_size) + '/Dialektversum_de+nds+bar_mixed.tsv'))
dialektversum_mix.mix(mrDialect_filenames_resized, str(str(target_size) + '/MrDialect_de+nds+bar_mixed.tsv'))

# preprocess after mix
tatoeba_preprocess = load_module("preprocess", "Real_Data/Tatoeba/4-Tatoeba_preprocess.py")
# tatoeba
tatoeba_preprocess.preprocess(str(str(target_size) + '/Tatoeba_de+nds+bar_mixed.tsv'), str(str(target_size) + '/Tatoeba_de+nds+bar_mixed.tsv')[:-4] + '_preprocessed.tsv')
tatoeba_preprocess.preprocess(str(str(target_size) + '/Tatoeba_de+nds+bar_mixed.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_Tatoeba.tsv'))
# dialektversum
tatoeba_preprocess.preprocess(str(str(target_size) + '/Dialektversum_de+nds+bar_mixed.tsv'), str(str(target_size) + '/Dialektversum_de+nds+bar_mixed.tsv')[:-4] + '_preprocessed.tsv')
tatoeba_preprocess.preprocess(str(str(target_size) + '/Dialektversum_de+nds+bar_mixed.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_Dialektversum.tsv'))
# mrdialect
tatoeba_preprocess.preprocess(str(str(target_size) + '/MrDialect_de+nds+bar_mixed.tsv'), str(str(target_size) + '/MrDialect_de+nds+bar_mixed.tsv')[:-4] + '_preprocessed.tsv')
tatoeba_preprocess.preprocess(str(str(target_size) + '/MrDialect_de+nds+bar_mixed.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_MrDialect.tsv'))

# add split label
files = [str(str(target_size) + '_final/' + str(target_size) + '_MrDialect.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_Dialektversum.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_Tatoeba.tsv'), str(str(target_size) + '_final/' + str(target_size) + '_Wikipedia.tsv')]
add_splitlabel.splitlabel(files)

