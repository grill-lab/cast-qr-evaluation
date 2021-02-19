from src.utils import download_from_url
import os

download_from_url("https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
                  'big_files/CAsT_2020_manual_evaluation_topics_v1.0.json')

download_from_url("https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv",
                  'big_files/CAsT_2019_evaluation_topics_annotated_resolved_v1.0.tsv')

download_from_url("https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json",
                  'big_files/CAsT_2019_evaluation_topics_v1.0.json')

download_from_url("https://obj.umiacs.umd.edu/elgohary/CANARD_Release.zip",
                  'big_files/CANARD_Release.zip')
os.system(f'unzip big_files/CANARD_Release.zip -d big_files')
