import pandas as pd
import json
import urllib
from tqdm import tqdm
import os
import requests
import wandb
import matplotlib.pyplot as plt


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cast_helper(file_path, year, q_rels):
    data = []

    with open(file_path) as json_file:
        conversations = json.load(json_file)
        for conversation in conversations:
            q_id = conversation['number']
            turns = conversation['turn']

            for turn in range(len(turns)):
                row = []
                true_q_id = str(q_id) + '_' + str(turns[turn]['number'])
                row.append(true_q_id)
                row.append(turns[turn]['raw_utterance'])

                if year == 2:
                    row.append(turns[turn]['manual_canonical_result_id'])

                conversation_history = []
                resolved_queries = []
                for history in range(0, turn+1):
                    conversation_history.append(
                        turns[history]['raw_utterance'])
                    if year == 2:
                        resolved_queries.append(
                            turns[history]['manual_rewritten_utterance'])

                row.append(conversation_history)
                row.append(resolved_queries)
                row.append(q_rels[true_q_id] if true_q_id in q_rels else [])
                data.append(row)

    return data


def trim_context(history):
    first_turn_context = history[:2]
    first_turn_context = ' '.join(first_turn_context)
    queries = []

    try:
        queries = history[2::2]
    except:
        pass

    return [first_turn_context] + queries


def insert_first_turns(df):
    idx = 0

    while idx < len(df):
        if df.iloc[idx]['Question_no'] == 1:
            row_value = [[], '', df.iloc[idx]['History'],
                         0, df.iloc[idx]['History'][0]]
            df1 = df[0:idx]
            df2 = df[idx:]

            df1.loc[idx] = row_value
            df = pd.concat([df1, df2])
            idx += 1

            df.index = [*range(df.shape[0])]

        idx += 1

    return df


def get_resolved_queries(resolved_path, source):
    resolved = pd.read_csv(resolved_path, sep='\t', header=None, names=[
                           'conversation_id', 'query'])

    query_list = resolved['query'].tolist()

    overall_history = []

    current_conversation = resolved.iloc[0]['conversation_id'].split('_')[0]
    start_idx = 0
    current_query = 0

    while current_query < len(resolved):

        history = []
        if current_conversation == resolved.iloc[current_query]['conversation_id'].split('_')[0]:
            history = query_list[start_idx: current_query+1]
            overall_history.append(history)
            current_query += 1
            history = []

        else:
            current_conversation = resolved.iloc[current_query]['conversation_id'].split('_')[
                0]
            start_idx = current_query
            history = [query_list[start_idx]]
            overall_history.append(history)
            current_query += 1

    source['all_manual'] = overall_history
    return source



def carnard_helper(dataframe):
    renamed_dataframe = dataframe.rename(
        columns={"History": "conversation_history", "Question": "query", })
    resolved_queries = renamed_dataframe['Rewrite'].tolist()
    all_manual = []

    current_query = 0
    start_idx = 0
    previous_question_no = renamed_dataframe.iloc[0]['Question_no']

    while current_query < len(renamed_dataframe):
        history = []
        if renamed_dataframe.iloc[current_query]['Question_no'] >= previous_question_no:
            history = resolved_queries[start_idx:current_query]
            all_manual.append(history)
            history = []
            previous_question_no = renamed_dataframe.iloc[current_query]['Question_no']
            current_query += 1

        else:
            previous_question_no = renamed_dataframe.iloc[current_query]['Question_no']
            start_idx = current_query
            all_manual.append(history)
            current_query += 1

    renamed_dataframe['all_manual'] = all_manual
    return renamed_dataframe[['conversation_history', 'query', 'all_manual', 'q_id']]


def parse_json_str(json_str, result_len):
    full_data = json.loads(json_str)
    result_arr = []

    for sample in range(result_len):
        sample_obj = {
            'all_raw_queries': None,
            'all_manual_queries': None,
            # 'query': None,
            'q_id': None,
            'canonical_doc': None
        }

        sample_obj['all_raw_queries'] = full_data['conversation_history'][str(
            sample)]
        sample_obj['all_manual_queries'] = full_data['all_manual'][str(sample)]
        # sample_obj['query'] = full_data['query'][str(sample)]
        sample_obj['q_id'] = full_data['q_id'][str(
            sample)]
        sample_obj['canonical_doc'] = full_data['canonical_doc'][str(sample)]
        sample_obj['q_rels'] = full_data['q_rels'][str(sample)]

        result_arr.append(sample_obj)

    return result_arr


def get_data(source, type):
    canard = pd.DataFrame()
    cast_y1 = pd.DataFrame()
    cast_y2 = pd.DataFrame()
    
    NIST_qrels=["/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/2019qrels.txt",
                '/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/2020qrels.txt']
    q_rels = {}
    for q_rel_file in NIST_qrels:
        with open(q_rel_file) as NIST_fp:
            for line in NIST_fp.readlines():
                q_id, _, d_id, score = line.split(" ")
                if int(score) < 3:
                    # ignore some of the worst ranked
                    continue
                if q_id not in q_rels:
                    q_rels[q_id] = []
                q_rels[q_id].append(d_id)

    if type == 'train':
        for data in source:
            if data == 'canard':
                canard = pd.read_json(
                    'big_files/CANARD_Release/train.json')  # path to Canard
                canard['History'] = canard.apply(
                    lambda row: trim_context(row['History']), axis=1)
                canard = insert_first_turns(canard)
                canard['canonical_doc'] = [None for i in range(len(canard))]
                canard['q_id'] = [None for i in range(len(canard))]
                canard = carnard_helper(canard)
            if data == 'cast_y1':
                cast_y1_data = cast_helper(
                    'big_files/CAsT_2019_evaluation_topics_v1.0.json', 1, q_rels)
                cast_y1 = pd.DataFrame(cast_y1_data, columns=[
                                       'q_id', 'query',  'conversation_history', 'all_manual', 'q_rels'])
                cast_y1['canonical_doc'] = [None for i in range(len(cast_y1))]
                cast_y1 = get_resolved_queries(
                    'big_files/CAsT_2019_evaluation_topics_annotated_resolved_v1.0.tsv', cast_y1)
            if data == 'cast_y2':
                cast_y2_data = cast_helper(
                    'big_files/CAsT_2020_manual_evaluation_topics_v1.0.json', 2, q_rels)
                cast_y2 = pd.DataFrame(cast_y2_data, columns=[
                                       'q_id', 'query', 'canonical_doc', 'conversation_history', 'all_manual', 'q_rels'])
    if type == 'test':
        for data in source:
            if data == 'canard':
                canard = pd.read_json(
                    'big_files/CANARD_Release/test.json')  # path to Canard
                canard['History'] = canard.apply(
                    lambda row: trim_context(row['History']), axis=1)
                canard = insert_first_turns(canard)
                canard['canonical_doc'] = [None for i in range(len(canard))]
                canard['q_id'] = [None for i in range(len(canard))]
                canard = carnard_helper(canard)
            if data == 'cast_y1':
                cast_y1_data = cast_helper(
                    'big_files/CAsT_2019_evaluation_topics_v1.0.json', 1, q_rels)
                cast_y1 = pd.DataFrame(cast_y1_data, columns=[
                                       'q_id', 'query',  'conversation_history', 'all_manual', 'q_rels'])
                cast_y1['canonical_doc'] = [None for i in range(len(cast_y1))]
                cast_y1 = get_resolved_queries(
                    'big_files/CAsT_2019_evaluation_topics_annotated_resolved_v1.0.tsv', cast_y1)
            if data == 'cast_y2':
                cast_y2_data = cast_helper(
                    'big_files/CAsT_2020_manual_evaluation_topics_v1.0.json', 2, q_rels)
                cast_y2 = pd.DataFrame(cast_y2_data, columns=[
                                       'q_id', 'query', 'canonical_doc', 'conversation_history', 'all_manual', 'q_rels'])

    result = canard.append(cast_y1).append(cast_y2)
    result_len = len(result)
    result.reset_index(drop=True, inplace=True)

    result = result.to_json()
    result = parse_json_str(result, result_len)

    return result

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

class Info_Plotter:
    def __init__(self):
        pass
        
    def tabulate_rewrites(self, samples):
        table = wandb.Table(columns=["Raw query", "Re-write", "Manual", "Model output", "ndcg_cut_3", 'recall_1000'])
        for sample_obj in samples:
            model_output = sample_obj['model output'] if "model output" in sample_obj else ""
            ndcg_cut_3 = sample_obj['ndcg_cut_3'] if "ndcg_cut_3" in sample_obj else ""
            recall_1000 = sample_obj['recall_1000'] if "recall_1000" in sample_obj else ""
            table.add_data(sample_obj['raw query'], sample_obj['re-write'], sample_obj['manual query'], model_output, ndcg_cut_3, recall_1000)
        return {'rewrites table': table}
    
    def get_turn_counts(self, samples):
        counts = {}
        for turn in samples:
            id = turn['q_id'].split('_')[1]
            if id in counts:
                counts[id] += 1
            else:
                counts[id] = 1
        return counts
    
    def per_turn_plots(self, samples):
        metrics = ['recall_500', 'recall_1000', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_1000', 'map_cut_1000']
        charts = {}
        turn_counts = self.get_turn_counts(samples)
        for metric in metrics:
            metric_dict = {}
            for turn in samples:
                id = turn['q_id'].split('_')[1]
                if id in metric_dict:
                    try:
                        metric_dict[id] += turn[metric] #not all turns might have a given metric
                    except:
                        pass
                else:
                    metric_dict[id] = turn[metric]

            turns = [*metric_dict]
            values = [metric_dict[turn]/turn_counts[turn] for turn in turns]
            
            fig, ax = plt.subplots()
            ax.set_ylabel(metric)
            ax.bar(turns, values)
            
            charts[f"per_turn_{metric}"] = wandb.Image(fig)
            
        return charts

