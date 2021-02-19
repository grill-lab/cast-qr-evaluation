import pandas as pd
import json
import urllib
from tqdm import tqdm
import os
import requests


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cast_helper(file_path, year):
    data = []

    with open(file_path) as json_file:
        conversations = json.load(json_file)
        for conversation in conversations:
            q_id = conversation['number']
            turns = conversation['turn']

            for turn in range(len(turns)):
                row = []
                row.append(str(q_id) + '_' +
                           str(turns[turn]['number']))
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
                           'q_id', 'query'])

    query_list = resolved['query'].tolist()

    overall_history = []

    current_conversation = resolved.iloc[0]['q_id'].split('_')[0]
    start_idx = 0
    current_query = 0

    while current_query < len(resolved):

        history = []
        if current_conversation == resolved.iloc[current_query]['q_id'].split('_')[0]:
            history = query_list[start_idx: current_query+1]
            overall_history.append(history)
            current_query += 1
            history = []

        else:
            current_conversation = resolved.iloc[current_query]['q_id'].split('_')[
                0]
            start_idx = current_query

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
    return renamed_dataframe[['conversation_history', 'query', 'all_manual']]


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

        result_arr.append(sample_obj)

    return result_arr


def get_data(source, type):
    canard = pd.DataFrame()
    cast_y1 = pd.DataFrame()
    cast_y2 = pd.DataFrame()

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
                    'big_files/CAsT_2019_evaluation_topics_v1.0.json', 1)
                cast_y1 = pd.DataFrame(cast_y1_data, columns=[
                                       'q_id', 'query',  'conversation_history', 'all_manual'])
                cast_y1['canonical_doc'] = [None for i in range(len(cast_y1))]
                cast_y1 = get_resolved_queries(
                    'big_files/CAsT_2019_evaluation_topics_annotated_resolved_v1.0.tsv', cast_y1)
            if data == 'cast_y2':
                cast_y2_data = cast_helper(
                    'big_files/CAsT_2020_manual_evaluation_topics_v1.0.json', 2)
                cast_y2 = pd.DataFrame(cast_y2_data, columns=[
                                       'q_id', 'query', 'canonical_doc', 'conversation_history', 'all_manual'])
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
                    'big_files/CAsT_2019_evaluation_topics_v1.0.json', 1)
                cast_y1 = pd.DataFrame(cast_y1_data, columns=[
                                       'q_id', 'query',  'conversation_history', 'all_manual'])
                cast_y1['canonical_doc'] = [None for i in range(len(cast_y1))]
                cast_y1 = get_resolved_queries(
                    'big_files/CAsT_2019_evaluation_topics_annotated_resolved_v1.0.tsv', cast_y1)
            if data == 'cast_y2':
                cast_y2_data = cast_helper(
                    'big_files/CAsT_2020_manual_evaluation_topics_v1.0.json', 2)
                cast_y2 = pd.DataFrame(cast_y2_data, columns=[
                                       'q_id', 'query', 'canonical_doc', 'conversation_history', 'all_manual'])

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