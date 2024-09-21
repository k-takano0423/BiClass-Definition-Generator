from openai import OpenAI
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

CONFIG = json.load(open('./config_file/config.json', 'r'))
PROMPT_DICT = json.load(open(f"./config_file/{CONFIG['LANGUAGE']}_prompt.json", 'r'))
client = OpenAI(
    api_key = CONFIG['OPENAI_API_KEY'],
)

def fn_compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def generate_text(prompt, model_name="gpt-4o-mini-2024-07-18", temperature=0):
    messages = [
                    {"role": "user", "content": prompt}
                ]
    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
    return response.choices[0].message.content.strip()

def make_first_prompt(df, SEED_GEN_TRUE_DATA_NUM=10, SEED_GEN_FALSE_DATA_NUM=10):
    prompt = PROMPT_DICT['first_prompt']
    input_test_list = []
    for no, text in enumerate(df[df['label'] == 1].sample(SEED_GEN_TRUE_DATA_NUM, weights='importance', random_state=0)['text']):
        input_test_list.append(f"True{no+1}: {text}")
    for no, text in enumerate(df[df['label'] == 0].sample(SEED_GEN_FALSE_DATA_NUM, weights='importance', random_state=0)['text']):
        input_test_list.append(f"False{no+1}: {text}")
    return prompt + '\n'.join(input_test_list)

def make_classify_prompt(text, definition_statement):
    prompt = definition_statement + PROMPT_DICT['classify_prompt'] + text
    return prompt

def make_prompt_turning_prompt(definition_statement, false_negative_text_list, false_positive_text_list):
    prompt = definition_statement + PROMPT_DICT['prompt_turning_prompt']
    if len(false_negative_text_list) > 0:
        prompt = prompt + PROMPT_DICT['false_negative_prompt']
        for text in false_negative_text_list:
            prompt = prompt + '\n' + text
    if len(false_positive_text_list) > 0:
        prompt = prompt + PROMPT_DICT['false_positive_prompt']
        for text in false_positive_text_list:
            prompt = prompt + '\n' + text
    return prompt

def turning_step_func(df, definition_statement, model_name="gpt-4o-mini-2024-07-18", MAX_SAMPLING_FALSE_NEGA_DATA_NUM=2, MAX_SAMPLING_FALSE_POSI_DATA_NUM=2):
    true_df = df[df['label']==1].copy()
    false_negative_text_list = []
    for text in tqdm(true_df.sample(len(true_df), weights='importance', random_state=0)['text']):
        prompt = make_classify_prompt(text, definition_statement)
        pred_label = generate_text(prompt, model_name)
        if 'False' in pred_label:
            false_negative_text_list.append(text)
        if len(false_negative_text_list) == MAX_SAMPLING_FALSE_NEGA_DATA_NUM:
            break
    false_df = df[df['label']==0].copy()
    false_positive_text_list = []
    for text in tqdm(false_df.sample(len(false_df), weights='importance', random_state=0)['text']):
        prompt = make_classify_prompt(text, definition_statement)
        pred_label = generate_text(prompt, model_name)
        if 'True' in pred_label:
            false_positive_text_list.append(text)
        if len(false_positive_text_list) == MAX_SAMPLING_FALSE_POSI_DATA_NUM:
            break
    if len(false_negative_text_list) + len(false_positive_text_list) == 0:
        return '', definition_statement, True

    prompt = make_prompt_turning_prompt(definition_statement, false_negative_text_list, false_positive_text_list)
    print(prompt)
    generated_output = generate_text(prompt, model_name)
    prompt = PROMPT_DICT['extract_definition_prompt'] + generated_output
    turning_definition_statement = generate_text(prompt, model_name)

    return generated_output, turning_definition_statement, False

def read_data():
    train_df = pd.read_excel(f"{CONFIG['DATA_PATH']}/train.xlsx",engine='openpyxl') # , encoding='shift-jis')
    kfold = StratifiedKFold(n_splits=CONFIG['N_SPLIT'], shuffle=True, random_state=0)
    for fold, (train_index, valid_index) in enumerate(kfold.split(X=train_df, y=train_df['label'])):
        train_df.loc[valid_index, 'cv_flag'] = fold
    train_df['cv_flag'] = train_df['cv_flag'].astype(np.int32)
    test_df = pd.read_excel(f"{CONFIG['DATA_PATH']}/test.xlsx",engine='openpyxl') # , encoding='shift-jis')
    return train_df, test_df

def label_predict(df, definition_statement, model_name="gpt-4o-mini-2024-07-18"):
    pred_label_list = []
    for text in tqdm(df['text']):
        prompt = make_classify_prompt(text, definition_statement)
        pred_label = generate_text(prompt, model_name)
        label = -100
        if 'True' in pred_label:
            label = 1
        if 'False' in pred_label:
            label = 0
        if label == -100:
            print(pred_label)
        pred_label_list.append(label)
    return pred_label_list

def main():
    model_name = CONFIG['MODEL_NAME']
    train_df, test_df = read_data()
    generated_text_list = []
    definition_statement_list = []
    definition_statement = CONFIG['SEED_DEFINITION_STATEMENT']
    valid_df = train_df[train_df['cv_flag'] == 0].reset_index(drop=True)
    train_df = train_df[train_df['cv_flag'] != 0].reset_index(drop=True)
    for k in range(1, CONFIG['N_SPLIT']):
        sample_df = train_df[train_df['cv_flag'] == k].copy()
        if definition_statement is None:
            prompt = make_first_prompt(sample_df, CONFIG['SEED_GEN_TRUE_DATA_NUM'], CONFIG['SEED_GEN_FALSE_DATA_NUM'])
            generated_output = generate_text(prompt)
            prompt = PROMPT_DICT['extract_definition_prompt'] + generated_output
            definition_statement = generate_text(prompt, model_name)
            generated_text_list.append(generated_output)
            definition_statement_list.append(definition_statement)
        else:
            generated_output, definition_statement, stop_flag = turning_step_func(sample_df, definition_statement, model_name, CONFIG['MAX_SAMPLING_FALSE_NEGA_DATA_NUM'], CONFIG['MAX_SAMPLING_FALSE_POSI_DATA_NUM'])
            generated_text_list.append(generated_output)
            definition_statement_list.append(definition_statement)
            if stop_flag:
                break


    pd.DataFrame(definition_statement_list, columns=['definition_statement']).to_excel(f"{CONFIG['OUTPUT_PATH']}/{CONFIG['EXPERIMENT_NAME']}_definition_statements.xlsx", index=False)

    valid_df['pred_label'] = label_predict(valid_df, definition_statement_list[-1])
    valid_df['pred_label'] = valid_df['pred_label'].astype(np.int32)
    test_df['pred_label'] = label_predict(test_df, definition_statement_list[-1])
    test_df['pred_label'] = test_df['pred_label'].astype(np.int32)

    print('Valid Score', fn_compute_metrics(valid_df['label'].values, valid_df['pred_label'].values))
    print('Test Score', fn_compute_metrics(test_df['label'].values, test_df['pred_label'].values))

    valid_df.to_excel(f"{CONFIG['OUTPUT_PATH']}/{CONFIG['EXPERIMENT_NAME']}_valid.xlsx", index=False)
    test_df.to_excel(f"{CONFIG['OUTPUT_PATH']}/{CONFIG['EXPERIMENT_NAME']}_test.xlsx", index=False)

if __name__ =='__main__':
    main()
