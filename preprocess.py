from datasets.preprocess import get_cm_protocols, get_dataset_annotation, random_split_train_dev, create_non_label_eval_json
import pathlib
import json

if __name__ == '__main__':
    # TODO: MAKE THIS PARSE ARGUMENTS
    print('----Start to Process Data -----')

    args = {}
    args['data_type'] = ['labeled','unlabeled'][1]

    if args['data_type'] == 'labeled':
        print('Start to process labeled data:')
        pathlib.Path('processed_data').mkdir(parents=True, exist_ok=True)
        LA_PRO_DIR = '../data/PA/ASVspoof2019_PA_cm_protocols'
        PRO_FILES = ('ASVspoof2019.PA.cm.train.trn.txt',
                     'ASVspoof2019.PA.cm.dev.trl.txt',
                     'ASVspoof2019.PA.cm.eval.trl.txt')
        SAVE_DIR = '2021_data/'
        DATA_DIR = '../data/'
        split_features= get_cm_protocols(pro_dir=LA_PRO_DIR,
                                         pro_files=PRO_FILES
                                         )
        get_dataset_annotation(split_features,
                               data_dir=DATA_DIR,
                               save_dir=SAVE_DIR,
                               )
        random_split_train_dev()
    elif args['data_type'] == 'unlabeled':
        print('Start to process unlabeled data:')

        create_non_label_eval_json(pro_file = '../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt',
                                   data_dir = '../data/LA/ASVspoof2021_DF_eval/flac/',
                                   output_file = './processed_data/cm_eval_2021.json')



    # # Clear eval file
    # with open('processed_data/cm_eval_2021.json','r') as f:
    #     data = json.load(f)
    # for k in data:
    #     data[k]['file_path'] = data[k]['file_path'].replace('ASVspoof2019_LA_eval', 'ASVspoof2021_LA_eval')
    # with open('processed_data/cm_eval_2021_cleaned.json','w') as f:
    #     data = json.dump(data, f)
