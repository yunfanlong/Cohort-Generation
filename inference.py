# Model Inference

# Import Packages (fastai for optimizing inference)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


import os
from fastai.text import *
import pandas as pd
from fastai.tabular import *
import sys
from keras_tqdm import TQDMNotebookCallback
from tqdm.auto import tqdm
tqdm.pandas()


from utils.quick_load import *
from utils.model import *
from utils.helper import *


def load_data_bunch():
    return load_data("../saved/", 'Reports_ALL.pkl')


def get_inputs():
    inputs = {"file": "", "dest": "", 'model': ''}
    in_arr = sys.argv

    for arg in inputs:
        if '-{}'.format(arg) not in in_arr:
            inputs[arg] = str(input("-{}: ".format(arg)))
        else:
            inputs[arg] = in_arr[in_arr.index('-{}'.format(arg)) + 1]

    return inputs

# `interpret_IMPRESSION`:
# Function that formats report text to return a formatting that emphasizes important words, as destermined by saliency in red


def interpret_IMPRESSION(text,thres,interp,red,black):
    txt,attention = interp.intrinsic_attention(text)
    important_word=[]
    txt=str(txt).split(" ")
    index=0
    for x in attention: 
        if 'xx' in txt[index]:
            y=None
        elif x.item()>thres:
            important_word.append(red)
            important_word.append(txt[index])
            important_word.append(black)
            important_word.append(" ")
        else:
            if txt[index] is " ":
                attention.remove(txt[index])
            else:
                important_word.append(txt[index])
                important_word.append(" ")
        index+=1
    return important_word


# `load_CTHem_models`:
# - takes in the data object from the supervised training and loads the named model
# - specific `load_ccds_model` arguments:
#     - text_data = data object from unsupervised training (defined above)
#     - labeled_data = data object from supervised training (input for function)
#     - path_to_lm = path to language model (in `../saved/models`)
#     - path_to_classifier = name of classifier model (in `../saved/models`)
#     - specific arguments if modified model:
#         - *format of this list is `{argument}:{default_value} = {definition}*
#         - encoding_size:400 = size of encoding layer
#         - layer_size:1152 = size of encoder layers
#         - num_layers:3 = number of layers in encoder model
#         - pad_token:1 = padding token
#         - hidden_dropout:0.2 = dropout in hidden layer
#         - input_dropout:0.6 = dropout in input layer
#         - embed_dropout:0.1 = dropout in embedding layer
#         - weight_dropout:0.5 = dropout in weight layer
#         - output_dropout:0.1 = dropout in output layer
#         - qrnn_cells:bool=False = whether model has qrnn cells
#         - bidirectional:False = whether model is bidirectional
#         - num_classes:2 = number of classes for model (1 for regression)
#         - decoder_layer_sizes:[50] = list of layer sizes for decoder
#         - decoder_dropout:[0.1] = list of dropout values for decoder layers (should be single value or equal to length of layers
#         - bptt_classifier:70 = backprop through time, how many samples before gradient update
#         - max_length:2000 = max length of report document to feed in before splice
#         
#     


def load_CTHem_models(labeled_data,data_path,name):
    try:
        learn = load_ccds_model(text_data=text_data,
                                labeled_data=labeled_data,
                                path_to_lm='mod_all_reports_enc',
                                path_to_classifier=name,
                                decoder_layer_sizes=[50],
                                decoder_dropout=[0.1]
                                )
    except:
        learn = load_ccds_model(text_data=text_data,
                                labeled_data=labeled_data,
                                path_to_lm='mod_all_reports_enc',
                                path_to_classifier=name,
                                decoder_layer_sizes=[50],
                                decoder_dropout=[0.1],
                                num_classes=1
                                )
    return learn


# Functions to generate relevant keywords based on saliency of the model

def _eval_dropouts(mod):
        module_name =  mod.__class__.__name__
        if 'Dropout' in module_name or 'BatchNorm' in module_name: mod.training = False
        for module in mod.children(): _eval_dropouts(module)
            
def intrinsic_attention(self, text, class_id=None):
    self.model.train()
    _eval_dropouts(self.model)
    self.model.zero_grad()
    self.model.reset()
    ids = self.data.one_item(text)[0]
    emb = self.model[0].module.encoder(ids).detach().requires_grad_(True)
    lstm_output = self.model[0].module(emb, from_embeddings=True)
    self.model.eval()
    cl = self.model[1](lstm_output + (torch.zeros_like(ids).byte(),))[0].softmax(dim=-1)
    if class_id is None: 
        class_id = cl.argmax()
    cl[0][class_id].backward()
    attn = emb.grad.squeeze().abs().sum(dim=-1)
    attn /= attn.max()
    tokens = self.data.single_ds.reconstruct(ids[0])
    return tokens, attn

def keywords_generator(text,thres,interp):
    txt,attention = interp.intrinsic_attention(text)
    important_words=[]
    txt=str(txt).split(" ")
    index=0
    for x in attention: 
        if x.item()>thres:
            important_words.append(txt[index])
        index+=1
    return important_words


# Function to create cohort folder:
# - intervals = list of numeric intervals to generate specific files based on model confidence
# - file_names = list of file names that correspond to the interval segments
# - whole_file = whether to save whole file


def createCohortFiles(data,learn,output_file,ranking_col='risk_score',intervals=[-np.inf,0.3,0.6,np.inf],file_names=['not_likely','possibly_likely','very_likely'],whole_file=True):
    os.system('mkdir ../data/{}_cohort_tool'.format(output_file))
    if whole_file:
        data.to_csv('../data/{}_cohort_tool/full.csv'.format(output_file))
    for cohort in range(len(intervals)-1):
        test = data[(intervals[cohort]<= data[ranking_col]) & (data[ranking_col]<=intervals[cohort+1])]
        test.reset_index(drop=True)
        test.to_csv('../data/{}_cohort_tool/{}.csv'.format(output_file,file_names[cohort]),index=False)
        test = pd.read_csv('../data/{}_cohort_tool/{}.csv'.format(output_file,file_names[cohort]))
        test.reset_index(drop=True)
        test.to_csv('../data/{}_cohort_tool/{}.csv'.format(output_file,file_names[cohort]))
    return data


# `report_generation` - function to generate reports:
# - file = input file, automatically set with arguments and functions above
# - data_path = path of folder of models
# - bs = batch size
# - ranking = whether to rank the reports based on confidence or not
# - cohort = whether to generate folder with all cohort samples
# - risk_score = whether to generate column of main model analysis
# - predict = whether to generate column of class predictions for main model (ints)
# - binary_model_name = name of main model, set in arguments up above
# - prediction_thres = prediction threshold defined between positive & negative to store when `predict = True`
# - interpret = whether to generate excel file with colored text on important words
# - other_y_columns = list of other models to generate necessary rows and prediction, should have same supervised data object stored with `{model_name}_data_clas_export.pkl` in the `../saved` folder
# - key_words = whether to generate key_words for reports
# - interpret_thres = what the saliency threshold is to add a word to key_words or interpret
# - output_file - output file name, set in previous arguments

def report_generation(file=inputs['file'],
                      data_path='../saved',
                      bs=128,
                      ranking=False, 
                      cohort=True,
                      risk_score=True, 
                      predict=True,
                      binary_model_name=inputs['model'],
                      prediction_thres=0.5,
                      interpret=True, 
                      other_y_columns=[],
                      key_words=False,
                      interpret_thres=0.6,
                      output_file=inputs['dest']):
    to_remove = ["\d+", "\r", "\n", "IMPRESSION:"]
    data = pd.read_csv(file)
    data = clean_impressions(data,'IMPRESSION',to_remove)
    writer = pd.ExcelWriter('../data/{}_cohort_tool/output_file{}'.format(output_file,'.xlsx'), engine='xlsxwriter')
    workbook  = writer.book
    print('Loading Data...')
    binary_data_clas = load_data(data_path, '{}_data_clas_export.pkl'.format(binary_model_name), bs=bs)
    print('Done')
    print('Loading Models...')
    binary_model = load_CTHem_models(binary_data_clas,data_path,'CTHem')
    print('Done')
    if ranking or risk_score:
        print('Evaluating File...')
        data['risk_score'] = data['IMPRESSION'].progress_apply(lambda x: float(binary_model.predict(x)[2][1]))
        print('Done')
        if ranking:
            print('Ranking File...')
            data = data.sort_values(by=['risk_score'],ascending=False)
            print('Done')
    if len(other_y_columns) != 0:
        print('Predicting File...')
        for output in other_y_columns:
            print("Predicting: {}".format(output.upper()))
            multi_data_clas = load_data(data_path, '{}_data_clas_export.pkl'.format(output), bs=64)
            multi_model = load_CTHem_models(multi_data_clas,data_path,output)
            data["{}_preds".format(output)] = data['IMPRESSION'].progress_apply(lambda x: round(float(multi_model.predict(x)[1]),2))
            print('Done')
    if predict:
        print('Predicting File...')
        data['prediction'] = data['risk_score'].progress_apply(lambda x: int(x>prediction_thres))
    if not risk_score:
            data.drop('risk_score')  
    if cohort:
        createCohortFiles(data,binary_model,output_file)
    if key_words or interpret:
        interp = TextClassificationInterpretation.from_learner(binary_model) 
        if key_words:
            print('Generating Keywords...')
            data['Keywords'] = data['IMPRESSION'].progress_apply(lambda x: keywords_generator(x,interpret_thres,interp))
            print('Done')
        if interpret:
            print('Interpreting File...')
            data['Interpret'] = ''
            red = workbook.add_format({'color': 'red'})
            black = workbook.add_format({'color': 'black'})
            data['Interpret'] = data['IMPRESSION'].progress_apply(lambda x: interpret_IMPRESSION(x,interpret_thres,interp,red,black))
            data.to_excel(writer, sheet_name='Sheet1', index=False)
            worksheet = writer.sheets['Sheet1']
            for idx, x in data['Interpret'].iteritems():
                worksheet.write_rich_string(idx+1, len(data.columns)-1, *x)
            writer.save()
    data.to_excel(writer, sheet_name='Sheet1', index=False) 
    worksheet = writer.sheets['Sheet1']
    writer.save()
    print('File Done')
    return data

if __name__ == "__main__":
    # List of items to remove from text
    inputs = get_inputs()
    text_data = load_data_bunch()
    data = report_generation(interpret=False, ranking=False, binary_model_name=inputs['model'], other_y_columns=['EDH', 'IPH', 'IVH', 'SAH', 'SDH'])