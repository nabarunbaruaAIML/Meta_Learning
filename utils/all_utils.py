import pandas as pd
import numpy as np
import yaml
import os
import json
import logging
import shutil
import time
import boto3
from datasets import load_dataset
from datasets import Dataset
from datasets import ClassLabel
# from collections import OrderedDict
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
# import onnx
# import onnxruntime
# from onnxruntime.quantization import quantize_dynamic, QuantType
# from onnxruntime.transformers.onnx_model import OnnxModel
# from onnxruntime import InferenceSession, SessionOptions
import torch
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import wandb
from os import listdir
from os.path import isfile



"""
Method reads content of a JSON file.
"""
def read_json(path_to_json: str) -> dict:
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        logging.info(f"JSON file: {path_to_json} loaded successfully")
    return data


"""
Method reads content of a YAML file.
"""
def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content


"""
Method takes a list of directories and creates all of them and throws no error if they already exist.
"""
def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"directory is created at {dir_path}")


"""
Method used to save data to csv format.
"""
def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    logging.info(f"data is saved at {data_path}")


"""
Method performs a straight json dump locally as reports.
"""
def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    logging.info(f"reports are saved at {report_path}")


"""
Method helps copy files from source to destination using a high level file operations library Shutil.
This lists all files irrespective of the file extensions
"""
def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    
    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)
        logging.info(f"Copying File from {src} to {dest} Completed! Succefully")
    # for file in tqdm(list_of_files, total=N, desc=f"copying file from {source_download_dir} to {local_data_dir}", colour="green"):
    #     src = os.path.join(source_download_dir, file)
    #     dest = os.path.join(local_data_dir, file)
    #     shutil.copy(src, dest)


"""
Method helps copy files from source to destination using a high level file operations library Shutil.
This has a variation which is a filter for ".csv" file.
"""
def copy_file_csv(source_download_dir, local_data_dir):
    list_of_files =[ listfile for listfile in os.listdir(source_download_dir) if listfile.endswith('.csv') ] 
    N = len(list_of_files)
    
    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)
        logging.info(f"Copying File from {src} to {dest} Completed! Succefully")
    # for file in tqdm(list_of_files, total=N, desc=f"copying file from {source_download_dir} to {local_data_dir}", colour="green"):
    #     src = os.path.join(source_download_dir, file)
    #     dest = os.path.join(local_data_dir, file)
    #     shutil.copy(src, dest)


"""
Method helps copy files from S3 bucket to local defined directory.
We obtain an s3 object using which the bucket directory is listed and the needed files are downloaded and written
locally
"""
def copy_file_from_S3(s3,S3_bucket, local_data_dir,List_S3_Folder):
    # print('S3 load start')    
    obj = s3.Bucket(S3_bucket)
    for folder in  List_S3_Folder:
        folder_list = []
        list_of_files = [] #[ i.key for i in obj.objects.filter(Prefix=folder ) if i.key.endswith(('.csv','.json'))  ] 
        for i in obj.objects.filter(Prefix=folder ):
            if i.key.endswith(('.csv','.json','.txt','.parquet')):
                list_of_files.append(i.key)
            else:
                folder_list.append(i.key) 
                
        # folder_list =  [i.key for i in obj.objects.filter(Prefix=folder ) if  i.key.endswith(('.csv','.json'))]  # '.csv' not in i.key]
        folder_list = [ os.path.join(local_data_dir,i ) for i in folder_list ]
        create_directory(folder_list)
        N = len(list_of_files)
         
        for file in list_of_files:
            
            # src = os.path.join(source_download_dir, file)
            dest = os.path.join(local_data_dir, file)
            
            # print(f'Destination{dest}')
            
            # shutil.copy(src, dest)
            s3.Bucket(S3_bucket).download_file(Key=file, Filename=dest)
            logging.info(f"Copying File from S3 to {dest} Completed! Succefully")
            
            # print(f"Copying File from S3 to {dest} Completed! Succefully")


# def copy_file_from_S3(s3,S3_location, local_data_dir):
#     # print('S3 load start')    
#     obj = s3.Bucket(S3_location)
#     list_of_files =[ i.key for i in obj.objects.all() ] 
#     N = len(list_of_files)
    
#     for file in list_of_files:
        
#         # src = os.path.join(source_download_dir, file)
#         dest = os.path.join(local_data_dir, file)
        
#         # print(f'Destination{dest}')
        
#         # shutil.copy(src, dest)
#         s3.Bucket(S3_location).download_file(Key=file, Filename=dest)
#         logging.info(f"Copying File from S3 to {dest} Completed! Succefully")
        
#         # print(f"Copying File from S3 to {dest} Completed! Succefully")





"""
Method  generates ASC time stamps
"""
def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name


"""
Deprecated:
This is a previously used method in method "read_data" to read and tokenize data. 
"""
def get_input_ids(df,tokeniser,padding,max_length,truncation,stop_words=None):
    df = df.tolist()
    output = tokeniser(text=df , truncation=truncation,padding =padding,max_length=max_length)
    return output

    #DeprecatedCode:-------------------------------------------------------------------------------
    # inp_ids = []
    # attension_mask= []
    # for i in df:
    #     output = tokeniser(text=i, truncation=truncation,padding =padding,max_length=max_length)#,return_tensors='np' )
    #     inp_ids.append(output['input_ids'])
    #     attension_mask.append(output['attention_mask'])
    # return inp_ids,attension_mask


"""
Deprecated:
This is a previously used method to read and tokenize data. However, this is a very hefty function on the memory.
"""
def read_data(string,tokenizer,padding,max_length,truncation,stop_words=None):
    # json = {}
    df = pd.read_csv(string)
    DataFrame_Filtered = df.iloc[:,0:2]
    Label_set = set(DataFrame_Filtered.iloc[:,0])
    Label_set = sorted(Label_set)
    label_num = len( Label_set )
    id2label = {k:v for k,v in enumerate(Label_set,0)}
    label2id = {v:k for k,v in enumerate(Label_set,0)}
    # inp_ids,attension_mask = get_input_ids(DataFrame_Filtered.iloc[:,1],tokenizer,padding,max_length,truncation,stop_words)
    json = get_input_ids(DataFrame_Filtered.iloc[:,1],tokenizer,padding,max_length,truncation,stop_words)
    DataFrame_Filtered1 = DataFrame_Filtered.iloc[:,0].map(lambda x: label2id[x]).copy()
    Label = DataFrame_Filtered1.tolist()
    # json['input_ids'] =inp_ids
    # json['attention_mask']= attension_mask
    json['labels']= Label 
    logging.info(f"Read Completed and Tokenise Succefully")
    return json,id2label,label2id,label_num,Label_set


"""
In Use:
Method reads the dataset, tokenizes it , maps it with labels and returns
 dataset,id2label,label2id,label_num,Label_set
"""
def read_dataset(list_of_files_with_path,cache_path,tokenizer,padding,max_length,truncation,combine ):    
#    ('.csv','.json','.txt','.parquet')
    if(list_of_files_with_path[0].endswith('.csv')):
        ds = load_dataset("csv", data_files= {'train':list_of_files_with_path} , cache_dir=cache_path )
    elif(list_of_files_with_path[0].endswith('.json')):
        ds = load_dataset("json", data_files= {'train':list_of_files_with_path} , cache_dir=cache_path ,field="data")
    # elif(list_of_files_with_path[0].endswith('.txt')):
    #     ds = load_dataset("txt", data_files= {'train':list_of_files_with_path} , cache_dir=cache_path )
    # elif(list_of_files_with_path[0].endswith('.parquet')):
    #     ds = load_dataset("parquet", data_files= {'train':list_of_files_with_path} , cache_dir=cache_path )
    
    def lower_label(example):
        return {'label':[i.lower() for i in example['label'] ] } 
    ds['train'] = ds['train'].map(lower_label, batched=True)   
    if combine:
        label_num = read_json('artifacts\Data\label_num.json')['Number_of_Label'] 
        label2id =  read_json('artifacts\Data\label2id.json') 
        id2label =  read_json('artifacts\Data\id2label.json')
        label_list = [v for k,v in enumerate(label2id)]
        classlabel = ClassLabel(names=label_list)
    else:
        label_list = ds['train'].unique('label')
        label_num = len(label_list)
        label2id = {v:k for k,v in enumerate(label_list)}
        id2label = {k:v for k,v in enumerate(label_list)}
        classlabel = ClassLabel(names=label_list)
    """
    This is an inner function which has access to all local variables initiated by the outer enclosing function.
    """
    def labelMap(example):
        return {'label':[label2id[i] for i in example['label'] ] }
    ds = ds['train'].map(labelMap, batched=True)
    ds = ds.cast_column('label',classlabel)
    ds = ds.align_labels_with_mapping(label2id=label2id,label_column='label')
    
    """
    This is an inner function which has access to all local variables initiated by the outer enclosing function.
    This can be the tokenizer from the transformers or a defined function.Its the tokenizer from transformers here.
    """
    def tokenise(example):
        if combine:
            # input = tokenizer(example['Combine'] ,truncation=True,max_length=max_length)
            pass
        else:
            input = tokenizer(example['evidence'],example['claim'],truncation=truncation,max_length=max_length)
        # input['labels'] = [i for i in example['label']]
        input['labels'] =  example['label']
        return input
    if combine:
        # ds = ds.map(tokenise,batched=True)
        def Merge_Text(examples):
            Combine = {}
            claim = examples['claim']
            evidence = examples['evidence']
            Combine['Combine'] = [ e + ' \n ' + c for e,c in zip(evidence,claim)]
            return Combine   #{'Combine':[ i['evidence'] + ' \n ' + i['claim'] for i in examples]} 
            # return{'Combine':[ i['evidence'] + ' \n ' + i['claim'] for i in examples]} 
            # return{ 'Combine': examples['evidence'] + ' \n ' + examples['claim']}
        ds = ds.map(Merge_Text,batched=True)
        # print(Merge_Text(ds[0:2]))
    else:
        ds = ds.map(tokenise,batched=True,remove_columns=ds.features)
    ds = ds.train_test_split(test_size=0.1) 
    ds['validation'] = ds.pop('test')
    tmp_ds = ds['train'].train_test_split(test_size=0.1)
    ds['train']= tmp_ds['train']
    ds['test'] = tmp_ds['test']
         
    # dataset.set_format('torch')
    logging.info(f"Read Completed and Tokenise Succefully")
    
    return ds,id2label,label2id,label_num,label_list
# def read_dataset(list_of_files_with_path,cache_path,tokenizer,padding,max_length,truncation ):    
   
#     ds = load_dataset("csv", data_files= {'train':list_of_files_with_path} , cache_dir=cache_path )
    
#     label_list = ds['train'].unique('label')
#     label_num = len(label_list)
#     label2id = {v:k for k,v in enumerate(label_list)}
#     id2label = {k:v for k,v in enumerate(label_list)}
#     classlabel = ClassLabel(names=label_list)
#     """
#     This is an inner function which has access to all local variables initiated by the outer enclosing function.
#     """
#     def labelMap(example):
#         return {'label':[label2id[i] for i in example['label'] ] }
#     ds = ds['train'].map(labelMap, batched=True)
#     ds = ds.cast_column('label',classlabel)
#     ds = ds.align_labels_with_mapping(label2id=label2id,label_column='label')
#     """
#     This is an inner function which has access to all local variables initiated by the outer enclosing function.
#     This can be the tokenizer from the transformers or a defined function.Its the tokenizer from transformers here.
#     """
#     def tokenise(example):
#         input = tokenizer(example['evidence'],example['claim'],truncation=truncation,max_length=max_length)
#         # input['labels'] = [i for i in example['label']]
#         input['labels'] =  example['label']
#         return input
#     ds = ds.map(tokenise,batched=True,remove_columns=ds.features)
#     ds = ds.train_test_split(test_size=0.1) 
#     ds['validation'] = ds.pop('test')
#     tmp_ds = ds['train'].train_test_split(test_size=0.1)
#     ds['train']= tmp_ds['train']
#     ds['test'] = tmp_ds['test']
         
#     # dataset.set_format('torch')
#     logging.info(f"Read Completed and Tokenise Succefully")
    
#     return ds,id2label,label2id,label_num,label_list


"""
Method saves the designated Json into a file locally.
"""
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")

"""
Method gets the input,output parameters and right Model.
"""    
# def parameters( task):
#         """
#         Defines inputs and outputs for an ONNX model.
#         Args:
#             task: task name used to lookup model configuration
#         Returns:
#             (inputs, outputs, model function)
#         """

#         inputs = OrderedDict(
#             [
#                 ("input_ids", {0: "batch", 1: "sequence"}),
#                 ("attention_mask", {0: "batch", 1: "sequence"}),
#                 ("token_type_ids", {0: "batch", 1: "sequence"}),
#             ]
#         )

#         config = {
#             "default": (OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}), AutoModel.from_pretrained),
#             # "pooling": (OrderedDict({"embeddings": {0: "batch", 1: "sequence"}}), lambda x: MeanPoolingOnnx(x, -1)),
#             "question-answering": (
#                 OrderedDict(
#                     {
#                         "start_logits": {0: "batch", 1: "sequence"},
#                         "end_logits": {0: "batch", 1: "sequence"},
#                     }
#                 ),
#                 AutoModelForQuestionAnswering.from_pretrained,
#             ),
#             "sequence-classification": (OrderedDict({"logits": {0: "batch"}}), AutoModelForSequenceClassification.from_pretrained),
#         }

#         # Aliases
#         config["zero-shot-classification"] = config["sequence-classification"]

#         return (inputs,) + config[task]
# """Model Compress the Onnx Model"""
# def CompressModel(uncompress,compress):
#     model=onnx.load(uncompress)
#     onnx_model=OnnxModel(model)
#     output_path = compress#f"model_compressed.onnx"
#     count = len(model.graph.initializer)
#     same = [-1] * count
#     for i in range(count - 1):
#         if same[i] >= 0:
#             continue
#         for j in range(i+1, count):
#             if has_same_value(model.graph.initializer[i], model.graph.initializer[j]):
#                 same[j] = i

#     for i in range(count):
#         if same[i] >= 0:
#             onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name, model.graph.initializer[same[i]].name)
#     onnx_model.update_graph()
#     onnx_model.save_model_to_file(output_path)
#     logging.info(f"Compressed the Onnx Model and Saved at {output_path}")

# """This Function Checks if Two Nodes in Model has same weights"""    
# def has_same_value(val_one,val_two):
#       if val_one.raw_data == val_two.raw_data:
#           return True
#       else:
#           return False
# """This Fucntion is Dynamically Quantizing the Model"""      
# def quantize_onnx_model(onnx_model_path, quantized_model_path):  
#     # onnx_opt_model = onnx.load(onnx_model_path)
#     quantize_dynamic(onnx_model_path,
#                      quantized_model_path,
#                      weight_type= QuantType.QInt8
#                     )
#     norm = 'ONNX full precision model size (MB):', os.path.getsize(onnx_model_path)/(1024*1024)
#     quant = 'ONNX quantized model size (MB):', os.path.getsize(quantized_model_path)/(1024*1024)
#     logging.info(norm)
#     logging.info(quant)

# """Reading Onnx Model and getting output"""    
# def Onnx_Sesion(Onnx_Model_Path,dataset,max_length):
#     options = SessionOptions()
#     options.optimized_model_filepath = Onnx_Model_Path
#     session = InferenceSession(Onnx_Model_Path, options, providers=['CPUExecutionProvider'])
#     label_list = []
#     output_list =[]
#     for i,single in enumerate(dataset):
        
#         ort = {
#                 'attention_mask': np.array(single['attention_mask'], dtype=np.int64).reshape(1,max_length), 
#                 'input_ids': np.array(single['input_ids'], dtype=np.int64).reshape(1,max_length), 
#                 'token_type_ids': np.array(single['token_type_ids'], dtype=np.int64).reshape(1,max_length)
#               }
        
#         outputs = session.run(None, ort )
#         outputs_argmax = np.argmax(outputs[0] ,axis=1)
#         output_list.extend(outputs_argmax)
#         label_list.append(single['labels'])
#     return label_list,output_list


def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def Upload_to_S3(s3,bucket_name, local_data_dir,unwanted_file):
    for dir in tqdm(local_data_dir, desc= 'Folders'):
        dir_file =  os.listdir(dir)
        for file in tqdm(dir_file, desc= 'Files from Directory'):
            if (file not in unwanted_file):  # ['.gitignore','model.onnx']
                # print(file)
                local_path = os.path.join(dir,file)
                s3.Object(bucket_name, file).upload_file(local_path)
                logging.info(f"Model {file} Uploaded to S3 Busket Successfully")

"""Weights & Baises Model Uploaded"""      
def Weights_Baises(run,folder,unwanted_file):
    for dir in tqdm(folder, desc= 'Folders'):
        dir_file =  os.listdir(dir)
        for file in tqdm(dir_file, desc= 'Files from Directory'):
            if (file not in unwanted_file):  # ['.gitignore','model.onnx'] etc
                local_path = os.path.join(dir,file)
                replaced_text = file.replace('.onnx', '')
                # print(replaced_text)
                raw_model = wandb.Artifact(replaced_text, type='model',description='Onnx Model')
                raw_model.add_file(local_path,file)
                run.log_artifact(raw_model)
                logging.info(f"Model {replaced_text} Uploaded to Weights and Baises Successfully")
                