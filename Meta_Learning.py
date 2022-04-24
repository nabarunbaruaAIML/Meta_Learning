import logging
import argparse
import os
from datasets import load_dataset, ClassLabel,load_metric
from transformers import   AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding
from accelerate import Accelerator
 
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import gc
from copy import deepcopy
import torch
import numpy as np
from utils.all_utils import read_yaml
from tqdm.auto import tqdm

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
 
os.makedirs(log_dir, exist_ok=True)
"""
The logging configurations are set here like logging level ,file name etc.
"""
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
MAX_GPU_BATCH_SIZE = 16
def Learner(tokenizer,model,Outer_optimizer,Dataset,inner_batch_size,inner_batch_size_eval,Inner_Learning_rate,train_task,test_task,Epoch,training):
    """Initializing Variables"""
    num_task = len(train_task)
    epoch = Epoch
    tasks = zip(train_task,test_task)
    Metric = []
    sum_gradients = []
    data_collator = DataCollatorWithPadding(tokenizer,padding = 'max_length', max_length= 512 )
    f1_metric = load_metric('f1',cache_dir='Artifact\Cache')
    gradient_accumulation_steps = 1
    accelerator = Accelerator()
    if inner_batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = inner_batch_size // MAX_GPU_BATCH_SIZE
        inner_batch_size = MAX_GPU_BATCH_SIZE
    """Looping over each Task  Train & Test Task"""    
    for task_id, task in enumerate(tasks):
        logging.info(f'\nTask ID: {task_id}')#
        print(f'\nTask ID:',task_id)
        support = Dataset[task[0]]
        query = Dataset[task[1]]
        fast_model = deepcopy(model)
        # fast_model.to('cuda')# --
        support_dataloader = DataLoader(support,collate_fn=data_collator,batch_size=inner_batch_size)
        query_dataloader = DataLoader(query,collate_fn=data_collator,batch_size=inner_batch_size_eval)
        Inner_optimizer = AdamW(fast_model.parameters(), lr=Inner_Learning_rate)
        
        """Accelerate"""
        fast_model, Inner_optimizer, support_dataloader, query_dataloader  = accelerator.prepare(
                                                                                                    fast_model, 
                                                                                                    Inner_optimizer, 
                                                                                                    support_dataloader, 
                                                                                                    query_dataloader 
                                                                                                )
        num_update_steps_per_epoch = len(support_dataloader)
        num_training_steps = (epoch * num_update_steps_per_epoch) // gradient_accumulation_steps
        lr_scheduler = get_scheduler(
                                  "linear",
                                  optimizer=Inner_optimizer,
                                  num_warmup_steps=100,
                                  num_training_steps=num_training_steps,
                                )
        
        fast_model.train()
        for i in range(epoch):
            all_loss = []
            for inner_step, batch in enumerate(support_dataloader):
                # batch.to('cuda')
                output =  fast_model(**batch)
                loss = output.loss
                loss = loss / gradient_accumulation_steps
                # loss.backward()#comment
                accelerator.backward(loss)
                if inner_step % gradient_accumulation_steps == 0:
                    Inner_optimizer.step()
                    lr_scheduler.step()
                    Inner_optimizer.zero_grad()
                all_loss.append(loss.item())
             
            logging.info(f'\nInner Epoch: {i} \tInner Loss: {np.mean(all_loss)}')
            print(f'\nInner Epoch: {i} \tInner Loss: {np.mean(all_loss)}')
            fast_model.eval()
            # query_dataloader = DataLoader(query,collate_fn=data_collator,batch_size=inner_batch_size_eval)
            for batch in query_dataloader:
                # batch.to('cuda')
                with torch.no_grad():
                    q_outputs =  fast_model(**batch)
                predictions = q_outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                
                """Necessary to pad predictions and labels for being gathered"""
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)
                
                f1_metric.add_batch(predictions=predictions_gathered, references=labels_gathered)
                 
        average={'average':'micro'}
        score_f1 = f1_metric.compute(**average)
        Metric.append(score_f1['f1'])
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(fast_model)
        unwrapped_model.to('cpu')
        
        """Calculating gradients for Meta Learning"""
        if training:
            meta_weights = list(model.parameters())
            fast_weights = list(unwrapped_model.parameters())
            # gradients = []
            for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                gradient = meta_params - fast_params
                if task_id == 0:
                    sum_gradients.append(gradient)
                else:
                    sum_gradients[i] += gradient
        # Accelerate
        accelerator.free_memory()
        # fast_model.to('cpu')
        del unwrapped_model #fast_model, Inner_optimizer
        torch.cuda.empty_cache()
        
    """Updating Gradients of Main Model"""
    if training:
        for i in range(0,len(sum_gradients)):
            sum_gradients[i] = sum_gradients[i] / float(num_task)
        for i, params in enumerate(model.parameters()):
            params.grad = sum_gradients[i]
        Outer_optimizer.step()
        Outer_optimizer.zero_grad()
        del sum_gradients
        gc.collect()
       
    return model,Outer_optimizer,np.mean(Metric)


def meta_task(num_task,Dataset,tokenizer,support,query,Train):
    """Selecting Domain"""
    if Train:
        domain =  Dataset['Whole'].unique('domain')
    else:
        domain =  Dataset['Whole_test'].unique('domain')

    """Based on Number of Task, Creat those many Dataset based on Random Domain Choice"""
    for i in range(num_task):
        choice = random.choice(domain)
        if Train:
            choice_domain = Dataset['Whole'].filter(lambda x: x['domain']==choice)
        else:
            choice_domain = Dataset['Whole_test'].filter(lambda x: x['domain']==choice)
        choice_domain =  choice_domain.shuffle(seed = random.randint(1,500))
        selected_examples_train = choice_domain.select(range(0,support)).shuffle(seed = random.randint(1,500))
        selected_examples_test = choice_domain.select(range(support,support+query)).shuffle(seed = random.randint(1,500))
        train = 'Train_'+ str(i)
        test = 'Test_'+ str(i)
        def tokenize_function(example):
            input = tokenizer(example["text"], truncation=True,max_length=512)
            input['labels'] =  example['label']
            return input
        selected_examples_train = selected_examples_train.map(tokenize_function,batched=True,remove_columns=selected_examples_train.features)
        selected_examples_test = selected_examples_test.map(tokenize_function,batched=True,remove_columns=selected_examples_test.features)
        Dataset[train]= selected_examples_train
        Dataset[test]= selected_examples_test
    return Dataset
def main(parsed_args):
    """Config and Params"""
    config = read_yaml(parsed_args.config)
    params = read_yaml(parsed_args.params)
    training_DS = config['training_DS']
    Best_Model = config['Best_Model']
    Model = config['Model']
    cache_dir = config['cache_dir']
    
    meta_epoch = params['meta_epoch']
    inner_batch_size = params['inner_batch_size']
    inner_batch_size_eval= params['inner_batch_size_eval']
    outer_update_lr= params['outer_update_lr']
    inner_update_lr= params['inner_update_lr']
    inner_Epoch = params['inner_Epoch']
    inner_Epoch_eval = params['inner_Epoch_eval']
    
    """Dataset Loading"""
    Dataset = load_dataset('json',data_files={"Whole":training_DS},field='data',cache_dir = cache_dir)
    label_list = Dataset['Whole'].unique('label')
    label_num = len(label_list)
    label2id = {v:k for k,v in enumerate(label_list)}
    id2label = {k:v for k,v in enumerate(label_list)}
    classlabel = ClassLabel(names=label_list)
    
    def labelMap(example):
        return {'label':[label2id[i] for i in example['label'] ] }
        
    Dataset['Whole'] = Dataset['Whole'].map(labelMap, batched=True)
    Dataset['Whole'] = Dataset['Whole'].cast_column('label',classlabel)
    Dataset['Whole'] = Dataset['Whole'].align_labels_with_mapping(label2id=label2id,label_column='label')
    Dataset['Whole_test'] = Dataset['Whole'].filter(lambda x: x['domain'] in ['automotive','computer_&_video_games','office_products'])
    Dataset['Whole'] = Dataset['Whole'].filter(lambda x: x['domain'] not in ['automotive','computer_&_video_games','office_products'])
    
    
    """Model Declaration"""
    model = AutoModelForSequenceClassification.from_pretrained(Model,
                                                            label2id=label2id, 
                                                            id2label=id2label,  
                                                            num_labels= label_num,
                                                            cache_dir= cache_dir )

    model.to('cpu')
    tokenizer = AutoTokenizer.from_pretrained( Model, use_fast=True,cache_dir= cache_dir )
    
    
    """Meta Training Starts"""
    logging.info(f'\nMeta Training Starts')
    global_step = tqdm(range(meta_epoch))
    Outer_optimizer = AdamW(model.parameters(), lr=outer_update_lr)
    for epoch in range(meta_epoch):
        logging.info(f'\nMeta Epoch: {epoch}')
        print(f'\nMeta Epoch: {epoch}')
        
        """Training"""
        Train_Dataset = meta_task(10,Dataset,tokenizer,100,30,True)
        column_name = Train_Dataset.column_names
        test= []
        train=[]
        for k,v in enumerate(column_name.keys()):
            if(v not in ['Whole','Whole_test']):
                if v.startswith('Train'):
                    train.append(v)
                else:
                    test.append(v)
        model,Outer_optimizer,metric = Learner(tokenizer,model,Outer_optimizer,Train_Dataset,inner_batch_size,inner_batch_size_eval,inner_update_lr,train,test,inner_Epoch,True)
        logging.info(f'\nMeta Epoch: {epoch} \tTraining F1: {metric}')
        print(f'\nMeta Epoch: {epoch} \tTraining F1: {metric}') 

        """Testing"""
        Test_Dataset = meta_task(10,Dataset,tokenizer,50,30,False)
        column_name = Test_Dataset.column_names
        test= []
        train=[]
        for k,v in enumerate(column_name.keys()):
            if(v not in ['Whole','Whole_test']):
                if v.startswith('Train'):
                    train.append(v)
                else:
                    test.append(v)
        model,Outer_optimizer,metric = Learner(tokenizer,model,Outer_optimizer,Test_Dataset,inner_batch_size,inner_batch_size_eval,inner_update_lr,train,test,inner_Epoch_eval,False)
        logging.info(f'\nMeta Epoch: {epoch} \tTest F1: {metric}')
        print(f'\nMeta Epoch: {epoch} \tTest F1: {metric}') 
        global_step.update(1)
         
    state_dict = model.state_dict()
    model.save_pretrained(Best_Model, state_dict=state_dict)
    tokenizer.save_pretrained(Best_Model)
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info('>>>>Start Meta-Learning<<<<')
        main(parsed_args)
        logging.info('>>>>End Meta-Learning<<<<') 
    except Exception as e:
        logging.exception(e)
        raise e