'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
import logging
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_paraphrase, model_eval_multitask, model_eval_test_multitask, model_eval_sts
from train_functions import training_loop as vanilla_training_loop, sst_batch_loss, para_batch_loss, sts_batch_loss, save_model
from train_functions_ray import training_loop as ray_training_loop
from train_functions import sst_batch_loss, sts_batch_loss, para_batch_loss
from utils import get_device, prepend_dir
import ray
import lightning.pytorch as pl
import ray.train.lightning as ray_pl
from lightning.pytorch.callbacks import EarlyStopping, Callback
from ray import train
from tempfile import TemporaryDirectory
import os
from ray.train import Checkpoint

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(pl.LightningModule):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config={}, args={}):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE // 2, N_SENTIMENT_CLASSES)
        )

        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )
        self.similarity_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )
        self.task_type = None
        self.args = args
        self.config = config


    def set_task_type(self, task_type):
        self.task_type = task_type

    def training_step(self, batch, _):
        if self.task_type == 'sst':
            loss = sst_batch_loss(self.args, self, batch)
        elif self.task_type == 'para':
            loss = para_batch_loss(self.args, self, batch)
        elif self.task_type == 'sts':
            loss = sts_batch_loss(self.args, self, batch)
        else:
            raise ValueError("Task type not set.")
        self.log('train_loss', round(loss.item(), 4))
        return loss

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
         
        x = self.bert(input_ids, attention_mask)['pooler_output']
        x = self.dropout(x)
        return x

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        hidden_states = self.forward(input_ids, attention_mask)
        return self.sentiment_classifier(hidden_states)

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        hidden_states_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_states_2 = self.forward(input_ids_2, attention_mask_2)
        hidden_states = torch.cat((hidden_states_1, hidden_states_2), dim=1)
        return self.paraphrase_classifier(hidden_states)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        hidden_states_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_states_2 = self.forward(input_ids_2, attention_mask_2)
        hidden_states = torch.cat((hidden_states_1, hidden_states_2), dim=1)
        return self.similarity_classifier(hidden_states)

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer

    def validation_step(self, batch, batch_idx, dataloader_idx):
        task = self.args.validation_tasks[dataloader_idx]
        if task == 'sst':
            acc, *_ = model_eval_sst(batch, self, single_batch=True)
            self.log('val_sentiment_accuracy', acc)
        elif task == 'para':
            acc, *_ = model_eval_paraphrase(batch, self, single_batch=True)
            self.log('val_paraphrase_accuracy', acc)
        elif task == 'sts':
            corr, *_ = model_eval_sts(batch, self, single_batch=True)
            self.log('val_sts_corr', corr)


def create_datasets(args):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args, isRegression=True)
    para_dev_data = SentencePairDataset(para_dev_data, args, isRegression=True)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    datasets = {
        'sst': (sst_train_dataloader, sst_dev_dataloader),
        'para': (para_train_dataloader, para_dev_dataloader),
        'sts': (sts_train_dataloader, sts_dev_dataloader),
        'num_labels': num_labels
    }

    return datasets

def get_config(args, datasets):
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': datasets['num_labels'],
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    return config



def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = get_device(args.use_gpu)
    datasets = args.datasets

    # Init model.

    config = get_config(args, datasets)
    

    task = args.task 

    lightening_params = {
        'precision': args.precision,
        'accelerator': 'auto' if args.use_gpu else None,
        'devices': 'auto'
    }

    if args.epochs_per_task:
        callbacks = []

    else:
        callbacks = [EarlyStopping(monitor='train_loss', patience=3)]

    if args.use_ray:
        callbacks.append(ray_pl.RayTrainReportCallback())
        lightening_params = {
            'precision': args.precision,
            'devices': 'auto',
            'accelerator': 'auto',
            'strategy': ray_pl.RayFSDPStrategy() if args.strategy == 'fsdp' else ray_pl.RayDDPStrategy(find_unused_parameters=True),
            'plugins': [ray_pl.RayLightningEnvironment()],
            'callbacks': callbacks
        }


    # create the model
    model = MultitaskBERT(config, args)
    model.set_task_type(task)

    # create the pl trainer

    trainer = pl.Trainer(max_epochs=args.max_epochs, **lightening_params)
    # decide if model is trained by ray or not
    if args.use_ray:
        ray.train.lightning.prepare_trainer(trainer)
    else:
        model = model.to(device)

    # get the data 
    train_dataloader, _ = datasets[task]

    # get the validation data
    validation_dataloaders = []
    for task in args.validation_tasks:
        validation_dataloaders.append(datasets[task][1])


    checkpoint = train.get_checkpoint()
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
            # validate existing metrics
            trainer.validate(model, dataloaders=validation_dataloaders, ckpt_path=ckpt_path)

            # run the training loop
            trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloaders,
                ckpt_path=ckpt_path)
    else:
         trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloaders)

    save_model(model, None, args, config, args.filepath)


def test_multitask(args, from_checkpoint=False):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = get_device(args.use_gpu)

        if from_checkpoint:
            config = get_config(args, args.datasets)
            model = MultitaskBERT.load_from_checkpoint(args.filepath, config=config, args=args)
        else:
            saved = torch.load(args.filepath, map_location=device)
            config = saved['model_config']

            model = MultitaskBERT(config, args)
            model.load_state_dict(saved['model'])
        
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, help="tasks to train for", nargs="+", default=["sst", "para", "sts"])
    parser.add_argument("--validation-tasks", type=str, help="tasks to run validations for", nargs="+", default=["sst", "para", "sts"])
    parser.add_argument("--use-ray", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--storage-path", help="Path where to store ray results and checkpoints", type=str, required='--use-ray' in sys.argv)
    parser.add_argument("--data-dir", help="Path to train and dev datasets, expects them to be under data folder", type=str, required='--use-ray' in sys.argv, default='')
    parser.add_argument("--output-dir", help="Path to store the best model at", type=str, default='', required='--filepath' not in sys.argv and '--use-ray' in sys.argv)
    parser.add_argument("--name", type=str, required='--use-ray' in sys.argv, help="Name of the experiment")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, required='--use-ray' in sys.argv)
    parser.add_argument("--cosine-similarity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tqdm-disable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--resume-from-checkpoint", type=str, help="Specify a checkpoint to resume from", default=None)
    parser.add_argument("--strategy", type=str, help="Type of strategy", default='ddp')
    parser.add_argument("--precision", type=str, help="Precision", default='16-mixed')
    parser.add_argument("--mode", type=str, default='train', help="train, test")
    parser.add_argument("--filepath", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s-%(levelname)s: %(message)s')
    args = get_args()

    # decide where to load the model from
    # if not specified use the default name
    # wih default output directory
    if not args.filepath:
        args.filepath = f'{args.fine_tune_mode}-{args.lr}-multitask.pt' # Save path.

        if args.output_dir:
            args.filepath = prepend_dir(args.output_dir, args.filepath)
    
    if args.data_dir:
        args.sst_train = prepend_dir(args.data_dir, args.sst_train)
        args.sst_dev = prepend_dir(args.data_dir, args.sst_dev)
        args.sst_test = prepend_dir(args.data_dir, args.sst_test)

        args.para_train = prepend_dir(args.data_dir, args.para_train)
        args.para_dev = prepend_dir(args.data_dir, args.para_dev)
        args.para_test = prepend_dir(args.data_dir, args.para_test)

        args.sts_train = prepend_dir(args.data_dir, args.sts_train)
        args.sts_dev = prepend_dir(args.data_dir, args.sts_dev)
        args.sts_test = prepend_dir(args.data_dir, args.sts_test)

    seed_everything(args.seed)  # Fix the seed for reproducibility.

    args.datasets = create_datasets(args)

    if args.mode == 'test':
        logging.info("-----------------Testing-----------------")
        is_checkpoint = False
        if args.resume_from_checkpoint is not None:
            args.filepath = args.resume_from_checkpoint
            is_checkpoint = True
        test_multitask(args, from_checkpoint=is_checkpoint)

    elif args.mode == 'train':
        checkpoint = None
        for task in args.tasks:
            args.task = task
            logging.info("-----------------Training-----------------")
            if not args.use_ray:
                train_multitask(args)
            else:
                logging.info(f"Using Ray for training with {args.num_workers} workers.")
                scaling_config = ray.train.ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu)
                # currently keeps the last checkpoint, can configure checkpoint_score_attribute with num to keep to save the best checkpoint
                checkpoint_config = ray.train.CheckpointConfig(num_to_keep=None)
                run_config = ray.train.RunConfig(storage_path=args.storage_path,
                                                 name=args.name, 
                                                 checkpoint_config=checkpoint_config)
                if not checkpoint:
                    checkpoint = Checkpoint(args.resume_from_checkpoint) if args.resume_from_checkpoint else None 
                trainer = ray.train.torch.TorchTrainer(train_multitask, 
                                                       train_loop_config=args, 
                                                       scaling_config=scaling_config, 
                                                       run_config=run_config,
                                                       resume_from_checkpoint=checkpoint)
                result = trainer.fit()
                checkpoint = result.checkpoint
                if args.epochs_per_task:
                    args.max_epochs += args.epochs_per_task
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
