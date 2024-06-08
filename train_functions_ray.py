from ray.train.torch import prepare_model, prepare_data_loader
from train_functions import training_loop as vanilla_training_loop

def training_loop(args, model, optimizer, compute_batch_loss, train_dataloader, dev_dataloader, device, config, eval_fn, task_type):
    vanilla_training_loop(args, prepare_model(model), optimizer, compute_batch_loss, prepare_data_loader(train_dataloader), prepare_data_loader(dev_dataloader), device, config, eval_fn, task_type)
    
