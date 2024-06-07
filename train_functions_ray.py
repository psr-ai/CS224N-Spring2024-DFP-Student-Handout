from ray.train.torch import prepare_model, prepare_data_loader
from train_functions import training_loop as single_training_loop

def training_loop(args, model, optimizer, compute_batch_loss, train_dataloader, dev_dataloader, device, config, eval_fn):
    model = prepare_model(model)
    train_loader = prepare_data_loader(train_dataloader)
    dev_loader = prepare_data_loader(dev_dataloader)
    single_training_loop(args, model, optimizer, compute_batch_loss, train_loader, dev_loader, device, config, eval_fn)

    
