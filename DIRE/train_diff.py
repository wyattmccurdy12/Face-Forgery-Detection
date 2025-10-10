from utils.config import cfg  # isort: split

import os
import time

# from tensorboardX import SummaryWriter
# from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger

if __name__ == "__main__":
    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    # data_loader = create_dataloader(cfg)

    ###################################################################################
    # insert own dataloader here: 
    
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(256, 256), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True), # Scale pixel values to [0.0, 1.0]
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create the training dataset using the full dataset
    train_dataset = FaceDataset(
        root_dir="train",
        sub_dir=args.type_modification,
        transform=transform
    )
    
    print(f"Using full dataset for training. Train size: {len(train_dataset)}")
    
    # Create the DataLoader for training
    data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True, # Shuffle training data each epoch
        num_workers=4,
        pin_memory=True
    )

    #######################################################################################
    
    dataset_size = len(data_loader)

    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    trainer = Trainer(cfg)
    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(cfg.nepoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for data in tqdm(data_loader, dynamic_ncols=True):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size

            trainer.set_input(data)
            trainer.optimize_parameters()

            # if trainer.total_steps % cfg.loss_freq == 0:
            #     log.write(f"Train loss: {trainer.loss} at step: {trainer.total_steps}\n")
            train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

            if trainer.total_steps % cfg.save_latest_freq == 0:
                log.write(
                    "saving the latest model %s (epoch %d, model.total_steps %d)\n"
                    % (cfg.exp_name, epoch, trainer.total_steps)
                )
                trainer.save_networks("latest")

        if epoch % cfg.save_epoch_freq == 0:
            log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)

        # Validation
        trainer.eval()
        val_results = validate(trainer.model, val_cfg)
        val_writer.add_scalar("AP", val_results["AP"], trainer.total_steps)
        val_writer.add_scalar("ACC", val_results["ACC"], trainer.total_steps)
        log.write(f"(Val @ epoch {epoch}) AP: {val_results['AP']}; ACC: {val_results['ACC']}\n")

        if cfg.earlystop:
            early_stopping(val_results["ACC"], trainer)
            if early_stopping.early_stop:
                if trainer.adjust_learning_rate():
                    log.write("Learning rate dropped by 10, continue training...\n")
                    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    log.write("Early stopping.\n")
                    break
        if cfg.warmup:
            # print(trainer.scheduler.get_lr()[0])
            trainer.scheduler.step()
        trainer.train()
