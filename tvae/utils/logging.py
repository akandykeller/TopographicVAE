import os

def get_dirs():
    cwd = os.path.dirname(os.path.realpath(__file__))

    local_savedir = cwd
    local_datadir = cwd
    local_wandbdir = cwd

    return local_savedir, local_datadir, local_wandbdir


def configure_logging(config, name, model):
    if config['wandb_on']:
        import wandb

        wandb.init(name=name,
                   project='YOUR_PROJECT_NAME', 
                   entity='YOUR_ENTITY_NAME', 
                   dir=config['wandb_dir'],
                   config=config)
        wandb.watch(model)

        def log(key, val):
            print(f"{key}: {val}")
            wandb.log({key: val})

        checkpoint_path = os.path.join(wandb.run.dir, 'checkpoint.tar')
    else:
        def log(key, val):
            print(f"{key}: {val}")
        checkpoint_path = './checkpoint.tar'

    return log, checkpoint_path
