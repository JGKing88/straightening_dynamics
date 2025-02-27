import os
import sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import LoggerType, DummyOptim, DummyScheduler
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
    AutoConfig,
)
from tqdm.auto import tqdm
import math
from straightening_dynamics.train.utils import Group_Texts
from pathlib import Path
import yaml
import argparse
from straightening_dynamics.train.run_manager import RunManager
from straightening_dynamics.models.gpt2.utils import initialize_gpt2_weights
import pickle


"""
Basic script to train a distill-gpt2 model using accelerate and grouping function.
Set config to use DeepSpeed
'accelerate config' -> enter in desired DeepSpeed configs or input path to deepspeed_config.json
'accelerate launch bplm/basic_accelerate_addedAug2022.py'
"""

# Evaluate function
def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch = torch.stack(batch["input_ids"]).transpose(1, 0)
            outputs = model(batch, labels=batch)
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    accelerator.print(
        f"validation loss: {loss.item()}, validation perplexity {perplexity.item()}"
    )
    return loss.item(), perplexity.item()


def train(config, run_id=None):
    # Extract configuration parameters
    model_name = config["model"]["name"]
    context_length = config["model"]["context_length"]
    
    dataset_size = config["data"]["dataset_size"]
    dataset_path = config["data"]["dataset_path"]
    
    batch_size = config["training"]["batch_size"]
    max_gpu_batch_size = config["training"]["max_gpu_batch_size"]
    eval_batch_size = config["training"]["eval_batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    seed = config["training"]["seed"]
    scheduler_type = config["training"]["scheduler"]
    warmup_steps = config["training"]["warmup_steps"]
    save_steps = config["training"]["save_steps"]
    eval_steps = config["training"]["eval_steps"]
    
    init_type = config["initialization"]["init_type"]
    init_std = config["initialization"]["init_std"]
    init_mean = config["initialization"]["init_mean"]
    
    checkpoint = config["run"]["checkpoint"]
    wandb_project_name = config["run"]["wandb_project_name"]
    
    # Set random seed
    set_seed(seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    grouped_pad_train = load_from_disk(
        os.path.join(
            dataset_path,
            f"train_context_len_{context_length}",
        )
    )
    grouped_pad_test = load_from_disk(
        os.path.join(
            dataset_path,
            f"test_context_len_{context_length}",
        )
    )
    grouped_pad_valid = load_from_disk(
        os.path.join(
            dataset_path,
            f"valid_context_len_{context_length}",
        )
    )

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    if batch_size > max_gpu_batch_size:
        gradient_accumulation_steps = batch_size // max_gpu_batch_size
        batch_size = max_gpu_batch_size
    
    # Initialize accelerator
    accelerator = Accelerator(
        log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps
    )

    # Create dataloaders
    eval_dataloader = DataLoader(
        grouped_pad_valid, shuffle=False, batch_size=eval_batch_size
    )
    test_dataloader = DataLoader(
        grouped_pad_test, shuffle=True, batch_size=eval_batch_size
    )
    train_dataloader = DataLoader(
        grouped_pad_train, shuffle=True, batch_size=batch_size
    )
    del grouped_pad_train, grouped_pad_valid, grouped_pad_test

    # Create run manager and get run directory
    run_manager = RunManager()
    if run_id is None:
        # This is a standalone run, not part of a sweep
        run_id = run_manager.setup_run(config)
    
    run_dir = run_manager.get_run_dir(run_id)

    # Logging initialization
    # Change name test to log to different project
    accelerator.init_trackers(
        wandb_project_name, config=config, init_kwargs={"wandb": {"name": run_id}}
    )
    
    # Initialize model
    config_obj = AutoConfig.from_pretrained(
        model_name, 
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = AutoModelForCausalLM.from_config(config_obj)
    
    # Initialize weights or load checkpoint
    if init_type == "gaussian":
        state_dict = initialize_gpt2_weights(model, permute=False)
    else:
        state_dict = model.state_dict()
        
    if checkpoint is not None:
        save_dir = Path(checkpoint)
        chkpnt_model = AutoModelForCausalLM.from_pretrained(save_dir)
        state_dict = chkpnt_model.state_dict()
        
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config_obj, state_dict=state_dict)
    del state_dict

    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
    model = model.to(accelerator.device)

    # Define optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates AdamW Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=learning_rate)
    
    # Define scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        if scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=(len(train_dataloader) * num_epochs)
                // gradient_accumulation_steps,
            )
        else:  # linear
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=(len(train_dataloader) * num_epochs)
                // gradient_accumulation_steps,
            )
    else:
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            total_num_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
            warmup_num_steps=warmup_steps,
        )

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
        )
    )

    # Logging variables
    batch_count = 0
    n_steps_per_epoch = math.ceil(
        len(train_dataloader.dataset) / batch_size
    )
    
    # Begin training for number of epochs
    abs_step = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        torch.cuda.empty_cache()
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            batch = [
                torch.stack(batch[x]).transpose(1, 0)
                for x in ["input_ids", "attention_mask"]
            ]
            total_loss = 0
            with accelerator.accumulate(model):
                outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                loss = outputs.loss
                total_loss += loss
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            batch_count += len(batch)
            accelerator.log({"train/train_loss": total_loss}, step=abs_step)
            accelerator.log(
                {
                    "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                    / n_steps_per_epoch
                },
                step=abs_step,
            )
            accelerator.log({"train/batch_count": batch_count}, step=abs_step)
            
            # Evaluate and save model
            if abs_step % eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_perplexity = evaluate(model, eval_dataloader, accelerator)
                accelerator.log({"validation/valid_loss": valid_loss}, step=abs_step)
                accelerator.log(
                    {"validation/valid_perplexity": valid_perplexity}, step=abs_step
                )
                
                # Save model checkpoint
                if abs_step % save_steps == 0:
                    checkpoint_dir = run_dir / f"checkpoint_{abs_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        checkpoint_dir, 
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model)
                    )
                    
                    # Save optimizer and scheduler states
                    accelerator.save(
                        {
                            "epoch": epoch,
                            "steps": abs_step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                        },
                        os.path.join(checkpoint_dir, 'accelerator_states')
                    )
                    
                    # Save gradients
                    gradients = {}
                    for name, param in unwrapped_model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            gradients[name] = param.grad.detach().cpu().clone()
                    
                    # Save gradients to the checkpoint directory
                    torch.save(gradients, os.path.join(checkpoint_dir, 'gradients.pt'))
                
            abs_step += 1
            
    # Save final model
    final_checkpoint_dir = run_dir / "final_checkpoint"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        final_checkpoint_dir, 
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    
    # Save final optimizer and scheduler states
    accelerator.save(
        {
            "epoch": num_epochs - 1,
            "steps": abs_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        },
        os.path.join(final_checkpoint_dir, 'accelerator_states')
    )
    
    # Save final gradients
    gradients = {}
    for name, param in unwrapped_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.detach().cpu().clone()
    
    # Save gradients to the final checkpoint directory
    torch.save(gradients, os.path.join(final_checkpoint_dir, 'gradients.pt'))
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_loss, test_perplexity = evaluate(model, test_dataloader, accelerator)
    
    # Update final metrics in the runs dataframe
    run_manager.update_metrics(run_id, {
        "final_valid_loss": valid_loss,
        "final_valid_perplexity": valid_perplexity,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "total_steps": abs_step,
        "total_epochs": num_epochs,
        "model_size_M": model_size / 1000 ** 2
    })
    
    accelerator.end_training()
    torch.cuda.empty_cache()
    
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a configuration")
    parser.add_argument("--config", type=str, default="straightening_dynamics/train/configs/train_config.yaml",
                        help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Train model
    run_id = train(config)
    print(f"Training completed. Run ID: {run_id}")
