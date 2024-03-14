import torch
import optuna
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from transformers import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

full_train_dataset = load_dataset("nlpai-lab/kullm-v2", split="train")
train_dataset, eval_dataset = train_test_split(full_train_dataset, test_size=0.1)

class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="Upstage/SOLAR-10.7B-v1.0", learning_rate=2e-5):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate

    def forward(self, input_ids, labels=None):
        output = self.model(input_ids, labels=labels)
        return output.loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        loss = self(input_ids=input_ids, labels=labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

def tokenize_and_encode(dataset, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return dataset.map(lambda examples: tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512), batched=True)

full_train_dataset = load_dataset("nlpai-lab/kullm-v2", split="train")
full_train_dataset = tokenize_and_encode(full_train_dataset, AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-v1.0"))
train_dataset, eval_dataset = train_test_split(full_train_dataset, test_size=0.1)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    
    model = TransformerModel(learning_rate=learning_rate)
    
    trainer = pl.Trainer(max_epochs=10, devices=2 if torch.cuda.is_available() else 0, accelerator="gpu", logger=False)
    
    trainer.fit(model, train_dataloader, eval_dataloader)
    eval_result = trainer.evaluate()
    
    return eval_result["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")