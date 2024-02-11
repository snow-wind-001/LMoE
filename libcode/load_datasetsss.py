
from datasets import load_dataset
from huggingface_hub import login
login()
dataset = load_dataset("shibing624/medical")

dataset.save_to_disk("/home/user/DISK/datasets/LMOE/new")