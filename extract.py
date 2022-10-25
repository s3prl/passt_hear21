import torch
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()

module = importlib.import_module(f"hear21passt.{args.name}")
model = module.load_model().cuda()
seconds = 15
audio = torch.ones((3, 32000 * seconds))*0.5
embed, time_stamps = module.get_timestamp_embeddings(audio, model)
print(embed.shape)
embed = module.get_scene_embeddings(audio, model)
print(embed.shape)
