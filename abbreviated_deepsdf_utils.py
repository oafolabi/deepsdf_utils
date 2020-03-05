import sys
import os 
import numpy as np
import json 


sys.path.append('/home/oladapo/code/DeepSDF/')  # change this to point to where deepsdf is for you
import deep_sdf
from collections import OrderedDict
import torch 


device = torch.device("cuda")

def load_deepsdf_model(model_dir, device=torch.device("cpu"), visible_devices=None):
    with torch.no_grad():
        experiment_directory = model_dir

        specs_filename = os.path.join(experiment_directory, "specs.json")

        if not os.path.isfile(specs_filename):
            raise Exception(
                'The experiment directory does not include specifications file "specs.json"'
            )

        specs = json.load(open(specs_filename))

        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

        latent_size = specs["CodeLength"]

        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
        # decoder = torch.nn.DataParallel(decoder)
        # decoder.to(device)

        # decoder = torch.nn.DataParallel(decoder)
        
        # put on multiple GPUs or just CPU
        if device.type == "cuda":
            if visible_devices is None:
                decoder = torch.nn.DataParallel(decoder)
                decoder.to(device)
            else: 
                # if with multiple GPUs, use specified ones
                decoder = torch.nn.DataParallel(decoder, device_ids=visible_devices)
                decoder.to(torch.device(type="cuda", index=visible_devices[0]))
        else:
            decoder.to(device)

        # set model to evaluation mode
        decoder.eval()

   
        model_save_path = model_dir + 'ModelParameters/latest.pth'
        saved_model_state = torch.load(model_save_path, map_location=device)
        print(saved_model_state["model_state_dict"].keys())
        tmp_model_state = saved_model_state["model_state_dict"]
        new_state_dict = OrderedDict()
        if device.type == "cpu":
            new_state_dict = OrderedDict()
            for k, v in tmp_model_state.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = tmp_model_state

        print(new_state_dict.keys())

        decoder.load_state_dict(new_state_dict)
        # decoder.load_state_dict(torch.load(model_save_path, map_location=device))
        decoder.to(device)

        latent_codes_save_path = model_dir + 'LatentCodes/latest.pth'
        latent_codes_dict = torch.load(latent_codes_save_path)
        latent_codes = latent_codes_dict['latent_codes']
        
        
        return decoder, latent_codes