import torch
import diffusion

DModel = diffusion.DiffusionModel2(torch.linspace(1e-4, 0.02, 1000))
DModel.load_state_dict(torch.load('best_diffusion_model_cpu.pth'))