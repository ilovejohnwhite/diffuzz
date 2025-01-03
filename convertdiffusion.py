#convertdiffusion.py

import torch
import coremltools as ct
import numpy as np
import yaml
from denoising_diffusion_pytorch.mask_cond_unet import Unet
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
from fvcore.common.config import CfgNode

class VarReplacementWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def _replace_var_with_mean(self, x, dim=None, keepdim=True):
        mean = torch.mean(x, dim=dim, keepdim=keepdim) 
        squared_diff = (x - mean) ** 2
        variance = torch.mean(squared_diff, dim=dim, keepdim=keepdim)
        return variance

    def forward(self, x):
        def wrap_var(module):
            original_forward = module.forward
            def new_forward(*args, **kwargs):
                if hasattr(module, 'var'):
                    return self._replace_var_with_mean(*args, **kwargs)
                return original_forward(*args, **kwargs)
            module.forward = new_forward
            
        self.model.apply(wrap_var)
        return self.model(x)

class LatentDiffusionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        self.autoencoder = model.first_stage_model
        self.diffusion = model.model

    def manual_var(self, x, dim=None, keepdim=True, unbiased=False):
        mean = torch.mean(x, dim=dim, keepdim=keepdim)
        squared_diff = (x - mean) ** 2
        
        # Handle unbiased variance calculation
        if unbiased:
            # Get number of elements being averaged
            if dim is None:
                n = x.numel()
            else:
                if isinstance(dim, (tuple, list)):
                    n = 1
                    for d in dim:
                        n *= x.shape[d]
                else:
                    n = x.shape[dim]
            # Apply Bessel's correction
            return torch.sum(squared_diff, dim=dim, keepdim=keepdim) / (n - 1)
        else:
            return torch.mean(squared_diff, dim=dim, keepdim=keepdim)

    def forward(self, x):
        # Replace torch.var with manual_var
        torch.var = self.manual_var

        posterior = self.autoencoder.encode(x)
        latent = posterior.mode()
        mask = torch.zeros_like(x).repeat(1, 3, 1, 1)

        t = torch.zeros(x.shape[0], device=x.device)
        
        print("Shapes before diffusion:")
        print("Latent:", latent.shape)
        print("t:", t.shape)
        print("Mask:", mask.shape)

        diffusion_output = self.diffusion(latent, t, mask=mask)

        print("Type of diffusion output:", type(diffusion_output))

        noise = diffusion_output[0] if isinstance(diffusion_output, tuple) else diffusion_output
        output = self.autoencoder.decode(noise)

        return output

    @torch.no_grad()
    def trace_test(self, x):
        print("Encode:")
        posterior = self.autoencoder.encode(x)
        print("Get latent:")
        latent = posterior.mode()
        print("Create mask:")
        mask = torch.zeros_like(x).repeat(1, 3, 1, 1)
        print("Apply diffusion:")
        t = torch.zeros(x.shape[0], device=x.device)
        diffusion_output = self.diffusion(latent, t, mask=mask)
        print("Process output:", type(diffusion_output))
        noise = diffusion_output[0] if isinstance(diffusion_output, tuple) else diffusion_output
        print("Decode:")
        output = self.autoencoder.decode(noise)
        return output
    
def convert_to_coreml():
    # Load configs
    with open("configs/NYUD_train.yaml", 'r') as f:
        train_config = yaml.safe_load(f)
    with open("configs/NYUD_sample.yaml", 'r') as f:
        sample_config = yaml.safe_load(f)

    cfg = CfgNode(train_config)
    sample_cfg = CfgNode(sample_config)  # Convert sample config to CfgNode too
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage

    # Create model instances
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path
    )

    # Create U-Net
    unet_cfg = model_cfg.unet
    unet = Unet(
        dim=unet_cfg.dim,
        channels=unet_cfg.channels,
        dim_mults=unet_cfg.dim_mults,
        learned_variance=unet_cfg.get('learned_variance', False),
        out_mul=unet_cfg.out_mul,
        cond_in_dim=unet_cfg.cond_in_dim,
        cond_dim=unet_cfg.cond_dim,
        cond_dim_mults=unet_cfg.cond_dim_mults,
        window_sizes1=unet_cfg.window_sizes1,
        window_sizes2=unet_cfg.window_sizes2,
        fourier_scale=unet_cfg.fourier_scale,
        cfg=unet_cfg,
    )

    # Create Latent Diffusion Model
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=sample_cfg.sampler.get('sampling_timesteps', model_cfg.sampling_timesteps),  # Use sample_cfg instead
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path="diffusion_edge_natrual.pt",
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    
    # Debug prints
    print(f"First stage model input conv weight shape: {first_stage_model.encoder.conv_in.weight.shape}")
    print(f"UNet first conv weight shape: {unet.init_conv[0].weight.shape}")
    
    # Create example input
    example_input = torch.randn(1, 1, 320, 320)
    print(f"Example input shape: {example_input.shape}")
    
# Create wrapped model
    wrapped_model = LatentDiffusionWrapper(ldm)
    wrapped_model.eval()

    # Create example input
    example_input = torch.randn(1, 1, 320, 320)
    
    # Trace and convert
    try:
        print("\nTracing model...")
        traced_model = torch.jit.trace(wrapped_model, example_input)
        
        print("\nConverting to Core ML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name="input",
                    shape=(1, 1, 320, 320),
                    scale=1/255.0,
                    bias=[0],
                    color_layout=ct.colorlayout.GRAYSCALE
                )
            ],
            outputs=[ct.TensorType(name="output")],
            source="pytorch",
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL
        )
        
        mlmodel.save("DiffusionEdge.mlpackage")
        print("Conversion successful!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        convert_to_coreml()
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()