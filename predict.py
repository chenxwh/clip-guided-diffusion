import cog
import tempfile
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from model import *
import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_and_diffusion_defaults()

    @cog.input("prompt", type=str, help="Text prompt")
    @cog.input("timesteps", type=int, help="Number of timesteps", default=1000)
    def predict(self, prompt, timesteps=1000):

        self.model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timesteps),  # Modify this value to decrease the number of timesteps.
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })

        model, diffusion = create_model_and_diffusion(**self.model_config)
        model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)
        for name, param in model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if self.model_config['use_fp16']:
            model.convert_to_fp16()

        clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(self.device)
        clip_size = clip_model.visual.input_resolution
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        batch_size = 1
        clip_guidance_scale = 1000  # Controls how much the image should look like the prompt.
        tv_scale = 150  # Controls the smoothness of the final output.
        cutn = 16
        n_batches = 1
        init_image = None  # This can be an URL or Colab local path and must be in quotes.
        skip_timesteps = 0  # This needs to be between approx. 200 and 500 when using an init image.
        # Higher values make the output look more like the init.
        seed = 0

        if seed is not None:
            torch.manual_seed(seed)

        text_embed = clip_model.encode_text(clip.tokenize(prompt).to(self.device)).float()

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((self.model_config['image_size'], self.model_config['image_size']), Image.LANCZOS)
            init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        make_cutouts = MakeCutouts(clip_size, cutn)

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                n = x.shape[0]
                my_t = torch.ones([n], device=self.device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
                image_embeds = clip_model.encode_image(clip_in).float().view([cutn, n, -1])
                dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
                losses = dists.mean(0)
                tv_losses = tv_loss(x_in)
                loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
                return -torch.autograd.grad(loss, x)[0]

        if self.model_config['timestep_respacing'].startswith('ddim'):
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.p_sample_loop_progressive

        out_path = Path(tempfile.mkdtemp()) / "out.png"

        for i in range(n_batches):
            cur_t = diffusion.num_timesteps - skip_timesteps - 1
            samples = sample_fn(
                model,
                (batch_size, 3, self.model_config['image_size'], self.model_config['image_size']),
                clip_denoised=False,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=init,
                randomize_class=True,
            )

            for j, sample in enumerate(samples):
                cur_t -= 1
                if cur_t == -1:
                    for k, image in enumerate(sample['pred_xstart']):
                        TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(str(out_path))

        return out_path


