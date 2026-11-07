"""The from-scratch model from text_to_image_diffusion_from_scratch.ipynb, for use outside the notebook.

The prompt lists, the U-Net and the sampler are copies of the notebook's, so a model trained there loads here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Words for each category, in the dataset's order.
WORDS = [
    ["t-shirt", "top", "tee"],
    ["pair of trousers", "pair of pants", "pair of jeans"],
    ["pullover", "sweater", "jumper"],
    ["dress", "summer dress"],
    ["coat", "jacket", "winter coat"],
    ["sandal", "summer sandal"],
    ["shirt", "button-up shirt"],
    ["sneaker", "trainer", "running shoe"],
    ["bag", "handbag"],
    ["ankle boot", "boot"],
]
TRAIN_TEMPLATES = ["a {}", "a photo of a {}", "a picture of a {}", "an image of a {}",
                   "a black and white photo of a {}", "a small photo of a {}"]
# One new wording per category, never used in training.
TEST_PROMPTS = ["a short-sleeved t-shirt", "a pair of long trousers", "a warm woollen sweater",
                "an elegant evening dress", "a long winter coat", "an open-toed sandal", "a collared shirt",
                "a white running shoe", "a leather handbag", "a leather ankle boot"]
OFF_TOPIC = ["a cat", "a red sports car", "a bowl of soup", "a mountain at sunset", "a guitar", "a bicycle"]


def timestep_embedding(t, dim):
    # Turns each noise level into a list of sine and cosine values, so nearby levels look similar.
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]   # the noise level and prompt steer this block
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, ch, text_dim):
        super().__init__()
        emb = ch * 4
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.text_mlp = nn.Sequential(nn.Linear(text_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.null_text = nn.Parameter(torch.zeros(text_dim))   # stands in for "no prompt"
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, emb)                                # 28x28
        self.shrink1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)          # to 14x14
        self.down2 = ResBlock(ch, ch * 2, emb)                            # 14x14
        self.shrink2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)  # to 7x7
        self.mid1 = ResBlock(ch * 2, ch * 4, emb)                         # 7x7
        self.attn = SelfAttention(ch * 4)
        self.mid2 = ResBlock(ch * 4, ch * 2, emb)
        self.up2 = ResBlock(ch * 4, ch, emb)                              # 14x14, after joining down2's output
        self.up1 = ResBlock(ch * 2, ch, emb)                              # 28x28, after joining down1's output
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t, text, use_prompt):
        text = torch.where(use_prompt[:, None], text, self.null_text.to(text.dtype).expand_as(text))
        emb = self.time_mlp(timestep_embedding(t, self.ch)) + self.text_mlp(text)
        h1 = self.down1(self.inc(x), emb)
        h2 = self.down2(self.shrink1(h1), emb)
        h = self.mid2(self.attn(self.mid1(self.shrink2(h2), emb)), emb)
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2.0, mode="nearest"), h2], dim=1), emb)
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=2.0, mode="nearest"), h1], dim=1), emb)
        return self.out(h)


class ScratchModel:
    """A trained checkpoint from the notebook, with CLIP for prompts and the same sampler."""

    def __init__(self, checkpoint_path, device):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        from transformers.utils import logging as transformers_logging

        # weights_only refuses anything in the file except tensors and plain values.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.device = device
        self.use_amp = device == "cuda"
        self.class_names = checkpoint["class_names"]
        self.net = UNet(checkpoint["base_channels"], checkpoint["text_dim"]).to(device)
        self.net.load_state_dict(checkpoint["ema"])
        self.net.eval()
        self.timesteps = checkpoint["timesteps"]
        betas = torch.linspace(1e-4, 0.02, self.timesteps, device=device)
        self.alpha_bars = torch.cumprod(1 - betas, dim=0)
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint["text_model"])
        # Loading only CLIP's text half lists every unused image weight. That report is expected, so it is hidden
        # while CLIP loads, and other warnings are left as they were.
        previous = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(checkpoint["text_model"]).to(device).eval()
        finally:
            transformers_logging.set_verbosity(previous)

        # The same closest-category check as the notebook, rebuilt from the same prompt lists.
        train_prompts = [[template.format(word) for word in WORDS[c] for template in TRAIN_TEMPLATES] for c in range(10)]
        self.centroids = F.normalize(torch.stack([self.encode(prompts).mean(0) for prompts in train_prompts]), dim=-1)
        test_score = (self.encode(TEST_PROMPTS) @ self.centroids.T).max(dim=1).values
        off_score = (self.encode(OFF_TOPIC) @ self.centroids.T).max(dim=1).values
        self.known_threshold = float((test_score.min() + off_score.max()) / 2)

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(list(texts), padding=True, return_tensors="pt").to(self.device)
        return F.normalize(self.text_encoder(**tokens).text_embeds.float(), dim=-1)

    def closest_category(self, prompt):
        # Returns the closest category's name, the similarity, and whether it clears the threshold.
        score, index = (self.encode([prompt]) @ self.centroids.T).max(dim=1)
        return self.class_names[int(index)], float(score), float(score) >= self.known_threshold

    @torch.no_grad()
    def generate(self, prompt, count, guidance, steps, seed, on_step=None):
        # Draws count 1x28x28 images (values -1 to 1) with the notebook's sampler. on_step(i, total) reports progress.
        device = self.device
        text = self.encode([prompt]).repeat_interleave(count, dim=0)
        x = torch.randn(count, 1, 28, 28, generator=torch.Generator(device=device).manual_seed(seed), device=device)
        both_text = torch.cat([text, text])
        use_prompt = torch.cat([torch.ones(count, dtype=torch.bool, device=device),
                                torch.zeros(count, dtype=torch.bool, device=device)])
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()
        for i, t in enumerate(times):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=self.use_amp):
                out = self.net(torch.cat([x, x]), t.repeat(2 * count), both_text, use_prompt).float()
            with_prompt, without_prompt = out.chunk(2)
            noise = without_prompt + guidance * (with_prompt - without_prompt)
            a = self.alpha_bars[t]
            a_next = self.alpha_bars[times[i + 1]] if i + 1 < len(times) else torch.ones((), device=device)
            clean = ((x - (1 - a).sqrt() * noise) / a.sqrt()).clamp(-1, 1)
            noise = (x - a.sqrt() * clean) / (1 - a).sqrt()
            x = a_next.sqrt() * clean + (1 - a_next).sqrt() * noise
            if on_step is not None:
                on_step(i, len(times))
        return x.clamp(-1, 1)
