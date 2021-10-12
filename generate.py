# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import glob
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import base58

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--count', 'total_count', type=int, default=32, help='Number of images to generate')
@click.option('--conf', 'min_confidence', type=float, default=-1.0, help='Minimum confidence of images to keep')
@click.option('--batch', 'batch_size', type=int, default=32, help='Batch size.')
@click.option('--zvec', 'zvec_infile', type=str, help='Path to b58 vector file.')
@click.option('--zperm', 'zvec_noise_perm', type=float, default=0.01, help='Amount of noise to permute zvec.')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    total_count : int,
    min_confidence : float,
    network_pkl: str,
    batch_size : int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    projected_w: Optional[str],
    zvec_infile : str,
    zvec_noise_perm : float,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        loaded = legacy.load_network_pkl(f)
        G = loaded['G_ema'].to(device) # type: ignore
        D = loaded['D'].to(device) # type: ignore

    zvec_in = None
    if zvec_infile:
        zvec_bytes = base58.b58decode(open(zvec_infile, 'rt').read())
        zvec_in = np.fromstring(zvec_bytes, "<f4")
        zvec_in = torch.tensor(zvec_in).to(device)

    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)

    while total_count > 0:
        z = torch.randn((batch_size, G.z_dim)).to(device)
        # z = z / torch.norm(z, 2, dim=-1, keepdim=True)
        if zvec_in is not None:
            z = z * zvec_noise_perm + zvec_in[None, ...]

        with torch.no_grad():
            images = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            logits = D(images, label)
            z = z.cpu().numpy()

        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        logits = logits[:, 0]
        odds = torch.exp(logits).cpu().numpy()
        probs = 1.0 / (1.0 + odds)

        for i, prob in enumerate(probs):
            prob = float(prob)
            if prob < min_confidence:
                continue

            img, zvec = images[i], z[i]

            dircount = len(glob.glob(f"{outdir}/*.png"))
            outpath = f'{outdir}/img_{dircount:04d}.png'
            print(f"Saving: {outpath} (prob={prob*100:.02f}%)")
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(outpath)

            zvec_b58 = zvec.astype("<f4").tostring()
            zvec_b58 = base58.b58encode(zvec_b58)
            with open(outpath.replace('.png', '_zvec.b58'), 'wt') as fptr:
                fptr.write(zvec_b58.decode('utf-8'))

            total_count -= 1
            if total_count <= 0:
                break

    print("Done!")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
