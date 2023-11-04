import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os


class ColorTransformModule:
    """Color transform module
    """
    def __init__(self,
                 device,
                 net_depth=1, 
                 net_width=64, 
                 net_activation=F.relu, 
                 color_activation=F.sigmoid,
                 multires_color=4,
                 lr=5e-2,
                 use_rand_rgb_cycle_forward=True,
                 use_rand_rgb_cycle_backward=True,
                 use_train_image_cycle=True,
                 weight_rand_rgb_cycle_forward=10.0,
                 weight_rand_rgb_cycle_backward=10.0,
                 weight_train_image_cycle=10.0,
                 weight_rgb_photo=1.0,
                 rand_color_batch=2**14,
                 pretrain_iters=1000,
                 rgbnet_out_dim=3,
                 ):

        self.use_rand_rgb_cycle_forward = use_rand_rgb_cycle_forward
        self.use_rand_rgb_cycle_backward = use_rand_rgb_cycle_backward
        self.use_train_image_cycle = use_train_image_cycle
        self.weight_rand_rgb_cycle_forward = weight_rand_rgb_cycle_forward
        self.weight_rand_rgb_cycle_backward = weight_rand_rgb_cycle_backward
        self.weight_train_image_cycle = weight_train_image_cycle
        self.weight_rgb_photo = weight_rgb_photo
        self.device = device
        self.pretrain_step = 0
        self.rand_color_batch = rand_color_batch
        self.pretrain_iters = pretrain_iters
        self.rgbnet_out_dim = rgbnet_out_dim
        
        self.color_encoder = ColorEncoder(net_depth, 
                                          net_width,
                                          net_activation,
                                          color_activation,
                                          multires_color,
                                          input_dim=3,
                                          output_dim=rgbnet_out_dim).to(self.device)

        self.color_decoder = ColorEncoder(net_depth, 
                                          net_width,
                                          net_activation,
                                          color_activation,
                                          multires_color,
                                          input_dim=rgbnet_out_dim,
                                          output_dim=3).to(self.device)

        self.optimizer = torch.optim.Adam(
            params=list(self.color_encoder.parameters()) + list(self.color_decoder.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
        )
    
    def encode_color(self, c, requires_grad=True):
        if not requires_grad:
            with torch.no_grad():
                return self.color_encoder(c)

        return self.color_encoder(c)

    def decode_color(self, c, requires_grad=True):
        if not requires_grad:
            with torch.no_grad():
                return self.color_decoder(c)

        return self.color_decoder(c)
    
    def cycle_color_forward(self, c, requires_grad=True):
        if not requires_grad:
            with torch.no_grad():
                return self.color_decoder(self.color_encoder(c))

        return self.color_decoder(self.color_encoder(c))

    def cycle_color_backward(self, c, requires_grad=True):
        if not requires_grad:
            with torch.no_grad():
                return self.color_encoder(self.color_decoder(c))

        return self.color_encoder(self.color_decoder(c))

    def pretrain(self, images, bg=-1):
        if self.pretrain_step != 0:
            return

        if bg >= 0:
            bg_color = torch.ones((1, 3)).to(self.device) * bg

        for i in tqdm(range(self.pretrain_iters), desc='Pretrain color transform module'):
            self.optimizer.zero_grad()
            img_i = np.random.choice(len(images))
            train_image = images[img_i]
            loss = self.identity_loss(train_image)
            # loss for background
            if bg >= 0:
                loss += ((bg_color - self.cycle_color_forward(bg_color)) ** 2).mean()

            loss.backward()
            self.optimizer.step()
        
        self.pretrain_step = self.pretrain_iters

    def identity_loss(self, image):
        """Regularize f(g) == I
        """

        loss = 0
        if self.use_rand_rgb_cycle_forward or self.use_rand_rgb_cycle_backward:
            # rand_rgb = torch.rand((self.rand_color_batch, 3)).to(self.device)
            rand_rgb = torch.cuda.FloatTensor(self.rand_color_batch, 3).uniform_()
            if self.rgbnet_out_dim == 3:
                rand_latent = rand_rgb
            else:
                rand_latent = torch.cuda.FloatTensor(self.rand_color_batch, self.rgbnet_out_dim).uniform_()
        
        if self.use_rand_rgb_cycle_forward:
            loss += self.weight_rand_rgb_cycle_forward * ((rand_rgb - self.cycle_color_forward(rand_rgb)) ** 2).mean() 
            
        if self.use_rand_rgb_cycle_backward:
            loss += self.weight_rand_rgb_cycle_backward * ((rand_latent - self.cycle_color_backward(rand_latent)) ** 2).mean() 

        # apply image cycle only for the forward pass.
        if self.use_train_image_cycle:
            loss += self.weight_train_image_cycle * ((image - self.cycle_color_forward(image)) ** 2).mean()

        return loss

    def save_model(self, path):
        ckpt = {
            'color_encoder_state_dict': self.color_encoder.state_dict(),
            'color_decoder_state_dict': self.color_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'pretrain_step': self.pretrain_step,
        }

        torch.save(ckpt, path)
        print('[INFO] Save checkpoint of color transform module at {}'.format(path))

    def load_model(self, path):
        assert os.path.exists(path)

        ckpt = torch.load(path)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.color_encoder.load_state_dict(ckpt['color_encoder_state_dict'])
        self.color_decoder.load_state_dict(ckpt['color_decoder_state_dict'])
        self.pretrain_step = ckpt['pretrain_step']

        print('[INFO] Load checkpoint of color transform module from {}'.format(path))

        

class ColorEncoder(nn.Module):
    """Color encoder module.
    Currently, decoder uses the same model.
    """
    def __init__(self, 
                 net_depth=1, 
                 net_width=64, 
                 net_activation=F.relu, 
                 color_activation=F.sigmoid,
                 multires_color=4,
                 input_dim=3,
                 output_dim=3,):
        """
        Args:
            net_depth: int
            net_width: int
            net_activation: F.relu
            color_activation: F.sigmoid
            multires_color: int, log scale of positional encoding scale. -1 means identity
        """
        super(ColorEncoder, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_activation = net_activation
        self.color_activation = color_activation
        self.multires_color = multires_color

        self.embed_fn, input_ch = get_embedder(self.multires_color, self.multires_color, input_dim)

        self.input_linear = nn.ModuleList(
            [nn.Linear(input_ch, self.net_width)] + 
            [nn.Linear(self.net_width, self.net_width) for _ in range(self.net_depth - 1)])
        
        self.output_linear = nn.Linear(self.net_width, output_dim)

    def forward(self, c):
        h = self.embed_fn(c)
        for i, l in enumerate(self.input_linear):
            h = self.input_linear[i](h)
            h = self.net_activation(h)
            
        h = self.output_linear(h)
        outputs = self.color_activation(h)

        return outputs

# from nerf-pytorch
# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), input_dim
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim