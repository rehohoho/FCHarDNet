import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Implementation differs from paper:
Layers and growth rates:
    Paper hardnet-68: 
        layers: [8,(16,16),16,4]
        growth rate: [14, 16, 20, 40, 160]
        (emphasis on stride-8 for local feature learning)
    Implementation hardnet-70:
        layers: [4, 4, 8, 8, 8]
        growth rate: [10, 16, 18, 24, 32]
    Hardnet-70 is a segmentation task whereas models in paper are used for classification
    Can't seem to find any papers supporting this implementation

Not implemented yet:
New transition layer (after downsampling hardblock)
    Current: 1x1conv, avgpoolx0.5 (densenet transition layer)
    Paper: maxpool input, avgpoolx0.85 hardblock output, concat, 1x1conv
        (0.85x pooling since there is already low-dimension compression, m, within hardnet)
    Supposed outcome - Less CIO at 1x1 conv

PingoLH hardblock-v2 (not pulled) includes conv with biases
"""

class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)
        

class DetectorHead(nn.Module):

    def __init__(self, params):
        super().__init__()
        assert 'neurons' in params and 'dropouts' in params and len(params['neurons']) == len(params['dropouts'])

        neurons = [20480] + params['neurons'] + [1] # add flattened feature size and final output size
        dropouts = params['dropouts'] + [0.0]
        n_layers = len(neurons)
        self.layers = nn.ModuleList([])

        for layer in range(n_layers - 1):
            self.layers.append( nn.Linear(neurons[layer], neurons[layer+1]) )
            self.layers.append( nn.Dropout2d(p = dropouts[layer]) )
            
            if layer != n_layers - 2:
                self.layers.append( nn.ReLU(inplace = True) )

    def forward(self, x):
        
        out = x.view(x.size(0), -1) # flatten downsampled feature block

        for layer in range(len(self.layers)):
            out = self.layers[layer](out)
        
        return torch.squeeze(out, dim = -1)


class HarDBlock(nn.Module):
    """ 3x3conv - bn - relu layers with harmonic links (k-2**i) """

    def get_link(self, layer, base_ch, growth_rate, grmul):
        """ calculate number of input/output channels and record which layers are linked """

        if layer == 0:
          return base_ch, 0, []

        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i       # layer is only linked to harmonic layers (layer_idx - 2**i)
          if layer % dv == 0:
            k = layer - dv
            link.append(k)  # record linked layer idx
            if i > 0:       # scale output channels for additional linked layers (low-dim compression)
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2

        in_channels = 0     # count input channels for additional linked layers
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels

        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)                   # record links for each layer for forward pass
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))    # 3x3 conv - BNnorm - relu
          
          if (i % 2 == 0) or (i == n_layers - 1):   # odd + last layers concat as hardblock output
            self.out_channels += outch
            
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]                   # outputs from each layer, will be appended further
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []                    # record which linked outputs to append
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            
            out = self.layers[layer](x) # apply layer
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):              # concat odd + last layer as hardblock output
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])

        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    """ interpolate input to skip size, concat with skip (skip is downsampling hardblock output) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out


class hardnet(nn.Module):

    def _freeze_layers(self, freeze):
        
        modules_to_freeze = []
        if 'hardnet' in freeze:
            modules_to_freeze += [self.base, self.transUpBlocks, self.conv1x1_up, self.denseBlocksUp, self.finalConv]
        if 'detectors' in freeze:
            modules_to_freeze += [self.detector]
        
        for module in modules_to_freeze:
            if isinstance(module, nn.ModuleList):
                for child in module.children():
                    for param in child.parameters():
                        param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = False
        
        n_frozen = 0
        n_not_frozen = 0
        for param in self.parameters():
            n_not_frozen += param.requires_grad
            n_frozen += param.requires_grad == False
        
        print('Number of frozen layers: %d, Number of active layers: %d' %(n_frozen, n_not_frozen))

    def __init__(self, n_classes=19, detector_heads=None, freeze = []):
        super(hardnet, self).__init__()

        first_ch  = [16,24,32,48]           # downsampling conv channels
        ch_list = [  64, 96, 160, 224, 320] # conv channels after each hardblock
        grmul = 1.7                         # hardblock: channels multiplier with each additional link
        gr       = [  10,16,18,24,32]       # hardblock: base channels
        n_layers = [   4, 4, 8, 8, 8]       # hardblock: layers

        blks = len(n_layers) 
        self.shortcut_layers = []           # skip idx used for forward pass

        #######################
        #   Downsampling path   #
        #######################

        self.base = nn.ModuleList([])
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2) )
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )
        self.base.append ( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )

        skip_connection_channel_counts = [] # skip channels used for upsampling concat
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)       # record skip layer number of channels
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1) # record skip layer idx

            # Densenet transition layer! Not the mapping method shown in paper that reduces CIO @ conv
            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append ( nn.AvgPool2d(kernel_size=2, stride=2) )

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))       # interpolate (skip size) and concat
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))  # upsample
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)
               
        self.detector = None
        if detector_heads is not None:
            self.detector = nn.ModuleList([])
            for detector, params in detector_heads.items():
                self.detector.append( DetectorHead(params = params) )
        
        self._freeze_layers(freeze)

    def forward(self, x):
        """ calls all the module lists in correct order """
        
        skip_connections = []
        size_in = x.size()
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:   # record outputs from skip layers
                skip_connections.append(x)
        out = x

        if self.detector is not None:
            detector_out = []
            for i in range(len(self.detector)):
                detector_out.append( self.detector[i](out) )        
            detector_out = torch.stack(detector_out, dim = 0).t()
        
        for i in range(self.n_blocks):
            skip = skip_connections.pop()   # get output from skip layers
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        
        out = self.finalConv(out)           # reduce space to number of classes
        
        out = F.interpolate(                # get back image size
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        
        if self.detector is not None:
            out = (out, detector_out)
        return out