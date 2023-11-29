import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,math
from torch.nn import init


def alpha_to_trimap_2(alpha):
	trimap = torch.ones_like(alpha)
	trimap[alpha==0] = 0
	trimap[alpha==1] = 1
	return trimap

    # swish
    # return x * torch.sigmoid(x)







def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=3):
    out = 1 - dilate(1 - bin_img, ksize)
    return out



def _generate_trimap(self,alpha):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128



def alpha_to_trimap(alpha):
	#fg = [alpha==0]

	# fg = np.array(np.equal(alpha, 255).astype(np.float32))
	# unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	# fg = torch.tensor(alpha==2)#.astype(torch.float32)
	# unknown = torch.tensor(alpha==0)#.astype(torch.float32))

	fg = torch.zeros_like(alpha)[alpha==1]=1
	unknown = torch.zeros_like(alpha)[alpha!=0]=0.5

	#unknown = dilate(unknown)
	trimap = fg * 1 + (unknown - fg) * 0.5


	# trimap = torch.ones_like(alpha)
	# trimap[alpha==0] = 0
	# trimap[alpha==2] = 2
	# trimap = dilate(trimap)
	# trimap[alpha==2] = 2

	return trimap


# def alpha_to_trimap(alpha):
# 	#fg = [alpha==0]

# 	# fg = np.array(np.equal(alpha, 255).astype(np.float32))
# 	# unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
# 	fg = torch.equal(alpha, 255)

# 	trimap = torch.ones_like(alpha)
# 	trimap[alpha==0] = 0
# 	trimap[alpha==2] = 2
# 	trimap = dilate(trimap)
# 	trimap[alpha==2] = 2

# 	return trimap


def alpha_to_trimap_2(alpha):
	# print (alpha.max(),alpha.min())

	trimap = torch.ones_like(alpha)/2.0
	trimap[alpha==0.0] = 0.0
	trimap[alpha==1.0] = 1.0
	return trimap




def remove_prefix_state_dict(state_dict, prefix="module"):
    """Removes the prefix from the key of pretrained state dict.

    Arguments:
        state_dict: The state dict to be modified.
        prefix: The prefix to be removed."""

    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()

    return new_state_dict

'''
def _tiled_processing(rim, y, x, h):
        """Process an image in tiles.
        Arguments:
            y: The observed input RGB image. A torch.Tensor of shape [Bx3xHxW].
            x: The tensor containing the candidate solutions. A torch.Tensor of shape [Bx7xHxW].
            h: The list of the hidden states of the RIM.
        Returns:
            The new candidate solution.
            The list of new hidden states."""

        _, _, height, width = x.shape
        tile_size = 512
        # Check if the image is smaller than the tile size.
        if height < self.tile_size and width < self.tile_size:
            x, h = self.rim(y, x, h)
            return x, h
        else:
            # The image needs to be tiled. Calculate number of vertical and horizontal tiles.
            vertical_tiles = int(np.ceil(height / (self.tile_size - self.rf + 1)))
            horizontal_tiles = int(np.ceil(width / (self.tile_size - self.rf + 1)))
            # Calculate padding.
            pad = (self.rf - 1) // 2
            bottom_pad = (vertical_tiles * (self.tile_size - self.rf + 1)) + pad - height
            left_pad = (horizontal_tiles * (self.tile_size - self.rf + 1)) + pad - width
            # Pad the input tensors.
            x = F.pad(x, (pad, left_pad, pad, bottom_pad))
            y = F.pad(y, (pad, left_pad, pad, bottom_pad))
            outputs = []  # Tiled results will be stored here.

            # Iterate over tiles and process.
            for i in range(vertical_tiles):
                for j in range(horizontal_tiles):
                    x_tiled = self._create_tensor_tile(x, i, j)
                    y_tiled = self._create_tensor_tile(y, i, j)
                    if h is not None:  # Create tiles of the hidden states.
                        h_tiled = self._create_hidden_tile(h, i, j)
                    else:
                        h_tiled = h

                    out = self.rim(y_tiled, x_tiled, h_tiled)
                    outputs.append(out)

            # If the initial hidden state was not yet defined, create a new empty list of hidden states.
            if h is None:
                sizes = [h_tiled.shape for h_tiled in outputs[0][1]]
                h = [torch.zeros(size[0], size[1], vertical_tiles * size[2], horizontal_tiles * size[3]).to(x.device) for size in sizes]

            # Put the tiled outputs back together to form a full sized output.
            k = 0
            for i in range(vertical_tiles):
                for j in range(horizontal_tiles):
                    out = outputs[k]
                    k += 1
                    x[:, :,
                      i*(self.tile_size - self.rf + 1)+pad:i*(self.tile_size - self.rf + 1)+self.tile_size-pad,
                      j*(self.tile_size - self.rf + 1)+pad:j*(self.tile_size - self.rf + 1)+self.tile_size-pad] = out[0][:, :, pad:-pad, pad:-pad]
                    for layer in range(len(h)):
                        h[layer][:, :,
                                 i*(int(self.tile_size * self.scales[layer])):i*(int(self.tile_size * self.scales[layer]))
                                                                              + int(self.tile_size * self.scales[layer]),
                                 j*(int(self.tile_size * self.scales[layer])):j*(int(self.tile_size * self.scales[layer]))
                                                                              + int(self.tile_size * self.scales[layer])] = out[1][layer]
            return x[:, :, pad:pad + height, pad:pad + width], 
'''