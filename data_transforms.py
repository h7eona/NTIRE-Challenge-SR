import numbers
import random

import numpy as np
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch

"""
Modified data transforms for [image, inter, gt] triplet
"""
class Resize(object):
        def __init__(self, size):
            self.size = size
        def __call__(self, image, label):
            return image.resize((self.size, self.size)), \
                    label.resize((self.size, self.size))

class Resize_pair(object):
        def __call__(self, image, label):
            w, h = image.size
            return image, label.resize((w, h))

class Resize_One(object):
        def __init__(self, size):
            self.size = size
        def __call__(self, image):
            return image.resize((self.size, self.size)),

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        assert label is None or (image.size == label.size), \
            "image and label doesn't have the same size {} / {}".format(
                image.size, label.size)
        w, h = image.size
        if w < self.size :
            image = image.resize((self.size, h))
            w, h = image.size
        if h < self.size :
            image = image.resize((w, self.size))
            w, h = image.size
        m = min(w, h)
        # crop_size = random.randint(self.size, m)
        # tw = crop_size
        # th = crop_size//3
        tw = self.size
        th = self.size//3
        # print(th, tw)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        results = [image.crop((x1, y1, x1 + tw, y1 + th))]
        if label is not None:
            results.append(label.crop((x1, y1, x1 + tw, y1 + th)))
        # print("## ",results[0].size)
        return results


class RandomCrop_One(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        w, h = image.size
        if w < self.size :
            image = image.resize((self.size, h))
            w, h = image.size
        if h < self.size :
            image = image.resize((w, self.size))
            w, h = image.size
        m = min(w, h)
        crop_size = random.randint(self.size, m)
        # tw = crop_size
        # th = crop_size//3
        tw = self.size
        th = self.size//3
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        results = [image.crop((x1, y1, x1 + tw, y1 + th))]
        return results
        

class RandomFlip(object):
    def __call__(self, image0, label0, image1, label1):
        A1 = random.random()
        A2 = random.random()
        if A1 < 0.5:
            image0 = image0.transpose(Image.FLIP_LEFT_RIGHT)
            label0 = label0.transpose(Image.FLIP_LEFT_RIGHT)
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            label1 = label1.transpose(Image.FLIP_LEFT_RIGHT)
        if A2 < 0.5 :
            image0 = image0.transpose(Image.FLIP_TOP_BOTTOM)
            label0 = label0.transpose(Image.FLIP_TOP_BOTTOM)
            image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
            label1 = label1.transpose(Image.FLIP_TOP_BOTTOM)
            
        return [image0, label0, image1, label1]

class RandomFlip_One(object):
    def __call__(self, image):
        A1 = random.random()
        A2 = random.random()
        if A1 < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if A2 < 0.5 :
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return [image]

class RandomRotate(object):
    def __call__(self, image, label):
        A = random.random()
        if  A < 0.25:
            angle = 90
            image = image.rotate(angle, resample=Image.BILINEAR)
            label = label.rotate(angle, resample=Image.BILINEAR)
        elif A < 0.5:
            angle = 180
            image = image.rotate(angle, resample=Image.BILINEAR)
            label = label.rotate(angle, resample=Image.BILINEAR)
        elif A < 0.75:
            angle = 270
            image = image.rotate(angle, resample=Image.BILINEAR)
            label = label.rotate(angle, resample=Image.BILINEAR)
        return image, label

class RandomRotate_One(object):
    def __call__(self, image):
        A = random.random()
        if A < 0.25:
            angle = 90
            image = image.rotate(angle, resample=Image.BILINEAR)
        elif A < 0.5:
            angle = 180
            image = image.rotate(angle, resample=Image.BILINEAR)
        elif A < 0.75:
            angle = 270
            image = image.rotate(angle, resample=Image.BILINEAR)
        return [image]


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic0, label0, pic1, label1):
        if isinstance(pic0, np.ndarray):
            # handle numpy array
            img0 = torch.from_numpy(pic0)
            img1 = torch.from_numpy(pic1)
        else:
            # handle PIL Image
            img0 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic0.tobytes()))
            img1 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic1.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic0.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic0.mode)
            img0 = img0.view(pic0.size[1], pic0.size[0], nchannel)
            img1 = img1.view(pic1.size[1], pic1.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img0 = img0.transpose(0, 1).transpose(0, 2).contiguous()
            img1 = img1.transpose(0, 1).transpose(0, 2).contiguous()
        img0 = img0.float().div(255)
        img1 = img1.float().div(255)

        
        if label0 is None:
            return img0, img1
        
        else:
            gt0 = torch.ByteTensor(torch.ByteStorage.from_buffer(label0.tobytes()))
            gt1 = torch.ByteTensor(torch.ByteStorage.from_buffer(label1.tobytes()))
            if label0.mode == 'YCbCr':
                nchannel=3
            else:
                nchannel = len(label0.mode)

            gt0 = gt0.view(label0.size[1], label0.size[0], nchannel)
            gt0 = gt0.transpose(0, 1).transpose(0, 2).contiguous()
            gt0 = gt0.float().div(255)
            gt1 = gt1.view(label1.size[1], label1.size[0], nchannel)
            gt1 = gt1.transpose(0, 1).transpose(0, 2).contiguous()
            gt1 = gt1.float().div(255)
            #return img, torch.LongTensor(np.array(label, dtype=np.int))
            return img0, gt0, img1, gt1

class ToTensor_One(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        
        return img,

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

"""
Original data transforms for [image, label] pair
"""

# class Resize(object):
#         def __init__(self, size):
#             self.size = size
#         def __call__(self, image, label):
#             return image.resize((self.size, self.size), Image.BILINEAR), \
#                     label.resize((self.size, self.size), Image.BILINEAR)

# class RandomCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size

#     def __call__(self, image, label, *args):
#         assert label is None or image.size == label.size, \
#             "image and label doesn't have the same size {} / {}".format(
#                 image.size, label.size)

#         w, h = image.size
#         tw, th = self.size
#         top = bottom = left = right = 0
#         if w < tw:
#             left = (tw - w) // 2
#             right = tw - w - left
#         if h < th:
#             top = (th - h) // 2
#             bottom = th - h - top
#         if left > 0 or right > 0 or top > 0 or bottom > 0:
#             label = pad_image(
#                 'constant', label, top, bottom, left, right, value=255)
#             image = pad_image(
#                 'reflection', image, top, bottom, left, right)
#         w, h = image.size
#         if w == tw and h == th:
#             return (image, label, *args)

#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         results = [image.crop((x1, y1, x1 + tw, y1 + th))]
#         if label is not None:
#             results.append(label.crop((x1, y1, x1 + tw, y1 + th)))
#         results.extend(args)
#         return results


# class RandomScale(object):
#     def __init__(self, scale):
#         if isinstance(scale, numbers.Number):
#             scale = [1 / scale, scale]
#         self.scale = scale

#     def __call__(self, image, label):
#         ratio = random.uniform(self.scale[0], self.scale[1])
#         w, h = image.size
#         tw = int(ratio * w)
#         th = int(ratio * h)
#         if ratio == 1:
#             return image, label
#         elif ratio < 1:
#             interpolation = Image.ANTIALIAS
#         else:
#             interpolation = Image.CUBIC
#         return image.resize((tw, th), interpolation), \
#                label.resize((tw, th), Image.NEAREST)


# class RandomRotate(object):
#     """Crops the given PIL.Image at a random location to have a region of
#     the given size. size can be a tuple (target_height, target_width)
#     or an integer, in which case the target will be of a square shape (size, size)
#     """

#     def __init__(self, angle):
#         self.angle = angle

#     def __call__(self, image, label=None, *args):
#         assert label is None or image.size == label.size

#         w, h = image.size
#         p = max((h, w))
#         angle = random.randint(0, self.angle * 2) - self.angle

#         if label is not None:
#             label = pad_image('constant', label, h, h, w, w, value=255)
#             label = label.rotate(angle, resample=Image.NEAREST)
#             label = label.crop((w, h, w + w, h + h))

#         image = pad_image('reflection', image, h, h, w, w)
#         image = image.rotate(angle, resample=Image.BILINEAR)
#         image = image.crop((w, h, w + w, h + h))
#         return image, label


# class RandomHorizontalFlip(object):
#     """Randomly horizontally flips the given PIL.Image with a probability of 0.5
#     """

#     def __call__(self, image, label):
#         if random.random() < 0.5:
#             results = [image.transpose(Image.FLIP_LEFT_RIGHT),
#                        label.transpose(Image.FLIP_LEFT_RIGHT)]
#         else:
#             results = [image, label]
#         return results


# class Normalize(object):
#     """Given mean: (R, G, B) and std: (R, G, B),
#     will normalize each channel of the torch.*Tensor, i.e.
#     channel = (channel - mean) / std
#     """

#     def __init__(self, mean, std):
#         self.mean = torch.FloatTensor(mean)
#         self.std = torch.FloatTensor(std)

#     def __call__(self, image, label=None):
#         for t, m, s in zip(image, self.mean, self.std):
#             t.sub_(m).div_(s)
#         if label is None:
#             return image,
#         else:
#             return image, label


# def pad_reflection(image, top, bottom, left, right):
#     if top == 0 and bottom == 0 and left == 0 and right == 0:
#         return image
#     h, w = image.shape[:2]
#     next_top = next_bottom = next_left = next_right = 0
#     if top > h - 1:
#         next_top = top - h + 1
#         top = h - 1
#     if bottom > h - 1:
#         next_bottom = bottom - h + 1
#         bottom = h - 1
#     if left > w - 1:
#         next_left = left - w + 1
#         left = w - 1
#     if right > w - 1:
#         next_right = right - w + 1
#         right = w - 1
#     new_shape = list(image.shape)
#     new_shape[0] += top + bottom
#     new_shape[1] += left + right
#     new_image = np.empty(new_shape, dtype=image.dtype)
#     new_image[top:top+h, left:left+w] = image
#     new_image[:top, left:left+w] = image[top:0:-1, :]
#     new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
#     new_image[:, :left] = new_image[:, left*2:left:-1]
#     new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
#     return pad_reflection(new_image, next_top, next_bottom,
#                           next_left, next_right)


# def pad_constant(image, top, bottom, left, right, value):
#     if top == 0 and bottom == 0 and left == 0 and right == 0:
#         return image
#     h, w = image.shape[:2]
#     new_shape = list(image.shape)
#     new_shape[0] += top + bottom
#     new_shape[1] += left + right
#     new_image = np.empty(new_shape, dtype=image.dtype)
#     new_image.fill(value)
#     new_image[top:top+h, left:left+w] = image
#     return new_image


# def pad_image(mode, image, top, bottom, left, right, value=0):
#     if mode == 'reflection':
#         return Image.fromarray(
#             pad_reflection(np.asarray(image), top, bottom, left, right))
#     elif mode == 'constant':
#         return Image.fromarray(
#             pad_constant(np.asarray(image), top, bottom, left, right, value))
#     else:
#         raise ValueError('Unknown mode {}'.format(mode))


# class Pad(object):
#     """Pads the given PIL.Image on all sides with the given "pad" value"""

#     def __init__(self, padding, fill=0):
#         assert isinstance(padding, numbers.Number)
#         assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
#                isinstance(fill, tuple)
#         self.padding = padding
#         self.fill = fill

#     def __call__(self, image, label=None, *args):
#         if label is not None:
#             label = pad_image(
#                 'constant', label,
#                 self.padding, self.padding, self.padding, self.padding,
#                 value=255)
#         if self.fill == -1:
#             image = pad_image(
#                 'reflection', image,
#                 self.padding, self.padding, self.padding, self.padding)
#         else:
#             image = pad_image(
#                 'constant', image,
#                 self.padding, self.padding, self.padding, self.padding,
#                 value=self.fill)
#         return (image, label, *args)


# class PadImage(object):
#     def __init__(self, padding, fill=0):
#         assert isinstance(padding, numbers.Number)
#         assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
#                isinstance(fill, tuple)
#         self.padding = padding
#         self.fill = fill

#     def __call__(self, image, label=None, *args):
#         if self.fill == -1:
#             image = pad_image(
#                 'reflection', image,
#                 self.padding, self.padding, self.padding, self.padding)
#         else:
#             image = ImageOps.expand(image, border=self.padding, fill=self.fill)
#         return (image, label, *args)


# class ToTensor(object):
#     """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#     """

#     def __call__(self, pic, label=None):
#         if isinstance(pic, np.ndarray):
#             # handle numpy array
#             img = torch.from_numpy(pic)
#         else:
#             # handle PIL Image
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#             if pic.mode == 'YCbCr':
#                 nchannel = 3
#             else:
#                 nchannel = len(pic.mode)
#             img = img.view(pic.size[1], pic.size[0], nchannel)
#             # put it from HWC to CHW format
#             # yikes, this transpose takes 80% of the loading time/CPU
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         img = img.float().div(255)
#         if label is None:
#             return img,
#         else:
#             gt = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
#             if label.mode == 'YCbCr':
#                 nchannel=3
#             else:
#                 nchannel = len(label.mode)
#             gt = gt.view(label.size[1], label.size[0], nchannel)
#             gt = gt.transpose(0, 1).transpose(0, 2).contiguous()
#             gt = gt.float().div(255)
#             #return img, torch.LongTensor(np.array(label, dtype=np.int))
#             return img, gt

# class Compose(object):
#     """Composes several transforms together.
#     """

#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, *args):
#         for t in self.transforms:
#             args = t(*args)
#         return args
