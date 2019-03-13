# System modules
import os
import sys
import glob
import argparse
import numpy as np
from natsort import natsorted

# Image processing, vision models
import cv2

# Machine learning modules
import tensorflow as tf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.utils.argtools import memoized
from tensorpack import (TowerTrainer,
                        ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import freeze_variables
# Customize required modules to a specific model
# from Dataflows import *
# from Losses import *
# from Models import *

# class ImageDataFlow(RNGDataFlow):
#   pass

# class ProcessingDataFlow(DataFlow):
#   def __init__(self, ds):
#       self.ds = ds

#   def reset_state(self):
#       self.ds.reset_state()

#   def __iter__(self):
#       for datapoint in self.ds:
#           # do something
#           yield new_datapoint

# def get_data(dataDir):
#     images = glob.glob(os.path.join(dataDir, '*.png'))
#     ds = ImageFromFile(imgs, channel=3, shuffle=True)
#     ds = AugmentImageComponent(ds, [imgaug.Resize((256, 256))])
#     ds = MapData(ds, lambda dp: [cv2.cvtColor(dp[0], cv2.COLOR_RGB2Lab)[:,:,0], dp[0]])
#     ds = BatchData(ds, 32)
#     ds = PrefetchData(ds, 4) # use queue size 4
#     ds = PrintData(ds, num=2) # only for debug
#     return ds

class ImageDataFlow(RNGDataFlow):
    def __init__(self, 
        imageDir, 
        labelDir, 
        size, 
        dtype='float32', 
        isTrain=False, 
        isValid=False, 
        isTest=False, 
        shape=[1, 512, 512]):

        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        self.imageFiles = natsorted (glob.glob(self.imageDir + '/*.*'))
        self.labelFiles = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(self.imageFiles)
        print(self.labelFiles)
      
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]

        self.images = []
        self.labels = []

        for imageFile in self.imageFiles:
            image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
            image = image[::2, ::2]
            self.images.append(image)
        for labelFile in self.labelFiles:
            label = cv2.imread(labelFile, cv2.IMREAD_GRAYSCALE)
            label = label[::2, ::2]
            self.labels.append(label)

    def size(self):
        return self._size

    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.rng.randint(0, len(self.imageFiles))
            # image_p = cv2.imread(self.imageFiles[rand_index], cv2.IMREAD_GRAYSCALE)
            # label_p = cv2.imread(self.labelFiles[rand_index], cv2.IMREAD_GRAYSCALE)
            image_p = self.images[rand_index].copy()
            label_p = self.labels[rand_index].copy()

            while (image_p.shape[0] <= 512 or image_p.shape[1] <= 512) or \
                (label_p.shape[0] <= 512 or label_p.shape[1] <= 512) or \
                image_p is None or label_p is None:
                rand_index = self.rng.randint(0, len(self.imageFiles))
                image_p = self.images[rand_index].copy()
                label_p = self.labels[rand_index].copy()

            if self.isTrain:
                dimy, dimx = image_p.shape
                
                randy = self.rng.randint(0, dimy-512+1)
                randx = self.rng.randint(0, dimx-512+1)
                image_p = image_p[randy:randy+512,randx:randx+512]
                label_p = label_p[randy:randy+512,randx:randx+512]

                image_p = cv2.resize(image_p, (512, 512), cv2.INTER_NEAREST)
                label_p = cv2.resize(label_p, (512, 512), cv2.INTER_NEAREST)
            # while (image_p.shape[0] <= 512 or image_p.shape[1] <= 512) or \
            #     (label_p.shape[0] <= 512 or label_p.shape[1] <= 512) or \
            #     image_p is None or label_p is None:
            #     rand_index = self.rng.randint(0, len(self.imageFiles))
            #     image_p = self.images[rand_index].copy()
            #     label_p = self.labels[rand_index].copy()
            # image_p = cv2.resize(image_p, (int(image_p.shape[0]/2), int(image_p.shape[1]/2)), interpolation=cv2.INTER_NEAREST)
            # label_p = cv2.resize(label_p, (int(label_p.shape[0]/2), int(label_p.shape[1]/2)), interpolation=cv2.INTER_NEAREST)
            # image_p = np.expand_dims(image_p, axis=-1)
            # label_p = np.expand_dims(label_p, axis=-1)
            # image_p = np.expand_dims(image_p, axis=0)
            # label_p = np.expand_dims(label_p, axis=0)                        
                yield [image_p.astype(np.float32), label_p.astype(np.float32)] 

def get_data(dataDir, isTrain=False, isValid=False, isTest=False, shape=[1, 512, 512]):
    # Process the directories 
    if isTrain:
        num=500
        names = ['trainA', 'trainB']
    if isValid:
        num=100
        names = ['trainA', 'trainB']
    if isTest:
        num=10
        names = ['validA', 'validB']

    dset = ImageDataFlow(imageDir=os.path.join(dataDir, names[0]),
                         labelDir=os.path.join(dataDir, names[1]),
                         size=num, 
                         isTrain=isTrain, 
                         isValid=isValid, 
                         isTest =isTest, 
                         shape=shape)
    return dset

def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, name=name)
    
def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, name=name)

###############################################################################
#                                   FusionNet 2D
###############################################################################
@layer_register(log_shape=True)
def residual(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        inputs = x
        x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4, activation=tf.identity)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8) 
        x = InstanceNorm('inorm', x) + inputs
        return x


@layer_register(log_shape=True)
def residual_enc(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv2D('enc_i', x, chan, stride=2) 
        x = residual('enc_r', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv2D('enc_o', x, chan, stride=1) #, activation=tf.identity) 
        #x = InstanceNorm('enc_n', x)
        return x


@layer_register(log_shape=True)
def residual_dec(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x  = Deconv2D('dec_i', x, chan, stride=1) 
        x  = residual('dec_r', x, chan, first=True)
        # x  = BilinearUpSample('upsample', x, 2)
        x = Deconv2D('dec_o', x, chan, stride=2, activation=tf.identity) 
        # x1 = Deconv2D('dec_o', x, chan, stride=2, activation=tf.identity) 
        # x2 = BilinearUpSample('upsample', x, 2)
        # x  = InstanceNorm('dec_n', (x1+x2)/2.0)
        x  = InstanceNorm('dec_n', x)
        return x

###############################################################################
def arch_fusionnet_encoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc('e0', img, nb_filters*1)
            e0 = Dropout('f0', e0, 0.5)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            e1 = Dropout('f1', e1, 0.5)
            e2 = residual_enc('e2',  e1, nb_filters*4)
            e2 = Dropout('f2', e2, 0.5)
            e3 = residual_enc('e3',  e2, nb_filters*8)
            e3 = Dropout('f3', e3, 0.5)
            return e3, [e2, e1, e0]
           

def arch_fusionnet_decoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e2, e1, e0 = feats 
            d4 = img 
            d4 = Dropout('r4', d4, 0.5)

            d3 = residual_dec('d3', d4, nb_filters*4)
            d3 = Dropout('r3', d3, 0.5)
            d3 = d3+e2 if e2 is not None else d3 
            
            d2 = residual_dec('d2', d3, nb_filters*2)
            d2 = Dropout('r2', d2, 0.5)
            d2 = d2+e1 if e1 is not None else d2 
            
            d1 = residual_dec('d1', d2, nb_filters*1)
            d1 = Dropout('r1', d1, 0.5)
            d1 = d1+e0 if e0 is not None else d1  
            
            d0 = residual_dec('d0', d1, nb_filters*1) 
            d0 = Dropout('r0', d0, 0.5)
            
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                                     [7//2,7//2], 
                                                                     [7//2,7//2], 
                                                                     [0,0]])
            dd = Conv2D('convlast', dp, last_dim, kernel_shape=7, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd, [d1, d2, d3]
###############################################################################
def arch_fusionnet_translator_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    enc, feat_enc = arch_fusionnet_encoder_2d(img, feats, last_dim, nl, nb_filters)
    dec, feat_dec = arch_fusionnet_decoder_2d(enc, feat_enc, last_dim, nl, nb_filters)
    return dec

def arch_fusionnet_classifier_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
        lin = (LinearWrap(img)
                 .Conv2D('conv0', nb_filters, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', nb_filters * 2)
                 .Conv2D('conv2', nb_filters * 4)
                 .Conv2D('conv3', nb_filters * 8, strides=1)
                 .Conv2D('conv4', 1, strides=1, activation=tf.identity, use_bias=True)())
        return lin

###############################################################################


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, args.dimy, args.dimx), name='image'), \
                tf.placeholder(tf.float32, (None, args.dimy, args.dimx), name='label')]

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

    def build_graph(self, image, label):
        NF = 64
        logit = label / 20.0 # 0~10
        image = (image / 128.0) - 1.0
        label = (label / 128.0) - 1.0

        image = tf.expand_dims(image, axis=-1)
        # logit = tf.expand_dims(logit, axis=-1)
        label = tf.expand_dims(label, axis=-1)
        
        # Network definition
        # with argscope(Conv2D, kernel_shape=4, stride=2,
        #               nl=lambda x, name: tf.nn.leaky_relu(BatchNorm('bn', x), name=name)):
        #     # encoder
        #     e1 = Conv2D('conv1', image, NF, nl=tf.nn.leaky_relu)
        #     e2 = Conv2D('conv2', e1, NF * 2)
        #     e3 = Conv2D('conv3', e2, NF * 4)
        #     e4 = Conv2D('conv4', e3, NF * 8)
        #     e5 = Conv2D('conv5', e4, NF * 8)
        #     e6 = Conv2D('conv6', e5, NF * 8)
        #     e7 = Conv2D('conv7', e6, NF * 8)
        #     e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
        # with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
        #     # decoder
        #     e8 = Deconv2D('deconv1', e8, NF * 8)
        #     e8 = Dropout(e8)
        #     e8 = tf.concat([e8, e7], axis=3)

        #     e7 = Deconv2D('deconv2', e8, NF * 8)
        #     e7 = Dropout(e7)
        #     e7 = tf.concat([e7, e6], axis=3)

        #     e6 = Deconv2D('deconv3', e7, NF * 8)
        #     e6 = Dropout(e6)
        #     e6 = tf.concat([e6, e5], axis=3)

        #     e5 = Deconv2D('deconv4', e6, NF * 8)
        #     e5 = Dropout(e5)
        #     e5 = tf.concat([e5, e4], axis=3)

        #     e4 = Deconv2D('deconv5', e5, NF * 4)
        #     e4 = Dropout(e4)
        #     e4 = tf.concat([e4, e3], axis=3)

        #     e3 = Deconv2D('deconv6', e4, NF * 2)
        #     e3 = Dropout(e3)
        #     e3 = tf.concat([e3, e2], axis=3)

        #     e2 = Deconv2D('deconv7', e3, NF * 1)
        #     e2 = Dropout(e2)
        #     e2 = tf.concat([e2, e1], axis=3)

        #     estim = Deconv2D('estim', e2, 11, nl=tf.nn.sigmoid)
        estim = arch_fusionnet_translator_2d(image, last_dim=11, nl=tf.nn.sigmoid, nb_filters=64)

        # Loss define here
        sfm = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=estim, labels=tf.cast(logit, tf.int32)), name='sfm')
        add_moving_summary(sfm)

        estim = tf.argmax(estim, axis=-1)
        estim = tf.expand_dims(estim, axis=-1)
        estim = tf.cast(estim, tf.float32)
        estim = estim * 20
        estim = tf.identity(estim, name='estim')
        estim = (estim / 128.0) - 1.0
        mae = tf.reduce_mean(tf.abs(estim - label), name='mae')
        add_moving_summary(mae)

        cost = sfm + mae

        # Visualization here
        viz = tf.concat([image, estim, label], axis=2)
        viz = (viz + 1.0) * 128.0
        tf.summary.image('colorized', viz, max_outputs=10)

        return cost

class OnlineExport(Callback):
    def __init__(self):
        pass

    def _setup_graph(self):
        pass

    def _trigger_epoch(self):
       pass
       

'''
MIT License
Copyright (c) 2018 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.
    Parameters
    ----------
    x : ndarray
        Array to create sliding window views.
    shape : sequence of int
        The shape of the window. Must have same length as number of input array dimensions.
    step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).
    Returns
    -------
    view : ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1
    See also
    --------
    as_strided: Create a view into the array with the given shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.
    For some cases, there may be more efficient approaches, such as FFT based algo discussed in #7753.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> step = (1,2)
    >>> sliding_window_view(x, shape, step)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[12, 13],
             [22, 23]]]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, np.int)
    except:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape)!= len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape)  - shape) // step + 1 # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    #view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok)#, writeable=writeable)

    if writeable:
        return view.copy()
    else:
        return view

def sample(dataDir, model_path):
    print("Starting...")
    print(dataDir)
    imageFiles = glob.glob(os.path.join(dataDir, '*.png'))
    print(imageFiles)

    # Load the model 
    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['mul:0']))

    for imageFile in imageFiles:
        head, tail = os.path.split(imageFile)
        print(tail)
        estimFile = tail
        print(estimFile)

        # Read the image file
        # image = skimage.io.imread(imageFile)
        image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
        image = image[::2, ::2]
        
        def getOptimalDeploySize(image, fov):
            old_shape = image.shape[:2]
            new_shape = (int(np.ceil(old_shape[0]/fov)*fov), 
                         int(np.ceil(old_shape[1]/fov)*fov))
            resized = cv2.resize(image, new_shape[::-1]) # x then y
            print(old_shape, new_shape)
            print(image.shape)
            return image, old_shape, resized, new_shape

        _, old_shape, image, new_shape = getOptimalDeploySize(image, 256)

        
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        print(image.shape)

        def weighted_map_blocks(arr, inner, outer, ghost, func=None): # work for 3D, inner=[1, 3, 3], ghost=[0, 2, 2], 
            dtype = np.float32 #arr.dtype

            arr = arr.astype(np.float32)
            # param
            if outer==None:
                outer = inner + 2*ghost
                outer = [(i + 2*g) for i, g in zip(inner, ghost)]
            shape = outer
            steps = inner
                
            print(outer)
            print(shape)
            print(inner)
            
            padding=arr.copy()
            print(padding.shape)
            #print(padding)
            
            weights = np.zeros_like(padding)
            results = np.zeros_like(padding)
            
            v_padding = sliding_window_view(padding, shape, steps)
            v_weights = sliding_window_view(weights, shape, steps)
            v_results = sliding_window_view(results, shape, steps)
            
            print('v_padding', v_padding.shape)


            #for z in range(v_padding.shape[0]):
            for y in range(v_padding.shape[1]):
                for x in range(v_padding.shape[2]):
    
                    v_result = np.array(func(
                                            (v_padding[0,y,x,0][...,0]) ) ) ### Todo function is here
                    v_result = np.squeeze(v_result, axis=0).astype(np.float32)
    
                    yy, xx = np.meshgrid(np.linspace(-1,1,shape[1], dtype=np.float32), 
                                         np.linspace(-1,1,shape[2], dtype=np.float32))
                    d = np.sqrt(xx*xx+yy*yy)
                    sigma, mu = 0.5, 0.0
                    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
                    v_weight = v_weight/v_weight.max()
                    #print(v_weight.shape)
                    #v_weight.tofile('gaussian_map.npy')
                    
                    v_weight = np.expand_dims(v_weight, axis=-1)
                    #print(shape)
                    #print(v_weight.shape)
                    v_weights[0,y,x] += v_weight

                    v_results[0,y,x] += v_result * v_weight
                        
            # Divided by the weight param
            results /= weights 
            
            
            return results.astype(dtype)
    
        if image.shape != [1, 512, 512, 1]:
            # Clean
            estim = weighted_map_blocks(image, inner=[1, 256, 256, 1], 
                                               outer=[1, 512, 512, 1], 
                                               ghost=[1, 256, 256, 0], 
                                               func=predict_func) # inner,  ghost

        estim = np.squeeze(estim)
        # Resize to old shape
        estim = cv2.resize(estim, old_shape[::-1], cv2.INTER_NEAREST)
        cv2.imwrite(estimFile, estim)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parsing the argument
    parser.add_argument('--gpu', help='comma seperated list of GPU(s) to use.',
                                 default='0')
    parser.add_argument('--data',  default='data/', required=True, 
                                   help='Data directory, contain trainA/trainB/validA/validB for training')
    parser.add_argument('--load',  help='Load the model path')
    parser.add_argument('--dimx',  type=int, default=512)
    parser.add_argument('--dimy',  type=int, default=512)
    parser.add_argument('--dimz',  type=int, default=1)
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')

    global args
    args = parser.parse_args()


    
    # Configuration for training and testing in here
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        print("Deploy the data")
        sample(args.data, args.load)
        pass

    else:
        # Set up configuration
        train_dset = get_data(args.data, isTrain=True, isValid=False, isTest=False, shape=[args.dimz, args.dimy, args.dimx])
        valid_dset = get_data(args.data, isTrain=False, isValid=True, isTest=False, shape=[args.dimz, args.dimy, args.dimx])
        # test_dset  = get_data(args.data, isTrain=False, isValid=False, isTest=True)

        # augs = [imgaug.RandomCrop((512, 512)), imgaug.Resize((512, 512), cv2.INTER_NEAREST), ]
        # augs = [imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.00, 1.0), 
        #    aspect_ratio_range=(0.9, 1.2), 
        #    target_shape=512, interp=cv2.INTER_NEAREST)]
        augs = []
        # train_dset = AugmentImageComponents(train_dset, augs, (0, 1)) # Random crop both channel
        # train_dset  = PrefetchDataZMQ(train_dset, 4)
        train_dset = BatchData(train_dset, 4)
        train_dset = PrefetchData(train_dset, nr_proc=2, nr_prefetch=50)
        train_dset  = PrintData(train_dset)

        # Set the logger directory
        logger.auto_set_dir()

        # Create a model
        model = Model()

        # Set up configuration
        config = TrainConfig(
            model           =   model, 
            dataflow        =   train_dset,
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=50),
                # PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
                ],
            max_epoch       =   1000, 
            session_init    =   SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())