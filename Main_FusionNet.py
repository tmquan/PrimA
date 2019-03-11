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

    def size(self):
        return self._size

    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.rng.randint(0, len(self.imageFiles))
            image_p = cv2.imread(self.imageFiles[rand_index], cv2.IMREAD_GRAYSCALE)
            label_p = cv2.imread(self.labelFiles[rand_index], cv2.IMREAD_GRAYSCALE)

            image_p = cv2.resize(image_p, (int(image_p.shape[0]/2), int(image_p.shape[1]/2)), interpolation=cv2.INTER_NEAREST)
            label_p = cv2.resize(label_p, (int(label_p.shape[0]/2), int(label_p.shape[1]/2)), interpolation=cv2.INTER_NEAREST)
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
        x  = BilinearUpSample('upsample', x, 2)
        # x = Deconv2D('dec_o', x, chan, stride=2, activation=tf.identity) 
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
            # e0 = Dropout('f0', e0, 0.5)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            # e1 = Dropout('f1', e1, 0.5)
            e2 = residual_enc('e2',  e1, nb_filters*4)
            # e2 = Dropout('f2', e2, 0.5)
            e3 = residual_enc('e3',  e2, nb_filters*8)
            # e3 = Dropout('f3', e3, 0.5)
            return e3, [e2, e1, e0]
           

def arch_fusionnet_decoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e2, e1, e0 = feats 
            d4 = img 
            # d4 = Dropout('r4', d4, 0.5)

            d3 = residual_dec('d3', d4, nb_filters*4)
            # d3 = Dropout('r3', d3, 0.5)
            d3 = d3+e2 if e2 is not None else d3 
            
            d2 = residual_dec('d2', d3, nb_filters*2)
            # d2 = Dropout('r2', d2, 0.5)
            d2 = d2+e1 if e1 is not None else d2 
            
            d1 = residual_dec('d1', d2, nb_filters*1)
            # d1 = Dropout('r1', d1, 0.5)
            d1 = d1+e0 if e0 is not None else d1  
            
            d0 = residual_dec('d0', d1, nb_filters*1) 
            # d0 = Dropout('r0', d0, 0.5)
            
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
       

def sample(dataDir, model_path):
    pass


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
        pass

    else:
        # Set up configuration
        train_dset = get_data(args.data, isTrain=True, isValid=False, isTest=False, shape=[args.dimz, args.dimy, args.dimx])
        valid_dset = get_data(args.data, isTrain=False, isValid=True, isTest=False, shape=[args.dimz, args.dimy, args.dimx])
        # test_dset  = get_data(args.data, isTrain=False, isValid=False, isTest=True)

        augs = [imgaug.RandomCrop(512)]
        train_dset = AugmentImageComponents(train_dset, augs, (0, 1)) # Random crop both channel
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
                PeriodicTrigger(ModelSaver(), every_k_epochs=200),
                # PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
                ],
            max_epoch       =   1000, 
            session_init    =   SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())