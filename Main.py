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
from tensorpack import * 

# Customize required modules to a specific model
# from Dataflows import *
# from Losses import *
# from Models import *
class ImageDataFlow(RNGDataFlow):
	pass

class Model(ModelDesc):
	def inputs(self):
        return [tf.placeholder(tf.float32, (None, args.dimy, args.dimx, 3), 'image'),
                tf.placeholder(tf.float32, (None, args.dimy, args.dimx, 1), 'label')]

    def build_graph(self, image, label):
        self.cost = 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Parsing the argument
	parser.add_argument('--gpu', help='comma seperated list of GPU(s) to use.',
								 default='0')
	parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
                                   help='Data directory, contain trainA/trainB/validA/validB for training')
    parser.add_argument('--load',  help='Load the model path')
    parser.add_argument('--dimx',  type=int, default=512)
    parser.add_argument('--dimy',  type=int, default=512)
    parser.add_argument('--dimz',  type=int, default=1)
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')

    global args
    args = parser.parse_args()