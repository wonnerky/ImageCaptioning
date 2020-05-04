import tensorflow as tf
import os

annotation_folder = 'F:\ml\MSCOCO\\annotations'
if not os.path.exists(annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir='F:\ml\MSCOCO',
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '\\annotations\captions_train2014.json'
    os.remove(annotation_zip)

image_folder = 'F:\ml\MSCOCO\\train2014'
if not os.path.exists(image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir='F:\ml\MSCOCO',
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    PATH = os.path.dirname(image_zip) + '\train2014'
    os.remove(image_zip)
else:
    PATH = 'F:\ml\MSCOCO\\train2014'

