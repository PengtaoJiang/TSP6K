#!/usr/bin/python
#
# Cityscapes labels
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!
    'drawId'     , 
    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #   name,                 id    trainId   drawId  category       catId  hasInstances      ignoreInEval   color
    Label('unlabeled',         0 ,      255 , 21,     'void'       , 0       , False        , True          , ( 0,  0, 0) ),
    Label('road',              1 ,        0 ,  1,     'flat'        , 1       , False        , False        , (128, 64,128) ),
    Label('sidewalk',          2 ,        1 ,  2,     'flat'        , 1       , False        , False        , (244, 35,232) ),
    Label('building',         3 ,        2 ,  3,    'construction' , 2       , False        , False        , ( 70, 70, 70) ),
    Label('wall',             4 ,        3 ,  3,    'construction' , 2       , False        , False        , ( 80, 90, 40) ),
    Label('railing',          5 ,        4 ,  9,    'construction' , 2       , False        , False         , (180,165,180) ),
    Label('vegetation',       6 ,        5 ,  10,    'nature'         , 3       , False        , False        , (107,142, 35) ),
    Label('terrain',          7 ,        6 ,   4,    'nature'         , 3       , False        , False        , (152,251,152) ),
    Label('sky',              8 ,        7 ,    5,   'sky'            , 4       , False        , False        , ( 70,130,180) ),
    Label('person',           9 ,        8 ,   11,   'human'          , 5       , True         , False        , (255,  0,  0) ),
    Label('rider',            10 ,       9 ,   12,   'human'          , 5       , True         , False        , (255, 100, 0) ),
    # Label(  '人组'                , 'person group',     11 ,        10 , 20,    'human'          , 7       , False        , False        , (220, 90,  120) ),
    Label('car',             11 ,       10 ,   13,    'vehicle'       , 6       , True         , False        , (  0,  0,142) ),
    Label('truck',          12 ,        11 ,  14,     'vehicle'       , 6       , True         , False        , (  0,  0, 70) ),
    Label('bus',             13 ,        12 , 15,    'vehicle'         , 6       , True         , False        , (  0, 60,100) ),
    Label('motorcycle',      14 ,       13 ,  16,     'vehicle'        , 6       , True         , False        , (  0,  0,230) ),
    Label('bicycle',         15 ,       14 ,  17,    'vehicle'         , 6       , True         , False        , (119, 11, 32) ),
    # Label(  '车组'                , 'car group',        17 ,       16 , 20,       'vehicle'      , 7       , False        , False        , (180,0,  80) ),
    Label('indication line', 16 ,       15 , 6, 'sign'            , 7       , False        , False        , (250,170,160) ),
    Label('lane line',       17 ,       16 ,  7, 'sign'            , 7       , False        , False        , (250,200,160) ),
    Label('crosswalk',       18 ,       17 ,  8, 'sign'            , 7       , False        , False        , (250,240,180) ),
    Label('pole',            19 ,       18 , 18, 'sign'            , 7       , False        , False        , (153,153,153) ),
    Label('traffic light',   20 ,       19 ,  19, 'sign'            , 7       , False        , False        , (250,170, 30) ),
    Label('traffic sign',    21 ,       20 , 20, 'sign'            , 7       , False        , False         , (220,220,  0) ),
    # Label(  '车牌'                , 'license plate',    24 ,       23,  21, 'vehicle'         , 6       , False        , True         , (255, 255,  255) ),
    # Label(  '时间'                , 'time',             24 ,      23  ,  21, 'time'          , 8       , False           , True         , (111, 100,  100) ),
]
    
#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
