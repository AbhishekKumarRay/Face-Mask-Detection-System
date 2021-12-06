# Face-Mask-Detection-System

<h2>Abstract</h2>
As we all know, Covid-19 pandemic is one of the most life-changing event ever happened in the history of mankind. To recover losses caused by this global pandemic, we have come across various stages of reopening of all public places and institutions. In this whole evolution, Face masks have become a vital element of our daily lives.
<br>Considering the importance of the Face Mask during the pandemic time, when the COVID-19 cases were increasing rapidly, I decided to make this project. This project was developed to monitor whether people are wearing face mask and following the basic safety principle or not. It aims to identify and mark down such individual who are not wearing a mask and alert the person sitting behind the camera. It can also easily filter out the individual who are without a face mask from a group of people. It could be implemented with live video surveillance camera to monitor the activity as well as on the stored image. The main motive was to implement it at densely populated areas where the chances of spread of COVID-19 cases were relatively higher.
<br>

<h2>One time Setup:</h2>

1.	Download and save it

        https://github.com/tensorflow/models
  
 <br>

2. Next open anaconda (I am using anaconda as it provides a one stop place)
We can use anaconda directly but it is advised to create a virtual environment and use it.

create virtual environment named "object" and then activate it

    conda create -n object

    conda activate object

Verify version

    python --version


To deactivate a environment 

    conda deactivate 

To delete a environment  

    conda env remove -n object

To delete cache

    conda clean --all
<br>

3.	Install python for this virtual environment
    
        conda install python=3.7

  Verify version
    
    python --version
<br>

4.   Download Protocol Buffer this is used for string parsing, etc and it is one of most important step.
Download this github (Scroll to end)  and download your specific windows or linux version for windows download  <b>protoc-3.4.0-win32.zip:</b>

    https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0

Extract this and then goto bin folder then copy the protoc.exe and paste it into step 1 downloaded <b>models/research  </b>folder
<br><br>

5.   Change Forlder and move Inside The tensor flow model downloaded in step 1

    cd Folder_name/models/research/
<br>

6.    Compile the protoc downloaded in Step 4

    protoc object_detection/protos/*.proto --python_out=.
<br>

7.    Before proceeding,

    pip install TensorFlow==1.15    

I am downloading tensorflow 1.15 because we need any tensorflow 1.x.x
 On Windows, you must have the Visual C++ 2015 build tools on your path. If you don't, make sure to install them from <a href='https://go.microsoft.com/fwlink/?LinkId=691126'>here.</a>  
<br>

8.    Run to build and install all required packages 
Copy  setup.py from  <b>models\research\object_detection\packages\tf2 and paste it into  models\research </b> 

    python -m pip install .

<h3>Training Steps</h3>

<b>1.	Data Acquiring:</b><br>
Collect data from any open source dataset repositories like Kaggle or Robocon

a.	Here I have created my own dataset by clicking images or downloading them individually from google.

b.	If you have created your custom data we need to make some changes accordingly. (Label Images)

To label image I am using <a href='https://github.com/tzutalin/labelImg'>LabelImage</a> but we have other alternative tools too <a href='http://www.robots.ox.ac.uk/~vgg/software/via/'>(VGG Image Annotation Tool</a> and <a href='https://github.com/microsoft/VoTT'>VoTT (Visual Object Tagging Tool).</a>
INSTALL Labelimage and open it from the instructions given in readme.
<b>Labelimage saves the file in XML Format</b>

<b>2.	Convert Xml file into CSV.</b><br>

Divide all the images into two parts (Test and Train). Keep the image in 80 : 20 Ratio and 
       move 80% Images into <b>Train Folder</b> whereas 20% Into <b>Test Folder.</b>

Copy the code and paste into a python file (Place the file in same location where training images is stored) and run it.  Change the path highlighted accordingly.
  
    python xml_to_csv.py
<br>

    import os
    import glob
    import pandas as pd
    import xml.etree.ElementTree as ET

    def xml_to_csv(path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df

    def main():
        for folder in ['train', 'test']:
            image_path = os.path.join(os.getcwd(), ('training/images/' + folder))
            xml_df = xml_to_csv(image_path)
            xml_df.to_csv(('training/'+folder+'_labels.csv'), index=None)
        print('Successfully converted xml to csv.')

    main()

<b>3.	Convert CSV Single file into TFRecord</b><br>

2 csv files (Train & Test) would have been created after executing the above codes.
To convert the CSV file into record file create a new python file and run the following Code.
Change the highlighted section as per the labels used while labelling the images.
  
    from __future__ import division
    from __future__ import print_function
    from __future__ import absolute_import

    import os
    import io
    import pandas as pd

    from tensorflow.python.framework.versions import VERSION
    if VERSION >= "2.0.0a0":
        import tensorflow.compat.v1 as tf
    else:
        import tensorflow as tf

    from PIL import Image
    from object_detection.utils import dataset_util
    from collections import namedtuple, OrderedDict

    flags = tf.app.flags
    flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
    flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
    flags.DEFINE_string('image_dir', '', 'Path to images')
    FLAGS = flags.FLAGS


    def class_text_to_int(row_label):
        if row_label == 'with_mask':
            return 1
        elif row_label == 'without_mask':
            return 2
        elif row_label == 'mask_weared_incorrect':
            return 3
        else:
            return None


    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

      def create_tf_example(group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def main(_):
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
        path = os.path.join(FLAGS.image_dir)
        examples = pd.read_csv(FLAGS.csv_input)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), FLAGS.output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))

    if __name__ == '__main__':
        tf.app.run()


 
<b>Run the following commands to execute above code</b>

    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

    python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
<br>

<b>NOW WE ARE READY FOR TRAINING</b>

Create a lable_map.pbtxt file

    item {
        id: 1
        name: 'with_mask'
    }

    item {
        id: 2
        name: 'without_mask'
    }

    item {
        id: 3
        name: 'mask_weared_incorrect'
    }


Goto models/ research/object_detection/configs/tf2

You can use any of the model configuration file. I will be using Efficientdet but there are other more available models in <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md'>TensorFlow 2 Object Detection model zoo.</a>
Download the model and store it from <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md'>TensorFlow 2 Object Detection model zoo.</a>

Open the configuration file and change following details:
1.	Change Num_classes = 3 as I have used 3 labels.
2.	Change fine_tune_checkpoint to the path of downloades_model/checkpoint/ckpt-0
3.	Change fine_tune_checkpoint_type to detection
4.	Change batch size to 8 or 12 or 16 or any number depending on you system capability.
5.	Change input_path of the train_input_reader to the path of the train.record file
6.	Change input_path of the eval_input_reader to the path of the test.record
7.	Change label_map_path of train_input as well as eval_input to the path of the label map.

Now we can run the training command inside model/research/object_detection

    python model_main_tf2.py \ --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \ --model_dir=training \ --alsologtostderr

TO VIEW STATUS OF TRAINING OPEN NEW TERMINAL AND RUN

    tensorboard --logdir=training/train

When the loss is significant enough low we can stop the training

<b>Export inference graph</b>

    python exporter_main_v2.py \ --trained_checkpoint_dir=training \ --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \ --output_directory inference_graph

<b>RUN WEBCAMERA VIEW OR IMAGE FILE</b>

    import os

    MODELS_DIR = "inference_graph"

    PATH_TO_CKPT = os.path.join('inference_graph', 'checkpoint/')
    PATH_TO_CFG = os.path.join( 'inference_graph','pipeline.config')

    PATH_TO_LABELS = "label_map.pbtxt"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import config_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder

    tf.get_logger().setLevel('ERROR')           

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    import cv2

    cap = cv2.VideoCapture(0)

    import numpy as np

    while True:
    
    ret, image_np = cap.read()

    
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        cap.release()
        cv2.destroyAllWindows()


