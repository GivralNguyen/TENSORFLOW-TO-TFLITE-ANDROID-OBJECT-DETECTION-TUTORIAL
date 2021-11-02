---


---

<h1 id="tensorflow-to-tflite-android-object-detection-tutorial">TENSORFLOW TO TFLITE ANDROID OBJECT DETECTION TUTORIAL</h1>
<p>A guide showing how to convert Tensorflow Frozen Graph to TensorFlow Lite object detection models and run them on Android.<br>
<img src="https://i.ibb.co/fFV32fW/tfliteresult.png" alt="Android TFlite detection result "></p>
<h2 id="step-1-install-tensorflow">Step 1: Install Tensorflow</h2>
<ul>
<li>The easiest way to install Tensorflow without using Docker is through Anaconda. First create a Anaconda Environment with Tensorflow-gpu. I tested with TF-gpu 2.3 and Python 3.6.</li>
<li>First create and activate the Anaconda Environment:</li>
</ul>
<pre><code>conda create -n tf2.3 python=3.6
</code></pre>
<pre><code>conda activate tf2.3
</code></pre>
<ul>
<li>Install CUDA and CUDNN. I find that this <a href="https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0">tutorial</a> was the easiest to implement. I used Cuda 10.1 and CUDNN 7.</li>
<li>Install tensorflow-gpu using</li>
</ul>
<pre><code>pip install tensorflow-gpu==2.3
</code></pre>
<h2 id="step-2-export-frozen-inference-graph-for-tensorflow-lite">Step 2: Export frozen inference graph for TensorFlow Lite</h2>
<p>Download and unzip ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz. This is a quantized int8 version of ssd_mobilenet_v2.</p>
<pre><code>wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz 
</code></pre>
<pre><code>tar xzvf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
</code></pre>
<p>Then clone the Tensorflow model repository:</p>
<pre><code>git clone https://github.com/tensorflow/models
</code></pre>
<p>Afterwards, the model can be exported for conversion to TensorFlow Lite using the export_tflite_ssd_graph.py script. First, create a folder in \object_detection called “TFLite_model” by issuing:</p>
<pre><code>mkdir TFLite_model
</code></pre>
<p>We will  start by creating a TensorFlow frozen graph with compatible ops that can be used with TensorFlow lite. This is done by running the command below from the <em>object_detection</em> folder. (Note, the XXXX in the second command should be replaced with the highest-numbered model.ckpt file in the \object_detection\training folder.)</p>
<pre><code>python export_tflite_ssd_graph.py \   
--pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config \   
--trained_checkpoint_prefix=training/model.ckpt-XXXX \   
--output_directory=tflite \  
--add_postprocessing_op=true
</code></pre>
<p>After the command has executed, there should be two new files in the \object_detection\tflite folder: tflite_graph.pb and tflite_graph.pbtxt.</p>
<h2 id="step-3-convert-to-tflite">Step 3: Convert to TFlite</h2>
<p>Next we’ll use TensorFlow Lite to get the optimized model by using <a href="https://www.tensorflow.org/lite/convert">TfLite Converter</a>, the TensorFlow Lite Optimizing Converter. This will convert the resulting frozen graph (tflite_graph.pb) to the TensorFlow Lite flatbuffer format (detect.tflite) via the following command. For a quantized model, run this from the tflite/ directory:</p>
<pre><code>tflite_convert --graph_def_file=tflite/tflite_graph.pb 
--output_file tflite/detect.tflite  
--output_format=TFLITE 
--input_shapes=1,300,300,3 
--input_arrays=normalized_input_image_tensor 
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  
--inference_type=QUANTIZED_UINT8 
--mean_values=128 
--std_dev_values=127 
--change_concat_input_ranges=false 
--allow_custom_ops 
--enable_v1_converter
</code></pre>
<p>This create a file called <em>detect.tflite</em> in the tflite folder. Now before you can run it using the Object detection <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">android example</a> by Tensorflow, you need to add metadata to the model .Thankfully, the process is quite simple using <a href="https://www.tensorflow.org/lite/convert/metadata_writer_tutorial">TensorFlow Lite Metadata Writer API</a>. Follow the tutorial for <a href="https://www.tensorflow.org/lite/convert/metadata_writer_tutorial#object_detectors">Object Detectors</a>. Change the _MODEL_PATH and _LABEL_FILE to the tflite model and its label path.</p>
<p><img src="https://i.ibb.co/Gp7q0Lz/metadata.png" alt="Metadata">Metadata of our model</p>
<h2 id="step-4--running-android-object-detection-example.">Step 4:  Running Android Object Detection Example.</h2>
<ul>
<li>To test this file in an android app, start by downloading and running the Object detection <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">android example</a> by Tensorflow. Or you can simply use the app i provided in this repository,</li>
</ul>
<p>. When you are able to run this project successfully on your android phone, now copy the detect.tflite file to the asset folder of your project and name it <strong>detect_quant_meta.tflite_</strong>.</p>
<p>Also create a text file called <strong>labelmap1.txt_</strong> in the asset folder. The content of this file should be similar to the example given by Tensorflow as the sample model was also trained using COCO. If you are not using the same classes like me, please use the classes you used to train your model.</p>
<p>Now update the <em>DetectorActivity</em> file. Change as below<br>
#from</p>
<pre><code>**_TF_OD_API_MODEL_FILE_** = **"detect.tflite"**;  
#to  
**_TF_OD_API_MODEL_FILE_** = **"detect_quant_meta.tflite"**;#from  
**_TF_OD_API_LABELS_FILE_** = **"labelmap.txt"**;  
#to  
**_TF_OD_API_LABELS_FILE_** = **"labelmap1.txt"**;
</code></pre>
<p>Here you are ! You made it. You can see the result at the beginning of this file</p>

