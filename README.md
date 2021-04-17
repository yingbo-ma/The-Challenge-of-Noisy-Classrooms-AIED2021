<h1>The-Challenge-of-Noisy-Classrooms-AIED2021</h1>

<h2>Description</h2>
<p>This is the repository for the following paper at the AIED conference 2021:</p> 
<p><em>The Challenge of Noisy Classrooms: Speaker Detection During Elementary Students’ Collaborative Dialogue</em></p>

<h2>Prerequisites</h2>
<p>Basics</p>
<pre>
Python3 
CPU or NVIDIA GPU + CUDA CuDNN
</pre>
<p>Prerequisites for feature extraction</p>
<pre>
opencv 3.4.3
librosa 0.8.0
</pre>
<p>Prerequisites for model training</p>
<pre>
tensowflow-gpu 2.1.0
keras 2.3.1
</pre>

<h2>Structure</h2>
<pre>
├───Feature_Extraction
│   ├───detect_facial_blob
│   │   └───model_data
│   │   └───detect_facial_blob.py
│   └───real_time_feature_extraction_with_webcam
│   │   └───model_data
│   │   └───detect_facial_blob.py
│   │   └───tract_motion_with_webcam_input.py
│   └───compute_dense_optical_flow.py
│   └───compute_mel_spectrogram.py
│   └───compute_sparse_optical_flow.py
└───Model
</pre>