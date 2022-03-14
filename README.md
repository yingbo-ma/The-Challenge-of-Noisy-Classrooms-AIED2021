<h1>The-Challenge-of-Noisy-Classrooms-AIED2021</h1>

<h2>Description</h2>
<p>This is the repository for the following paper at the AIED conference 2021:</p> 
<p><em>The Challenge of Noisy Classrooms: Speaker Detection During Elementary Students’ Collaborative Dialogue</em></p>
<h3>Introduction</h3>
<p>Adaptive and intelligent collaborative learning support systems are effective for supporting learning and building strong collaborative skills. This potential has not yet been realized within noisy classroom environments, where automated speech recognition (ASR) is very difficult. A key challenge is to differentiate each learner’s speech from the background noise, which includes the teachers’ speech as well as other groups’ speech. In this paper, we explore a multimodal method to identify speakers by using visual and acoustic features from ten video recordings of children pairs collaborating in an elementary school classroom. The results indicate that the visual modality was better for identifying the speaker when in-group speech was detected, while the acoustic modality was better for differentiating in-group speech from background speech. Our analysis also revealed that recurrent neural network (RNN)-based models outperformed convolutional neural network (CNN)-based models with higher speaker detection F-1 scores. This work represents a critical step toward the classroom deployment of intelligent systems that support collaborative learning.</p>
<h3>Citation</h3>
@inproceedings{ma2021challenge,
  title={The Challenge of Noisy Classrooms: Speaker Detection During Elementary Students’ Collaborative Dialogue},
  author={Ma, Yingbo and Wiggins, Joseph B and Celepkolu, Mehmet and Boyer, Kristy Elizabeth and Lynch, Collin and Wiebe, Eric},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={268--281},
  year={2021},
  organization={Springer}
}
<pre></pre>


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

<h2>Structure (Keep Updating...)</h2>
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

<h2>Other Supplementary Materials</h2>
<p>Please refer to the image files in './Images/' path.</p>
<p>1. Annotation Example</p>
<p>2. Feature Extraction Process</p>
