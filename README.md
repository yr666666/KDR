# KDR
Code of paper “Knowledge Decomposition and Replay: A Novel Cross-modal Image-Text Retrieval Continual Learning Method”

（1）Pre-run Setup:

  **Weight Setup:**
  Place the weights in the appropriate location. The weights for ViT have already been uploaded to the checkpoints folder. You can change the path to ViT's weights in net.py at lines 274 and 275. For BERT's weights and configuration files, you need to download them yourself through Hugging Face. Replace the paths in net.py at lines 279 and 284, in data.py at line 50, and in kdr.py at line 20 with your own BERT weights, vocabulary, and configuration file directories.
  
  **Data Preparation:**
  We have placed the splits for 5 datasets in ./data (both zipped and unzipped files). You need to download the MSCOCO dataset separately, and then replace the root file path. The MSCOCO root file paths to be replaced are in data.py at line 76 and in kdr.py at line 274.
<img width="615" alt="图片1" src="https://github.com/yr666666/KDR/assets/41632617/f2519a17-ecef-48a7-aa1a-79cc6df02572">

  
  **Configuration Settings:**
  Set the number of tasks in main.py at line 87.
  Adjust the learning rate in kdr.py at lines 362-363.
  Configure batch size and epochs in main.py at lines 62-67.

  （2）To Run: python main.py
