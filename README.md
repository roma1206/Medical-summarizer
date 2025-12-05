# Transformer Summarizer Project
- Python
- Streamlit
- PyTorch
- License
A web-based application built with Streamlit and PyTorch to generate concise summaries of medical transcripts using a custom-trained Transformer model. This project was developed as part of a Data Science and Machine Learning course to demonstrate natural language processing (NLP) for medical text summarization.

Table of Contents
- Project Overview
- Features
- Dataset
- Model Architecture
- Installation
- Usage
- Training the Model
- Project Structure
- Troubleshooting
- Future Improvements
- License
- Acknowledgments

Project Overview
The Medical Transcript Summarizer is designed to assist healthcare professionals by generating brief, coherent summaries from detailed medical transcripts. The application uses a custom Transformer model trained from scratch to process medical text and produce summaries. The model is trained on the MTSamples dataset, and the web interface is built using Streamlit for ease of use.
Key components:

Tokenizer: A WordPiece tokenizer trained on medical transcripts and summaries.
Model: A Transformer with 6 encoder and 6 decoder layers, trained for summarization.
Interface: A Streamlit app allowing users to input transcripts and view, copy, or download summaries.

Features

Text Input: Enter or paste medical transcripts via a user-friendly text area.
Summary Generation: Produces concise summaries using a custom Transformer model.
Interactive UI: Includes buttons to generate, clear, copy, and download summaries.
Word Count: Displays the word count of the input transcript.
Error Handling: Validates input and provides clear error messages for missing files or empty inputs.
GPU Support: Accelerates inference on CUDA-enabled GPUs if available.

Dataset
The project uses the MTSamples dataset, which contains medical transcriptions and their corresponding summaries. The dataset is stored as mtsamples.csv with columns:

transcription: Full medical transcript.
description: Brief summary (used as the target).

Preprocessing:

Removed rows with missing transcriptions or summaries.
Appended [SEP] tokens to summaries for consistent training.
Tokenized using a custom WordPiece tokenizer with a vocabulary size of 30,000.

Model Architecture
The summarization model is a custom Transformer implemented in PyTorch, with the following specifications:

Tokenizer: WordPiece with special tokens ([PAD], [UNK], [CLS], [SEP], [MASK]).
Embedding Dimension: d_model=512.
Attention Heads: 8.
Layers: 6 encoder layers, 6 decoder layers.
Feedforward Dimension: 2048.
Dropout: 0.1.
Training: 100 epochs with Adam optimizer, cosine annealing scheduler, and label smoothing.

The model uses positional encoding and is trained to generate summaries autoregressively, with greedy decoding for faster inference in the Streamlit app.
Installation
Prerequisites

Python: 3.8 or higher.
Operating System: Windows (tested), Linux, or macOS.
Hardware: CUDA-enabled GPU recommended for faster inference (CPU supported).
Dependencies:bashpip install streamlit torch tokenizers pandas rouge-score

GPU Setup (Optional)
For GPU acceleration:

Install NVIDIA CUDA Toolkit (e.g., CUDA 11.8) and cuDNN.
Install PyTorch with CUDA support:bashpip install torch --index-url https://download.pytorch.org/whl/cu118
Verify GPU availability:pythonimport torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

Setup Instructions

Clone the Repository:bashgit clone https://github.com/your-username/medical-transcript-summarizer.git
cd medical-transcript-summarizer
Install Dependencies:bashpip install -r requirements.txt
Download Model and Tokenizer:
Place model.pt (trained model weights) and tokenizer.json (WordPiece tokenizer) in the project directory: C:\Users\shris\Documents\College\Study\Fourth Year\Sem 7\DS&ML\Project\.
Alternatively, train the model yourself (see Training the Model).

Prepare Dataset:
Place mtsamples.csv in the project directory or a Google Drive folder (for training in Colab).


Usage
Running the Streamlit App

Navigate to the project directory:bashcd C:\Users\ROMA\Documents\College\Study\Fourth Year\Sem 7\DS&ML\Project
Run the Streamlit app with file watcher disabled (to avoid Windows-specific issues):bashstreamlit run app.py --server.fileWatcherType none
Open the app in your browser at http://localhost:8501.

Using the App

Input Transcript:
Paste or type a medical transcript into the text area.
The word count is displayed below the input.

Generate Summary:
Click "Generate Summary" to produce a summary.
A spinner indicates processing, and the summary appears in the output text area.

Other Actions:
Clear: Resets the input and output fields.
Copy Summary: Copies the summary to the clipboard.
Download Summary: Downloads the summary as summary.txt.


Example
Input Transcript:
textHISTORY OF PRESENT ILLNESS: The patient presents today for followup. No dysuria, gross hematuria, fever, chills. She continues to have urinary incontinence, especially while changing from sitting to standing position, as well as urge incontinence...
Output Summary:
textPatient reports urinary incontinence and vaginal protrusion, concerned about prolapse. No dysuria, hematuria, or fever. Impression notes improved urine retention post-vaginal reconstruction and possible prolapse. Plan includes continuing Flomax, reducing Urecholine, pelvic exam, and CT scan for abdominal distention.
Training the Model
To train the Transformer model from scratch:

Setup in Google Colab:
Mount Google Drive:pythonfrom google.colab import drive
drive.mount('/content/drive')
Install dependencies:bash!pip install pandas torch tokenizers rouge-score

Prepare Dataset:
Place mtsamples.csv in /content/drive/MyDrive/Transformer/.

Run Training Script:
Use the training notebook (train.ipynb) or copy the training code from the project repository.
Key parameters:
Epochs: 100
Batch Size: 4
Learning Rate: 0.0001 (with cosine annealing)
Input Length: 512 tokens
Output Length: 128 tokens

The script saves model.pt and tokenizer.json to /content/drive/MyDrive/Transformer/.

Download Files:pythonfrom google.colab import files
files.download('/content/drive/MyDrive/Transformer/model.pt')
files.download('/content/drive/MyDrive/Transformer/tokenizer.json')
Move Files:
Place model.pt and tokenizer.json in the project directory on your local machine.


Project Structure
textmedical-transcript-summarizer/
├── app.py                  # Streamlit application
├── train.ipynb             # Training notebook for the Transformer model
├── mtsamples.csv           # Dataset (not included in repo, must be sourced)
├── model.pt                # Trained model weights
├── tokenizer.json          # WordPiece tokenizer
├── requirements.txt        # Python dependencies
├── README.md               # This file
Troubleshooting

Slow Summary Generation:
Ensure GPU is enabled (torch.cuda.is_available() should return True).
Reduce max_length in app.py (e.g., from 128 to 64) for faster inference.
Re-train with a smaller model (d_model=256, 3 layers) if CPU-bound.

Model Loading Error:
Verify model.pt matches the TransformerModel architecture (d_model=512, 6 layers).
Check file paths:pythonimport os
print(os.path.exists("C:\\Users\\shris\\Documents\\College\\Study\\Fourth Year\\Sem 7\\DS&ML\\Project\\model.pt"))

Streamlit Errors:
Update Streamlit: pip install --upgrade streamlit.
Run with --server.fileWatcherType none to avoid Windows file watcher issues.

Incoherent Summaries:
Ensure model.pt is trained with 100 epochs and [SEP] tokens in summaries.
Revert to beam search (beam_size=2) in app.py for better quality:pythondef generate_summary(text, max_length=128, beam_size=2, temperature=0.8):


Future Improvements

Beam Search: Re-enable beam search with optimized parameters for better summary quality.
Model Compression: Apply quantization or pruning to reduce model size and speed up inference.
Summary History: Store previous summaries in session_state for user reference.
PDF Export: Add support for downloading summaries as PDF files.
Dark Mode: Implement a dark theme for the Streamlit app.
Cloud Deployment: Deploy the app to Streamlit Community Cloud or Render for public access.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: MTSamples for providing medical transcript data.
Libraries: PyTorch, Streamlit, Hugging Face Tokenizers, and rouge-score.
Guidance: Course instructors and peers at [Your University] for feedback and support.
