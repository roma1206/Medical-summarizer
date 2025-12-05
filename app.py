import streamlit as st
import torch
from tokenizers import Tokenizer
import os
import math

# Streamlit page configuration
st.set_page_config(page_title="Medical Transcript Summarizer", layout="wide")

# Load tokenizer
tokenizer_path = "C:\\Users\\shris\\Documents\\College\\Study\\Fourth Year\\Sem 7\\DS&ML\\Project\\tokenizer.json"
if not os.path.exists(tokenizer_path):
    st.error("Tokenizer file not found. Ensure tokenizer.json is in ./Transformer/")
    st.stop()
tokenizer = Tokenizer.from_file(tokenizer_path)

# Define Transformer model
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

# Load model
model_path = "C:\\Users\\shris\\Documents\\College\\Study\\Fourth Year\\Sem 7\\DS&ML\\Project\\model.pt"
if not os.path.exists(model_path):
    st.error("Model file not found. Ensure model.pt is in ./Transformer/")
    st.stop()
vocab_size = tokenizer.get_vocab_size()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Function to generate summary
def generate_summary(text, max_length=128):
    try:
        input_ids = tokenizer.encode(text).ids
        input_ids = input_ids[:512]
        input_ids = input_ids + [tokenizer.token_to_id("[PAD]")] * (512 - len(input_ids))
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        generated_ids = [tokenizer.token_to_id("[CLS]")]
        for _ in range(max_length):
            tgt_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            with torch.no_grad():
                output = model(input_tensor, tgt_tensor, tgt_mask=tgt_mask)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated_ids.append(next_token)
            if next_token == tokenizer.token_to_id("[SEP]"):
                break

        summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Medical Transcript Summarizer")
st.markdown("Enter a medical transcript below to generate a concise summary using a Transformer model.")

# Initialize session state
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""
if "download_data" not in st.session_state:
    st.session_state.download_data = ""

# Input section
with st.form(key="transcript_form"):
    transcript = st.text_area(
        "Medical Transcript",
        placeholder="Paste or type your medical transcript here...",
        height=200,
        key="transcript"
    )
    word_count = len(transcript.split()) if transcript.strip() else 0
    st.caption(f"Word count: {word_count}")
    submit_button = st.form_submit_button("Generate Summary")

# Output section
st.subheader("Generated Summary")
summary_output = st.text_area(
    "Summary",
    value=st.session_state.generated_summary,
    height=100,
    disabled=True,
    key="summary_output"  # Changed key to avoid conflict
)

# Status and actions
status = st.empty()
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    clear_button = st.button("Clear", key="clear")
with col2:
    copy_button = st.button("Copy Summary", key="copy")
with col3:
    download_button = st.download_button(
        label="Download Summary",
        data=st.session_state.download_data,
        file_name="summary.txt",
        mime="text/plain",
        key="download",
        disabled=not st.session_state.download_data
    )

# Handle form submission
if submit_button:
    if transcript.strip():
        status.info("Generating summary...")
        summary = generate_summary(transcript)
        st.session_state.generated_summary = summary
        st.session_state.download_data = summary
        status.success("Summary generated successfully!")
    else:
        status.error("Please enter a transcript.")
        st.session_state.generated_summary = ""
        st.session_state.download_data = ""

# Handle clear button
if clear_button:
    st.session_state.transcript = ""
    st.session_state.generated_summary = ""
    st.session_state.download_data = ""
    status.empty()
    st.rerun()

# Handle copy button
if copy_button and st.session_state.generated_summary:
    st.write("<script>navigator.clipboard.writeText(`{}`)</script>".format(st.session_state.generated_summary), unsafe_allow_html=True)
    status.success("Summary copied to clipboard!")
elif copy_button:
    status.error("No summary to copy.")

# CSS for styling
st.markdown("""
<style>
    .stTextArea textarea {
        width: 100%;
        max-width: 700px;
    }
    .stButton>button {
        width: 150px;
    }
    .stForm {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)