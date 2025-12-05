import streamlit as st
import torch
from tokenizers import Tokenizer
import os
import math

# Streamlit page configuration
st.set_page_config(page_title="Medical Transcript Summarizer", layout="wide")

# Load tokenizer
tokenizer_path = r"C:\\Users\\shris\\Documents\\College\\Study\\Fourth Year\\Sem 7\\DS&ML\\Project\\tokenizer2.json"
if not os.path.exists(tokenizer_path):
    st.error("Tokenizer file not found!")
    st.stop()
tokenizer = Tokenizer.from_file(tokenizer_path)

# Define model (same as training)
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
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
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
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Load model
model_path = r"C:\\Users\\shris\\Documents\\College\\Study\\Fourth Year\\Sem 7\\DS&ML\\Project\\model2.pt"
if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.get_vocab_size()
model = TransformerModel(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize session state properly
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "download_data" not in st.session_state:
    st.session_state.download_data = ""

# Beam search generation
def generate_summary(text, max_length=256, beam_size=3, temperature=0.8):
    try:
        input_ids = tokenizer.encode(text).ids[:512]
        input_tensor = torch.tensor([input_ids + [tokenizer.token_to_id("[PAD]")] * (512 - len(input_ids))], 
                                   dtype=torch.long).to(device)

        beams = [([tokenizer.token_to_id("[CLS]")], 0.0)]
        completed = []

        for _ in range(max_length):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == tokenizer.token_to_id("[SEP]"):
                    completed.append((seq, score))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

                with torch.no_grad():
                    output = model(input_tensor, tgt_tensor, tgt_mask=tgt_mask)
                probs = torch.softmax(output[:, -1, :] / temperature, dim=-1)
                top_probs, top_ids = torch.topk(probs, beam_size, dim=-1)

                for p, tid in zip(top_probs[0], top_ids[0]):
                    new_seq = seq + [tid.item()]
                    new_score = score + torch.log(p).item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if not beams:
                break

        all_seqs = completed + beams
        best_seq = max(all_seqs, key=lambda x: x[1])[0]
        return tokenizer.decode(best_seq, skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

# UI
st.title("Medical Transcript Summarizer")
st.markdown("Enter a medical transcript to generate a concise summary.")

with st.form("input_form"):
    transcript = st.text_area(
        "Medical Transcript",
        value=st.session_state.transcript,
        height=200,
        placeholder="Paste your transcript here...",
        key="transcript_input"  # Different key!
    )
    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.form_submit_button("Generate Summary")
    with col2:
        clear_btn = st.form_submit_button("Clear All")

# Generate summary
if generate_btn:
    if transcript.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(transcript)
        st.session_state.summary = summary
        st.session_state.download_data = summary
        st.success("Summary generated!")
    else:
        st.error("Please enter a transcript.")

# Clear button
if clear_btn:
    st.session_state.transcript = ""
    st.session_state.summary = ""
    st.session_state.download_data = ""
    st.rerun()

# Display summary
st.subheader("Generated Summary")
summary_display = st.text_area(
    "Summary",
    value=st.session_state.summary,
    height=120,
    disabled=True,
    key="summary_display"
)

# Action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Copy Summary", key="copy"):
        if st.session_state.summary:
            st.write(f'<script>navigator.clipboard.writeText(`{st.session_state.summary}`)</script>', 
                     unsafe_allow_html=True)
            st.success("Copied to clipboard!")
        else:
            st.error("No summary to copy.")

with col2:
    st.download_button(
        label="Download Summary",
        data=st.session_state.download_data,
        file_name="summary.txt",
        mime="text/plain",
        disabled=not st.session_state.download_data
    )

with col3:
    if st.button("Refresh", key="refresh"):
        st.rerun()

# Styling
st.markdown("""
<style>
    .stTextArea textarea { max-width: 700px; }
    .stButton>button { width: 150px; }
    .stForm { background: #f5f5f5; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)