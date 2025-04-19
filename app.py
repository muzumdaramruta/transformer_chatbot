import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import base64
import random

############################################
# PAGE CONFIGURATION AND CUSTOM CSS
############################################

st.set_page_config(
    page_title="Livermore Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS: right-align user messages (bubble) and hide avatars
st.markdown(
    """
    <style>
    /* Hide avatar images */
    [data-testid="stChatMessageAvatar"] {
        display: none !important;
    }
    /* Style user messages as right-aligned bubbles */
    .stChatMessageUser {
        background-color: #E9F5FF !important;
        color: #000 !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        margin: 10px 0 !important;
        border: 1px solid #c6ddf0 !important;
        max-width: 80%;
        text-align: right !important;
        margin-left: auto !important;
    }
    /* Style assistant messages so they blend with the background */
    .stChatMessageAssistant {
        background-color: transparent !important;
        color: #000 !important;
        border: none !important;
        padding: 0 !important;
        margin: 10px 0 !important;
        max-width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#################################################
# MODEL & TOKENIZER DEFINITIONS (Transformer Code)
#################################################

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = {}

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        sorted_vocab = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 4]
        for idx, (word, _) in enumerate(sorted_vocab, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text.split()]
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            text = " ".join([self.idx2word.get(idx, "<unk>") for idx in seq])
            texts.append(text)
        return texts

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output

# Multi-Head Attention Layer
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.dense = torch.nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(attention)

# Feed Forward Network
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model, ffn_units, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, ffn_units)
        self.linear2 = torch.nn.Linear(ffn_units, d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

# Encoder Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, x, mask):
        attn_output = self.attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout(ffn_output))

# Decoder Layer
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.attention1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout(attn1))
        attn2 = self.attention2(out1, enc_output, enc_output, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout(attn2))
        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + self.dropout(ffn_output))

# Positional Encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Transformer Model
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ffn_units, num_layers, dropout_rate, max_len):
        super(Transformer, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])
        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])
        self.fc = torch.nn.Linear(d_model, vocab_size)
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1
    def forward(self, encoder_input, decoder_input, encoder_mask=None, decoder_mask=None):
        encoder_embedded = self.embedding(encoder_input)
        encoder_embedded = self.positional_encoding(encoder_embedded)
        encoder_output = encoder_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_mask)
        decoder_embedded = self.embedding(decoder_input)
        decoder_embedded = self.positional_encoding(decoder_embedded)
        look_ahead_mask = self.create_look_ahead_mask(decoder_input.size(1)).to(decoder_input.device)
        decoder_output = decoder_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, look_ahead_mask, encoder_mask)
        return self.fc(decoder_output)

# Chat Response Function
def chat_response(question, tokenizer, model, max_len=40, device="cpu"):
    model.eval()
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = [1] + question_seq[:max_len - 2] + [2]  # Add <start> and <end> tokens
    question_seq = question_seq + [0] * (max_len - len(question_seq))  # Pad sequence
    question_tensor = torch.tensor([question_seq]).to(device)
    decoder_input = torch.tensor([[1]]).to(device)  # Start token
    response = []
    for _ in range(max_len):
        with torch.no_grad():
            output = model(question_tensor, decoder_input)
        predicted_id = torch.argmax(output[:, -1, :], dim=-1).item()
        if predicted_id == 2:  # End token
            break
        response.append(predicted_id)
        decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]]).to(device)], dim=-1)
    return tokenizer.sequences_to_texts([response])[0].replace("<start>", "").replace("<end>", "").strip()

##############################################
# Load Pretrained Model and Tokenizer (Cached)
##############################################
@st.cache_resource
def load_model_and_tokenizer():
    MAX_LEN = 40
    NUM_HEADS = 8
    D_MODEL = 512
    FFN_UNITS = 2048
    DROPOUT = 0.1
    NUM_LAYERS = 4
    VOCAB_SIZE = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with open('model/tokenizer.pkl', 'rb') as f:
            loaded_tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Tokenizer file not found. Please ensure 'model/tokenizer.pkl' exists.")
        return None, None, None, device
    try:
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, FFN_UNITS, NUM_LAYERS, DROPOUT, MAX_LEN)
        model = model.to(device)
        model.load_state_dict(torch.load("model/transformer_chatbot_gpu_deco_1.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model/transformer_chatbot_gpu_deco_1.pth' exists.")
        return loaded_tokenizer, None, None, device
    return loaded_tokenizer, model, chat_response, device

#########################################
# Livermore Trading Strategy Backtesting
#########################################
@st.cache_data
def run_livermore_strategy(stock_symbol, start_date, end_date):
    try:
        stock_data = yf.download(stock_symbol, start_date, end_date)
        if stock_data.empty:
            return None, "No data available for this stock in the selected date range"
        if len(stock_data) < 200:
            return None, "Not enough data for analysis. Please select a longer date range (at least 200 trading days)"
        df = pd.DataFrame(index=stock_data.index)
        df['Close'] = stock_data['Close']
        df['50MA'] = df['Close'].rolling(window=50).mean()
        df['200MA'] = df['Close'].rolling(window=200).mean()
        df['20High'] = df['Close'].rolling(window=20).max().shift(1)
        df['20Low'] = df['Close'].rolling(window=20).min().shift(1)
        clean_df = df.dropna()
        if len(clean_df) < 50:
            return None, "Not enough data for analysis after preparing indicators"
        positions = []
        current_position = 0
        for i in range(len(clean_df)):
            row = clean_df.iloc[i]
            if current_position == 0 and row['Close'] > row['20High'] and row['Close'] > row['50MA'] and row['Close'] > row['200MA']:
                current_position = 1
            elif current_position == 1 and row['Close'] < row['20Low']:
                current_position = 0
            positions.append(current_position)
        clean_df['Position'] = positions
        clean_df['Buy-and-Hold Return'] = clean_df['Close'].pct_change().fillna(0)
        clean_df['Strategy Return'] = clean_df['Position'].shift(1).fillna(0) * clean_df['Buy-and-Hold Return']
        clean_df['Cumulative Buy-and-Hold'] = (1 + clean_df['Buy-and-Hold Return']).cumprod() - 1
        clean_df['Cumulative Strategy'] = (1 + clean_df['Strategy Return']).cumprod() - 1

        strategy_return = clean_df['Cumulative Strategy'].iloc[-1] * 100
        buy_hold_return = clean_df['Cumulative Buy-and-Hold'].iloc[-1] * 100
        buy_signals = (clean_df['Position'].diff() > 0).sum()
        sell_signals = (clean_df['Position'].diff() < 0).sum()
        clean_df['Buy-and-Hold High'] = clean_df['Cumulative Buy-and-Hold'].cummax()
        clean_df['Strategy High'] = clean_df['Cumulative Strategy'].cummax()
        clean_df['Buy-and-Hold Drawdown'] = (clean_df['Cumulative Buy-and-Hold'] - clean_df['Buy-and-Hold High']) / (1 + clean_df['Buy-and-Hold High'])
        clean_df['Strategy Drawdown'] = (clean_df['Cumulative Strategy'] - clean_df['Strategy High']) / (1 + clean_df['Strategy High'])
        max_bh_drawdown = clean_df['Buy-and-Hold Drawdown'].min() * 100
        max_strategy_drawdown = clean_df['Strategy Drawdown'].min() * 100
        days = (clean_df.index[-1] - clean_df.index[0]).days
        years = max(days / 365, 0.01)
        annual_bh_return = ((1 + buy_hold_return/100) ** (1/years) - 1) * 100
        annual_strategy_return = ((1 + strategy_return/100) ** (1/years) - 1) * 100
        risk_free_rate = 0.02
        daily_rfr = (1 + risk_free_rate) ** (1/252) - 1
        bh_excess_return = clean_df['Buy-and-Hold Return'] - daily_rfr
        strategy_excess_return = clean_df['Strategy Return'] - daily_rfr
        bh_std = max(bh_excess_return.std(), 0.0001)
        strategy_std = max(strategy_excess_return.std(), 0.0001)
        bh_sharpe = np.sqrt(252) * (bh_excess_return.mean() / bh_std)
        strategy_sharpe = np.sqrt(252) * (strategy_excess_return.mean() / strategy_std)
        metrics = {
            'Total Return (%)': [buy_hold_return, strategy_return],
            'Annualized Return (%)': [annual_bh_return, annual_strategy_return],
            'Max Drawdown (%)': [max_bh_drawdown, max_strategy_drawdown],
            'Sharpe Ratio': [bh_sharpe, strategy_sharpe],
            'Number of Trades': ['N/A', buy_signals]
        }
        metrics_df = pd.DataFrame(metrics, index=['Buy-and-Hold', 'Livermore Strategy'])
        return clean_df, metrics_df
    except Exception as e:
        return None, f"Error in strategy calculation: {str(e)}"

#############################################
# SENTIMENT ANALYSIS AND CSV MAPPING FUNCTIONS
#############################################

def load_sentiment_mapping(csv_file):
    """
    Load a CSV file with columns 'sentiment' and 'response' and return a dictionary
    mapping each sentiment (e.g., "positive", "negative", "neutral") to a list of responses.
    """
    try:
        df = pd.read_csv(csv_file)
        mapping = {}
        for sentiment, group in df.groupby('sentiment'):
            mapping[sentiment] = group['response'].tolist()
        return mapping
    except Exception as e:
        st.error(f"Error loading sentiment mapping: {e}")
        return {}

def classify_sentiment(results_df):
    """
    Classify sentiment based on the final cumulative strategy return.
    Thresholds:
      - If return > 10%: positive
      - If return < -5%: negative
      - Otherwise: neutral
    """
    final_return = results_df['Cumulative Strategy'].iloc[-1]
    if final_return > 0.10:
        return "positive"
    elif final_return < -0.05:
        return "negative"
    else:
        return "neutral"

def get_livermore_response(sentiment, sentiment_mapping):
    responses = sentiment_mapping.get(sentiment, ["No comment."])
    return random.choice(responses)

##############################################
# PLOTLY VISUALIZATION HELPER FUNCTIONS
##############################################
def create_plotly_line_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cumulative Buy-and-Hold'] * 100,
        mode='lines',
        name='Buy-and-Hold',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cumulative Strategy'] * 100,
        mode='lines',
        name='Livermore Strategy',
        line=dict(color='green')
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Return (%)',
        legend_title='Strategy',
        height=600,
        hovermode='x unified'
    )
    return fig

def create_plotly_drawdown_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Buy-and-Hold Drawdown'] * 100,
        mode='lines',
        name='Buy-and-Hold',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Strategy Drawdown'] * 100,
        mode='lines',
        name='Livermore Strategy',
        line=dict(color='green')
    ))
    fig.update_layout(
        title='Drawdown Comparison',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        legend_title='Strategy',
        height=600,
        hovermode='x unified'
    )
    return fig

def create_plotly_positions_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='black')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['50MA'],
        mode='lines',
        name='50-day MA',
        line=dict(color='blue', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['200MA'],
        mode='lines',
        name='200-day MA',
        line=dict(color='red', width=1, dash='dot')
    ))
    buy_signals = df[df['Position'].diff() > 0]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    sell_signals = df[df['Position'].diff() < 0]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    fig.update_layout(
        title=f'{symbol} Price with Livermore Strategy Buy/Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend_title='Legend',
        height=600,
        hovermode='x unified'
    )
    return fig

def get_csv_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

#####################################
# SIDEBAR NAVIGATION AND MAIN LOGIC #
#####################################
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Strategy Backtesting"])

# Load sentiment mapping from CSV (make sure 'sentiments.csv' exists)
sentiment_mapping = load_sentiment_mapping("data/sentiments.csv")

try:
    loaded_tokenizer, model, chat_response_func, device = load_model_and_tokenizer()

    if page == "Chatbot":
        st.title("Jesse Livermore Trading Wisdom Chatbot")
        st.markdown("""
        This chatbot is trained on Jesse Livermore's trading principles.  
        Ask it questions about trading wisdom, market analysis, or Livermore's approach.
        """)
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["text"])
            else:
                st.chat_message("assistant").write(message["text"])
        if user_input := st.chat_input("Ask a question about trading or Livermore's strategies"):
            st.session_state.conversation_history.append({"role": "user", "text": user_input})
            st.chat_message("user").write(user_input)
            if loaded_tokenizer and model:
                bot_response = chat_response_func(user_input, loaded_tokenizer, model, device=device)
                st.session_state.conversation_history.append({"role": "assistant", "text": bot_response})
                st.chat_message("assistant").write(bot_response)
            else:
                st.error("Model or tokenizer failed to load. Please check your files.")

    elif page == "Strategy Backtesting":
        st.title("Jesse Livermore Trading Strategy Backtester")
        st.markdown("""
        ### Backtest Jesse Livermore's Trading Strategy
        
        This tool allows you to test a trading strategy inspired by Jesse Livermore's principles:
        - Buy on breakouts above the previous 20-day high
        - Only trade when the price is above both 50-day and 200-day moving averages
        - Exit when the trend reverses (price drops below the previous 20-day low)
        - Position management: No averaging down; only trade in the direction of strength
        """)
        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                stock_symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=4*365)
                start_date = st.date_input("Start Date", start_date)
                end_date = st.date_input("End Date", end_date)
            with col2:
                st.markdown("""
                **Popular Stock Symbols:**
                - AAPL (Apple)
                - MSFT (Microsoft)
                - NVDA (NVIDIA)
                - TSLA (Tesla)
                - AMZN (Amazon)
                - GOOGL (Google)
                - META (Meta/Facebook)
                - JPM (JPMorgan Chase)
                - V (Visa)
                - BRK-B (Berkshire Hathaway)
                """)
            submit_button = st.form_submit_button("Run Backtest")
        if submit_button:
            with st.spinner(f"Running Livermore strategy backtest on {stock_symbol}..."):
                results, metrics_df = run_livermore_strategy(stock_symbol, start_date, end_date)
                if isinstance(metrics_df, str):
                    st.error(metrics_df)
                else:
                    st.subheader(f"Backtesting Results for {stock_symbol} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
                    st.markdown("### Strategy Performance Metrics")
                    st.dataframe(metrics_df.style.format({
                        'Total Return (%)': '{:.2f}',
                        'Annualized Return (%)': '{:.2f}',
                        'Max Drawdown (%)': '{:.2f}',
                        'Sharpe Ratio': '{:.2f}',
                    }), use_container_width=True)
                    st.plotly_chart(create_plotly_line_chart(results, f"{stock_symbol}: Livermore Strategy vs Buy-and-Hold"), use_container_width=True)
                    tab1, tab2, tab3 = st.tabs(["Drawdown Analysis", "Buy/Sell Signals", "Raw Data"])
                    with tab1:
                        st.plotly_chart(create_plotly_drawdown_chart(results), use_container_width=True)
                    with tab2:
                        st.plotly_chart(create_plotly_positions_chart(results, stock_symbol), use_container_width=True)
                        buy_signals = (results['Position'].diff() > 0).sum()
                        sell_signals = (results['Position'].diff() < 0).sum()
                        col1, col2 = st.columns(2)
                        col1.metric("Buy Signals", int(buy_signals))
                        col2.metric("Sell Signals", int(sell_signals))
                    with tab3:
                        st.dataframe(results, use_container_width=True)
                        st.markdown(get_csv_download_link(results, f"{stock_symbol}_livermore_backtest.csv"), unsafe_allow_html=True)

                    # Sentiment analysis of the strategy chart
                    sentiment = classify_sentiment(results)
                    response_text = get_livermore_response(sentiment, sentiment_mapping)
                    st.markdown("### Jesse Livermore's Take on This Chart")
                    st.write(f"**Sentiment:** {sentiment.capitalize()}")
                    st.write(response_text)

                    st.markdown("""
                    ### About the Livermore Trading Strategy
                    **Key Principles:**
                    1. **Trend Following:** Only trade when the stock is above both its 50-day and 200-day moving averages.
                    2. **Breakout Entry:** Buy on breakouts above the previous 20-day high.
                    3. **Cut Losses:** Exit when the price falls below the previous 20-day low.
                    4. **Ride Winners:** Stay in the trade as long as the trend holds.
                    """)

                    st.success("Backtest completed successfully.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
