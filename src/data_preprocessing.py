import os
import json
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

TECH_FEATURES = [
    'Return', 'Log_Return', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 
    'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
    'Momentum_5', 'Momentum_10', 'Volatility_5', 'Volatility_10', 
    'Volume_Change', 'Volume_SMA_5', 'Volume_Ratio', 'High_Low_Ratio', 'Close_Open_Ratio'
]

SENTIMENT_FEATURES = [
    'Tweet_Count', 'Polarity_Mean', 'Subjectivity_Mean', 
    'Vader_Pos_Mean', 'Vader_Neg_Mean', 'Vader_Neu_Mean', 'Vader_Comp_Mean'
]

def compute_technical_features(group):
    df = group.sort_values('Date').copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    
    df['Return'] = close.pct_change()
    df['Log_Return'] = np.log(close / close.shift(1))
    
    df['SMA_5'] = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_Upper'] = bb_mid + 2 * bb_std
    df['BB_Lower'] = bb_mid - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (bb_mid + 1e-10)
    df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    df['Momentum_5'] = close / close.shift(5) - 1
    df['Momentum_10'] = close / close.shift(10) - 1
    df['Volatility_5'] = df['Return'].rolling(5).std()
    df['Volatility_10'] = df['Return'].rolling(10).std()
    
    df['Volume_Change'] = volume.pct_change()
    df['Volume_SMA_5'] = volume.rolling(5).mean()
    df['Volume_Ratio'] = volume / (df['Volume_SMA_5'] + 1e-10)
    
    df['High_Low_Ratio'] = (high - low) / (low + 1e-10)
    df['Close_Open_Ratio'] = (close - df['Open']) / (df['Open'] + 1e-10)
    df['Target'] = (close.shift(-1) > close).astype(int)
    
    return df

def process_tweets_for_stock(ticker, tweet_preproc_dir):
    tweet_dir = os.path.join(tweet_preproc_dir, ticker)
    if not os.path.exists(tweet_dir):
        return pd.DataFrame()
    
    sid = SentimentIntensityAnalyzer()
    records = []
    
    for date_file in sorted(os.listdir(tweet_dir)):
        filepath = os.path.join(tweet_dir, date_file)
        try:
            date = pd.to_datetime(date_file)
        except:
            continue
            
        tb_polarities, tb_subjectivities = [], []
        vader_pos, vader_neg, vader_neu, vader_comp = [], [], [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    tweet_data = json.loads(line)
                    text = " ".join(tweet_data.get('text', []))
                    if not text: continue
                    
                    analysis = TextBlob(text)
                    tb_polarities.append(analysis.sentiment.polarity)
                    tb_subjectivities.append(analysis.sentiment.subjectivity)
                    
                    vs = sid.polarity_scores(text)
                    vader_pos.append(vs['pos'])
                    vader_neg.append(vs['neg'])
                    vader_neu.append(vs['neu'])
                    vader_comp.append(vs['compound'])
                except:
                    continue
                    
        if tb_polarities:
            records.append({
                'Date': date, 'Ticker': ticker,
                'Tweet_Count': len(tb_polarities),
                'Polarity_Mean': np.mean(tb_polarities),
                'Subjectivity_Mean': np.mean(tb_subjectivities),
                'Vader_Pos_Mean': np.mean(vader_pos),
                'Vader_Neg_Mean': np.mean(vader_neg),
                'Vader_Neu_Mean': np.mean(vader_neu),
                'Vader_Comp_Mean': np.mean(vader_comp)
            })
            
    return pd.DataFrame(records)
