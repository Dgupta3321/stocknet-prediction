import os
import pandas as pd
from tqdm import tqdm
from data_preprocessing import (
    compute_technical_features, 
    process_tweets_for_stock, 
    TECH_FEATURES, 
    SENTIMENT_FEATURES
)

def build_aligned_multimodal_dataset(price_raw_dir, tweet_preproc_dir, output_path=None):
    """
    Ingests raw prices and preprocessed tweets, engineers multimodal feature spaces,
    and applies calendar-alignment logic across asynchronous data timelines.
    """
    price_files = sorted([f for f in os.listdir(price_raw_dir) if f.endswith('.csv')])
    print(f"Aligning {len(price_files)} stock streams...")
    
    master_frames = []
    
    for f in tqdm(price_files, desc="Processing Multimodal Fusion"):
        ticker = f.replace('.csv', '')
        
        # 1. Ingest & engineer technical indicator dataframe
        csv_path = os.path.join(price_raw_dir, f)
        raw_price_df = pd.read_csv(csv_path, parse_dates=['Date'])
        raw_price_df['Ticker'] = ticker
        
        # Settle baseline structural timeline filters (Jan 2014 - Dec 2015)
        raw_price_df = raw_price_df[(raw_price_df['Date'] >= '2014-01-01') & (raw_price_df['Date'] <= '2015-12-31')]
        if raw_price_df.empty:
            continue
            
        tech_df = compute_technical_features(raw_price_df)
        
        # 2. Ingest & extract textual sentiment features from daily JSON blocks
        sentiment_df = process_tweets_for_stock(ticker, tweet_preproc_dir)
        
        if sentiment_df.empty:
            # If a stock has no associated text metrics, default fill with zeros
            for col in SENTIMENT_FEATURES:
                tech_df[col] = 0.0
            master_frames.append(tech_df)
            continue
            
        # 3. Calendar Alignment (Handling weekend tweet text leakage)
        # Sort dataframes by date to prepare for chronological merge
        tech_df = tech_df.sort_values('Date').reset_index(drop=True)
        sentiment_df = sentiment_df.sort_values('Date').reset_index(drop=True)
        
        # Use an asymmetric 'merge_asof' to shift weekend/holiday tweet metrics 
        # to forward-facing active market trading days safely.
        aligned_df = pd.merge_asof(
            tech_df, 
            sentiment_df, 
            on='Date', 
            by='Ticker', 
            direction='backward' # Grabs the most recent available text records
        )
        
        # Impute residual missing fields using zero values for missing signals
        aligned_df[SENTIMENT_FEATURES] = aligned_df[SENTIMENT_FEATURES].fillna(0.0)
        
        # Drop the initial boundary rows containing null rolling technical indicators
        aligned_df = aligned_df.dropna().reset_index(drop=True)
        master_frames.append(aligned_df)
        
    # Combine individual stock dataframes into a single master pool
    final_dataset = pd.concat(master_frames, ignore_index=True)
    
    print("\n✅ Multimodal Alignment Complete.")
    print(f"Total Aligned Samples: {len(final_dataset):,}")
    print(f"Total Structural Features: {len(TECH_FEATURES) + len(SENTIMENT_FEATURES)}")
    
    if output_path:
        final_dataset.to_csv(output_path, index=False)
        print(f"Dataset successfully compiled and stored at: {output_path}")
        
    return final_dataset
