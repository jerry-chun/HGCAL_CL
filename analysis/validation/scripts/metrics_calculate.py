import argparse
import pandas as pd 
import torch
from src.metrics.calculations import calc_purity, calc_efficiency, calc_response, calc_ratio

def main():
    ap = argparse.ArgumentParser(description="Calculate evaluation metrics form a dataframe.")
    ap.add_argument("-i","--input", type=str, required=True, help="Path to the input CSV file containing predictions and ground truth.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #DF: event_id, cp_id, reco_id, reco_energy, shared_energy, RtS 
    df = pd.read_csv(args.input)

    purity = calc_purity(df, threshold = 0.2)
    print(f"Purity: {purity:.4f}")

    efficiency = calc_efficiency(df, threshold = 0.7)
    print(f"Efficiency: {efficiency:.4f}")

    response, resolution = calc_response(df)
    print(f"Response: {response:.4f}, Resolution: {resolution:.4f}")

    ratio = calc_ratio(df)
    print(f"Ratio: {ratio:.4f}")

if __name__ == "__main__":
    main()

