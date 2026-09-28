from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
DEST = Path("data/raw/heart.csv")

DEST.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(URL, DEST)
print(f"Saved to {DEST} ({DEST.stat().st_size} bytes)")