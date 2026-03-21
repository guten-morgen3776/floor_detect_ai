## データパスの書き方
データパスはローカルで実行することはないので以下のようなフォーマットに則って書いてください　くれぐれもローカルのパスをそのまま書かないように
kaggle環境ならkaggleパス、そうでないならgoogle colabのパスにしてください
kaggleではinputは/kaggle/input/~, outputは/kaggle/working
google colabではoutputを/content/drive/MyDrive/~

```
# ─────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────
import os
from pathlib import Path


IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
if IS_KAGGLE:
    _data_root     = "/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge"
    _train_img_dir = "/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge/train"
    _train_csv     = "/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge/train.csv"
    _output_dir    = "/kaggle/working/exp001"
else:
    _data_root     = Path(os.environ.get("KAGGLE_DATA_ROOT"))
    _train_img_dir = _data_root / "train"
    _train_csv     = _data_root / "train.csv"
    _output_dir    = "/content/drive/MyDrive/output001"
```