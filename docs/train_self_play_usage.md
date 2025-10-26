# `train_self_play.py` の使い方

このスクリプトは、エミュレータ上で CPU 同士の自己対戦を行い、ターン終了時の盤面スナップショットを特徴量に変換して単純なロジスティック回帰モデルを学習します。以下の手順で実行できます。

## 前提条件

- Python 3.9 以上
- 追加ライブラリ: `numpy`
- リポジトリ直下で実行すること（`emulator.py` などにパスが通っている必要があります）

```bash
pip install numpy
```

## 基本的な実行方法

もっともシンプルな実行例は以下です。

```bash
python train_self_play.py
```

デフォルト設定では 100 ゲームの自己対戦を行い、そのデータでロジスティック回帰モデルを 200 エポック学習します。標準出力に学習・検証統計が表示されます。

## 主なオプション

| オプション | デフォルト | 説明 |
|-------------|------------|------|
| `--games` | `100` | 自己対戦させるゲーム数。増やすと学習データが増えますが時間も伸びます。 |
| `--learning-rate` | `0.1` | 勾配降下法の学習率。 |
| `--epochs` | `200` | エポック数。 |
| `--validation-split` | `0.2` | データセットのうち検証用に回す割合 (0～1)。0 や 1 にすると分割しません。 |
| `--l2` | `1e-4` | L2 正則化係数。 |
| `--seed` | `1234` | 乱数シード。 |
| `--log-interval` | `20` | 何エポックごとに損失を stderr に表示するか。 |
| `--save-model` | なし | 指定すると学習済み重みと正規化統計を `.npz` で保存します。 |
| `--save-dataset` | なし | 収集した特徴量とラベル、正規化統計を `.npz` で保存します。 |
| `--metrics-output` | なし | ゲーム毎の統計などを JSON で保存します。 |
| `--agent` | `safe-simple` | 自己対戦で使う CPU エージェントのプリセット (`simple` または `safe-simple`)。 |

任意のオプションは CLI 引数で指定します。例: 200 ゲームを遊ばせ、モデルとデータセットを保存する場合:

```bash
python train_self_play.py \
  --games 200 \
  --save-model artifacts/model.npz \
  --save-dataset artifacts/dataset.npz \
  --metrics-output artifacts/metrics.json
```

## 出力内容

- 標準出力: 学習・検証データ数、損失、正解率。
- `--log-interval` に応じた学習経過が標準エラーに出力されます。
- `--save-model` を指定すると以下の内容を含む `.npz` ファイルが生成されます。
  - `weights`: 学習済み重みベクトル
  - `bias`: バイアス項
  - `feature_mean` / `feature_std`: 正規化に使った平均と標準偏差
- `--save-dataset` を指定すると特徴量 (`features`) とラベル (`labels`)、正規化統計が `.npz` として保存されます。
- `--metrics-output` を指定すると JSON にシミュレーション概要（勝敗数や各ゲームの最大スタック深度など）が書き出されます。

## よくある質問

### Q. 途中で `Unexpected input request` というエラーが出ます

A. デフォルトの `safe-simple` エージェントは、対話的なターゲット指定が必要な呪文カードを自動的にスキップするようになっており、通常はこのエラーは発生しません。独自エージェントを組み込んだ場合は、ターン進行中に `Game` が `input_func` を呼ばないよう配慮してください。

### Q. 実行に時間がかかります

A. `--games` の値を減らす、または `--epochs` を小さくすることで短縮できます。初回は `--games 20 --epochs 50` などから試すと挙動を確認しやすいです。

### Q. 保存したモデルをどう使えばいいですか？

A. `.npz` の中身は単純なロジスティック回帰の重みと正規化統計なので、NumPy で読み込んで推論処理を書けばそのまま利用できます。例えば:

```python
import numpy as np

data = np.load("artifacts/model.npz")
weights = data["weights"]
bias = float(data["bias"][0])
feature_mean = data["feature_mean"]
feature_std = data["feature_std"]

# features は `train_self_play.py` と同じ形状 (dim,) のベクトル
normalised = (features - feature_mean) / feature_std
logits = normalised @ weights + bias
prob = 1.0 / (1.0 + np.exp(-logits))
```

これをベースに、エージェントの行動選択ロジックへ組み込むことができます。

