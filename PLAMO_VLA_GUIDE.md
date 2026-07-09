# Plamo-2.1-2B-VL を使用したVLA学習ガイド

## 概要

このガイドでは、Plamo-2.1-2B-VL をベースとした VLA（Vision-Language-Action）モデルの学習方法を説明します。

## ファイル構成

- `model_plamo.py`: Plamo-VL ベースの VLA モデル実装
- `train_libero_plamo.py`: Libero データセットを使用した学習スクリプト

## 実装の特徴

### 現在のアーキテクチャ（従来）
```
DINOv2 + SigLIP (ビジョンエンコーダ)
         ↓
    プロジェクター
         ↓
   Llama-3.2-3B (言語モデル)
         ↓
  アクション予測
```

### Plamo-VL ベースのアーキテクチャ
```
Plamo-2.1-2B-VL（統合 VL モデル）
     ビジョンエンコーダ + 言語モデル（統合）
         ↓
  アクション予測ヘッド
         ↓
  アクション予測
```

## 主な特徴

1. **統合 VL モデル**: Plamo は既に視覚と言語が統合されているため、個別のエンコーダを用意する必要がない
2. **効率的な学習**: パラメータ数が少なく（2B）、メモリ効率が良い
3. **LoRA 微調整**: 言語モデル部分のみ LoRA で効率的に微調整可能
4. **マルチタスク学習**: 複数のタスクに対するタスクごとの損失追跡に対応

## 使用方法

### 基本的な学習実行

```bash
python train_libero_plamo.py
```

### パラメータカスタマイズ

```python
train(
    dataset_dir=["libero/libero/datasets/libero_spatial_256"],
    plamo_model_name="pfnet/plamo-2.1-2b-vl",
    batch_size=2,
    num_epochs=5,
    lr_rate=1e-5,
    lora_r=32,
    lora_alpha=128,
    freeze_vision=True,  # ビジョンエンコーダを凍結
    image_aug=True,      # 画像拡張を有効化
)
```

## 重要な実装詳細と調整点

### 1. Plamo の Processor と Tokenizer

**現在の実装:**
```python
self.processor = AutoProcessor.from_pretrained(plamo_model_name)
self.tokenizer = self.processor.tokenizer
```

**注意点:**
- Plamo の `AutoProcessor` が画像処理をどのように行うか確認が必要
- 入力画像の解像度や正規化方法が DINOv2/SigLIP と異なる可能性
- 必要に応じて `_combine_modalities()` メソッドを調整

### 2. 画像特徴の抽出

**現在の実装:**
```python
processed = self.processor(images=images, return_tensors="pt")
image_features = processed.get("image_features")
```

**調整が必要な可能性:**
```python
# Plamo の内部構造に応じて以下のように修正する可能性あり:
# - image_features のキー名
# - 出力形状 (B, seq_len, dim) vs (B, dim)
# - ビジョンエンコーダへの直接アクセス方法
```

### 3. 画像と テキストの融合

**現在の実装:**
```python
def _combine_modalities(self, image_embeds, text_embeds):
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(1)
    combined = torch.cat([image_embeds, text_embeds], dim=1)
    return combined
```

**調整が必要:**
Plamo の正確な融合方法を確認し、必要に応じて以下を実装:
- 次元の投影層
- 正しい埋め込み次元へのマッピング
- 注意マスクの再構築

### 4. 正規化と前処理

Plamo と従来モデルでは異なる画像正規化を使用する可能性:

```python
# DINOv2: ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
# SigLIP: mean=std=0.5
# Plamo: ? (確認が必要)
```

## トラブルシューティング

### 問題 1: 画像特徴の形状が予期と異なる

**原因:** Plamo の processor が返す形状が異なる可能性

**対処法:**
```python
# デバッグコード
processed = self.processor(images=images, return_tensors="pt")
print(f"Processor output keys: {processed.keys()}")
for key, val in processed.items():
    if isinstance(val, torch.Tensor):
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")
```

### 問題 2: 損失が NaN になる

**原因:** 
- 正規化ミスマッチ
- 埋め込み次元の不一致
- 勾配の爆発

**対処法:**
- `freeze_vision=False` で視覚エンコーダも微調整してみる
- 学習率を下げる
- 勾配クリッピングを追加

### 問題 3: メモリ不足

**対処法:**
```python
train(
    batch_size=1,
    gradient_accumulation_steps=4,  # 実効バッチサイズ = 4
    lora_r=8,  # LoRA ランクを下げる
)
```

## 微調整のベストプラクティス

### 1. 段階的な学習

```python
# Phase 1: 言語モデルのみ微調整
freeze_vision=True

# Phase 2: 全体を微調整
freeze_vision=False
lr_rate=5e-6  # より低い学習率
```

### 2. 学習率スケジューリング

```python
# 現在の実装: Cosine annealing with warmup
# 代案: 複数段階の学習率低下
```

### 3. LoRA パラメータ

```python
# 推奨設定
lora_r=32        # LoRA ランク
lora_alpha=128   # スケーリング係数
lora_dropout=0.1 # ドロップアウト率
```

## Plamo の詳細確認方法

以下を実行して、Plamo の構造を確認してください：

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("pfnet/plamo-2.1-2b-vl")
processor = AutoProcessor.from_pretrained("pfnet/plamo-2.1-2b-vl")

# モデル構造の確認
print(model)
print(f"Config: {model.config}")

# Processor の出力確認
import torch
dummy_images = torch.randn(1, 3, 384, 384)
processed = processor(images=dummy_images, return_tensors="pt")
print(f"Processed keys: {processed.keys()}")
for key, val in processed.items():
    if isinstance(val, torch.Tensor):
        print(f"{key}: {val.shape}")
```

## 次のステップ

1. **Plamo の内部構造を確認**: 上記のコードを実行して、exact API を確認
2. **model_plamo.py を調整**: 確認結果に基づいて `_combine_modalities()` などを修正
3. **小規模データで検証**: 少数のサンプルで学習を試して動作確認
4. **パラメータ微調整**: 最適なハイパーパラメータを探索

## 参考資料

- [Plamo GitHub](https://github.com/pfnet/plamo)
- [HuggingFace - pfnet/plamo-2.1-2b-vl](https://huggingface.co/pfnet/plamo-2.1-2b-vl)
- [OpenVLA リポジトリ](https://github.com/openvla/openvla) - VLA アーキテクチャの参考実装

## ライセンスと帰属

- Plamo: pfnet による CC-BY-NC-ND 4.0
- 本実装: 相応するライセンスで提供
