### Marching Cubes の `sigma` と `level` の決め方

| 条件 | `sigma`（ガウス平滑化） | `level`（Marching Cubesの閾値） | 特徴 |
|------|-----------------|-----------------|-----------------------------|
| **標準的なケース** | `0.8` | `0.01` | バランスの取れたスムージングと閾値 |
| **ノイズが多い場合（形状をなめらかにしたい）** | `1.2` | `0.05` | 強めのスムージング、ノイズを削減 |
| **細かいディテールを維持したい場合** | `0.5` | `0.001` | スムージングを弱め、形状の詳細を保持 |
| **形状が粗すぎる場合** | `sigma` を上げる（例: `1.2`） | - | よりスムーズな形状を作成 |
| **形状が消えたり欠損する場合** | - | `level` を下げる（例: `0.001`） | 形状を詳細に復元 |
| **ノイズが多く残る場合** | - | `level` を上げる（例: `0.05`） | 高密度部分のみをメッシュ化 |

### **最適な値を決める実験方法**
1. `sigma = 0.8` と `level = 0.01` でまず試す。
2. 形状が粗すぎるなら `sigma` を上げる（例: `1.2`）。
3. 形状が消えたり、欠損するなら `level` を下げる（例: `0.001`）。
4. ノイズが多く残るなら `level` を上げる（例: `0.05`）。

このように **`sigma` と `level` のバランスを調整しながら試すのがベスト** です！

