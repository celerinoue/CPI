
# アッセイデータ作成

## アッセイデータ配置
data/Class_mini # 小さいデータセット
data/Class # 全タンパク質

## アッセイデータ作成
```
kgcn-chem --assay_dir data/Class_mini --output multimodal_mini.jbl --multimodal >& log/multimodal_mini.log

```

## 










## アッセイデータ配置
```
ChEMBL_sample/gpcr
```
## データ作成
```
sh build_dataset.sh
```

## 実行
```
kgcn train --config  config_mm_gin.json
```

## 可視化
```
kgcn visualize --config  config_mm_gin.json
```
全データ可視化しようとするので注意

## テストデータ作成
```
kgcn-chem --assay_dir ChEMBL_sample/gpcr -a 50 --assay_num_limit 100 --output multimodal.jbl --multimodal \
  --assay_domain_name_clone multimodal.jbl \
  --max_len_seq <学習時の配列の最大長>
```
