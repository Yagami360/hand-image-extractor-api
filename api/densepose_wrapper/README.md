# densepose_wrapper
[DensePose](https://github.com/facebookresearch/DensePose) の推論スクリプト `tools/infer_simple.py` のラッパーモジュール。<br>
以下の機能を追加しています。

- docker 環境に対応。（公式の dockerfile やインストール方法では動作しなかったため）
- サーバー機能を追加。

- ToDo
    - [x] UV 値の等高線表示画像を出力
    - [x] パースラベルを出力
    - [ ] keypoints も取得可能

## ■ 動作環境
- NVIDIA Tesla K80
- docker
- nvidia-docker2
- docker-compose
- 5003 番ポートの開放（サーバー機能使用時のデフォルト設定）
- tqdm
- OpenCV
- matplotlib

## ■ 使い方

<!--
### ◎ サーバー API 機能非使用時

`run_densepose.sh` 内のパラメーターを適切な値に変更後、以下のコマンドを実行
```sh
$ sh run_densepose.sh
```
-->

### ◎ サーバー API 機能使用時

サーバー機能使用時は、デフォルト設定では、`5003` 番ポートが開放されている必要があります。
使用するポート番号は、`docker-compose.yml` 内の `ports:` タグ、及び、`api/app.py`, `api/request.py` の `--port` 引数の値を設定することで変更できます。<br>

`run_densepose_api.sh` のパラメーターを変更後、以下のコマンドを実行
```sh
$ sh  run_densepose_api.sh
```
