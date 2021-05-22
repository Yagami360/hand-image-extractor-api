# graphonomy_wrapper
[Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) の推論スクリプト [`inference.py`](https://github.com/Gaoyiminggithub/Graphonomy/blob/master/exp/inference/inference.py) のラッパーモジュール。<br>
以下の機能を追加しています。

- 単一の画像ではなく、指定したフォルダ内の全人物画像に対して、人物パース画像を生成。<br>
- CPU と GPU の両方での動作に対応。<br>
- サーバー機能を追加。
- AMP [Automatic Mixed Precision] を用いて、処理時間と使用メモリを削減（GPU 動作時のみ有効）
- docker 環境に対応。

## ■ 動作環境

### ◎ conda 環境使用時

- Ubuntu : シェルスクリプト `.sh` のみ Ubuntu での動作を想定しています
- Python : 3.6
- Anacoda : 
- Pytorch = 0.4.0 or Pytorch = 1.1.x
    - オリジナルの Graphonomy は Pytorch = 0.4.0 での動作環境になっているが、推論スクリプトは 1.x 系でも動作することを確認済み
- Pillow : < 7.0.0
- OpenCV : 
- tqdm : 
- networkx : 
- scipy : 
- Apex (Amp: Automatic Mixed Precision)
    - Apex のインストール方法は、https://github.com/NVIDIA/apex を参照してください。

- サーバー起動使用時は、以下のモジュールが追加で必要になります
    - flask : 
    - flask_cors :
    - requests : 

### ◎ Docker 環境使用時

- GPU 版
    - nvidia 製 GPU 搭載マシン
    - nvidia-docker2

## ■ 使い方

1. 事前学習済みモデルのダウンロード<br>
    まず以下のスクリプトを実装し、学習済みモデルをダウンロードして、checkpoints 以下のフォルダに保管する必要があります。
    ```sh
    $ sh download_model.sh
    ```

    - [Download 先 (Universal trained model)](https://drive.google.com/file/d/1sWJ54lCBFnzCNz5RTCGQmkVovkY9x8_D/view)<br>
    - 事前学習済みモデルの詳細は、オリジナルの [Graphonomy](ttps://github.com/Gaoyiminggithub/Graphonomy) の `README.md` を参照

### ◎ サーバー機能非使用時（Docker 非使用時）

- 推論スクリプトの実行（GPU で実行時）<br>
    ```sh
    # 視覚用の人物パース画像の RGB画像も生成する場合（AMP 使用）
    # --in_image_dir : 入力人物画像のディレクトリ
    # --results_dir : 人物パース画像のディレクトリ
    $ python run_inference_all.py \
        --device gpu \
        --in_image_dir sample_n5 \
        --results_dir results \
        --load_checkpoints_path checkpoints/universal_trained.pth \
        --save_vis \
        --use_amp
    ```
    ```sh
    # グレースケール画像のみ生成する場合（AMP 使用）
    $ python run_inference_all.py \
        --use_gpu gpu \
        --in_image_dir sample_n5 \
        --results_dir results \
        --load_checkpoints_path checkpoints/universal_trained.pth \
        --use_amp
    ```

- 推論スクリプトの実行（CPU で実行時）<br>
    ```sh
    # 視覚用の人物パース画像の RGB画像も生成する場合
    # --in_image_dir : 入力人物画像のディレクトリ
    # --results_dir : 人物パース画像のディレクトリ
    $ python run_inference_all.py \
        --device cpu \
        --in_image_dir sample_n5 \
        --results_dir results \
        --load_checkpoints_path checkpoints/universal_trained.pth \
        --save_vis
    ```
    ```sh
    # グレースケール画像のみ生成する場合
    $ python run_inference_all.py \
        --use_gpu cpu \
        --in_image_dir sample_n5 \
        --results_dir results \
        --load_checkpoints_path checkpoints/universal_trained.pth
    ```

### ◎ サーバー機能使用時 （Docker 非使用時）
サーバー機能使用時は、デフォルト設定では、5001 番ポートが開放されている必要があります。 使用するポート番号は、`app.py`, `request.py` の --port 引数の値を設定することで変更できます。

- サーバーの起動（GPU使用時）
    ```sh
    $ python app.py \
        --host 0.0.0.0 --port 5001 --device gpu
    ```

- サーバーの起動（CPU使用時）
    ```sh
    $ python app.py \
        --host 0.0.0.0 --port 5001 --device cpu
    ```

- リクエストの送信
    ```sh
    $ python request.py \
        --host 0.0.0.0 --port 5001 \
        --in_image_dir sample_n5 \
        --results_dir results
    ```

### ◎ サーバー機能使用時 （Docker 使用時）
コンテナ内でサーバーを起動して Graphonomy を実行します。<br>
サーバー機能使用時は、デフォルト設定では、5001 番ポート（GPUコンテナ） と 5002 番ポート（CPUコンテナ）が開放されている必要があります。 <br>
使用するポート番号は、docker-compose.yml 内の ports: タグ、及び、`app.py`, `request.py` の --port 引数の値を設定することで変更できます。<br>

- docker イメージの作成 & コンテナの起動 & サーバーの起動
    ```sh
    $ docker-compose stop
    $ docker-compose up -d
    ```

- リクエストの送信（GPU コンテナ）
    ```sh
    $ python request.py \
        --host 0.0.0.0 --port 5001 \
        --in_image_dir sample_n5 \
        --results_dir results
    ```

- リクエストの送信（CPU コンテナ）
    ```sh
    $ python request.py \
        --host 0.0.0.0 --port 5002 \
        --in_image_dir sample_n5 \
        --results_dir results
    ```
