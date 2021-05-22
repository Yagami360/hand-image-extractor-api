# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
from tqdm import tqdm 
import requests

# 自作モジュール
from api.utils import conv_base64_to_pillow, conv_pillow_to_base64

# グローバル変数
IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="API サーバーのホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5000", help="API サーバーのポート番号")
    parser.add_argument('--in_image_dir', type=str, default="datasets/sample_n5", help="入力人物画像のディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="出力人物パース画像を保存するディレクトリ")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    api_server_url = "http://" + args.host + ":" + args.port + "/api"
    if( args.debug ):
        print( "api_server_url : ", api_server_url )

    image_names = sorted( [f for f in os.listdir(args.in_image_dir) if f.endswith(IMG_EXTENSIONS)] )
    for img_name in tqdm(image_names):
        #----------------------------------
        # リクエスト送信データの設定
        #----------------------------------
        pose_img_pillow = Image.open( os.path.join(args.in_image_dir, img_name) )
        pose_img_base64 = conv_pillow_to_base64(pose_img_pillow)
        if( args.debug ):
            print( "os.path.join(args.in_image_dir, img_name) : ", os.path.join(args.in_image_dir, img_name) )
            print( "pose_img_pillow.size : ", pose_img_pillow.size )

        #----------------------------------
        # リクエスト処理
        #----------------------------------
        api_msg = {'pose_img_base64': pose_img_base64 }
        api_msg = json.dumps(api_msg)
        try:
            api_responce = requests.post( api_server_url, json=api_msg )
            api_responce = api_responce.json()
        except Exception as e:
            print( "通信失敗 [hand-image-extractor-api]" )
            print( "Exception : ", e )
            continue

        #----------------------------------
        # ファイルに保存
        #----------------------------------
        hand_img_base64 = api_responce["hand_img_base64"]
        hand_img_mask_base64 = api_responce["hand_img_mask_base64"]

        hand_img_pillow = conv_base64_to_pillow(hand_img_base64)
        hand_img_mask_pillow = conv_base64_to_pillow(hand_img_mask_base64)

        hand_img_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_hand.png" ) )
        hand_img_mask_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_hand_mask.png" ) )
        