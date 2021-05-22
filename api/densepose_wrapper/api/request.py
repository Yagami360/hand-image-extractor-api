# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
from tqdm import tqdm 
import requests

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64

# グローバル変数
IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="ホスト名（コンテナ名 or コンテナ ID）")
    #parser.add_argument('--host', type=str, default="localhost", help="ホスト名（コンテナ名 or コンテナ ID）")
    #parser.add_argument('--host', type=str, default="densepose_container", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5003", help="DensePose サーバーのポート番号")
    parser.add_argument('--in_image_dir', type=str, default="../infer_data/sample_n5", help="入力人物画像のディレクトリ")
    parser.add_argument('--results_dir', type=str, default="../results_api/sample_n5", help="出力人物パース画像を保存するディレクトリ")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    densepose_server_url = "http://" + args.host + ":" + args.port + "/densepose"
    if( args.debug ):
        print( "densepose_server_url : ", densepose_server_url )

    image_names = sorted( [f for f in os.listdir(args.in_image_dir) if f.endswith(IMG_EXTENSIONS)] )
    for img_name in tqdm(image_names):
        #----------------------------------
        # リクエスト送信データの設定
        #----------------------------------
        pose_img_pillow = Image.open( os.path.join(args.in_image_dir, img_name) )
        pose_img_base64 = conv_pillow_to_base64(pose_img_pillow)

        #----------------------------------
        # リクエスト処理
        #----------------------------------
        densepose_msg = {'pose_img_base64': pose_img_base64 }
        densepose_msg = json.dumps(densepose_msg)     # dict を JSON 文字列として整形して出力
        try:
            densepose_responce = requests.post( densepose_server_url, json=densepose_msg )
            densepose_responce = densepose_responce.json()
        except Exception as e:
            print( "通信失敗 [DensePose]" )
            print( "Exception : ", e )
            continue

        #----------------------------------
        # ファイルに保存
        #----------------------------------
        # IUV
        iuv_img_base64 = densepose_responce["iuv_img_base64"]
        iuv_img_pillow = conv_base64_to_pillow(iuv_img_base64)
        iuv_img_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_IUV.png" ) )

        # IND
        inds_img_base64 = densepose_responce["inds_img_base64"]
        inds_img_pillow = conv_base64_to_pillow(inds_img_base64)
        inds_img_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_INDS.png" ) )

        # パース画像
        """
        parse_img_base64 = densepose_responce["parse_img_base64"]
        parse_img_pillow = conv_base64_to_pillow(parse_img_base64)
        parse_img_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_parse.png" ) )
        """
        
        # 等高線画像
        """
        contour_img_base64 = densepose_responce["contour_img_base64"]
        contour_img_pillow = conv_base64_to_pillow(contour_img_base64)
        contour_img_pillow.save( os.path.join( args.results_dir, img_name.split(".")[0] + "_contour.png" ) )
        """