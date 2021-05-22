# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
import cv2
import numpy as np
import itertools
import torch

# flask
import flask
from flask_cors import CORS

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64

#======================
# グローバル変数
#======================
args = None
device = None

GRAPHONOMY_NAME_TO_IDX = {
	"Background" : 0,
	"Hat" : 1, "Hair" : 2, "Glove" : 3, "Sunglasses" : 4,
	"UpperClothes" : 5, "Dress" : 6, "Coat" : 7,
	"Socks" : 8, "Pants" : 9, "Neck" : 10, "Scarf" : 11, "Skirt" : 12,
	"Face" : 13,
	"LeftArm" : 14, "RightArm" : 15,
	"LeftLeg" : 16, "RightLeg" : 17,
	"LeftShoe" : 18, "RightShoe" : 19,
	"RightHand" : 20, "LeftHand" : 21,
}

#-------------------
# flask 関連
#-------------------
app = flask.Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}}, methods=['POST', 'GET'])  # OPTIONS を受け取らないようにする（Access-Control-Allow-Origin エラー対策）
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

#================================================================
# "http://host_ip:port_id" リクエスト送信時の処理
#================================================================
@app.route('/')
def index():
    print( "リクエスト受け取り" )
    return

#================================================================
# "http://host_ip:port_id/openpose" にリクエスト送信時の処理
#================================================================
@app.route('/api', methods=['POST'])
def responce():
    print( "リクエスト受け取り" )
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    #------------------------------------------
    # 送信された json データの取得
    #------------------------------------------
    if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
        json_data = json.loads(flask.request.json)
    else:
        json_data = flask.request.get_json()

    #------------------------------------------
    # 送信された画像データの変換
    #------------------------------------------
    pose_img_base64 = json_data["pose_img_base64"]
    pose_img_pillow = conv_base64_to_pillow( pose_img_base64 )
    if( args.debug ):
        pose_img_pillow.save( os.path.join( "debug", "pose_img.png" ) )

    #------------------------------------------
    # DensePose の実行
    #------------------------------------------
    densepose_msg = {'pose_img_base64': pose_img_base64 }
    densepose_msg = json.dumps(densepose_msg)
    try:
        densepose_responce = requests.post( args.densepose_server_url, json=densepose_msg )
        densepose_responce = densepose_responce.json()
    except Exception as e:
        print( "通信失敗 [DensePose]" )
        print( "Exception : ", e )
        http_status_code = 400
        response = flask.jsonify(
            {
                'status':'NG',
            }
        )
        return http_status_code, response

    pose_densepose_base64 = densepose_responce["iuv_img_base64"]
    pose_densepose_pillow = conv_base64_to_pillow(pose_densepose_base64)
    if( args.debug ):
        pose_densepose_pillow.save( os.path.join( "debug", "pose_densepose.png" ) )

    #------------------------------------------
    # Graphonomy
    #------------------------------------------
    graphonomy_msg = {'pose_img_base64': pose_img_base64 }
    graphonomy_msg = json.dumps(graphonomy_msg)     # dict を JSON 文字列として整形して出力
    try:
        graphonomy_responce = requests.post( args.graphonomy_server_url, json=graphonomy_msg )
        graphonomy_responce = graphonomy_responce.json()

    except Exception as e:
        print( "通信失敗 [Graphonomy]" )
        print( "Exception : ", e )
        http_status_code = 400
        response = flask.jsonify(
            {
                'status':'NG',
            }
        )
        return http_status_code, response

    pose_parse_img_base64 = graphonomy_responce["pose_parse_img_base64"]
    pose_parse_img_RGB_base64 = graphonomy_responce["pose_parse_img_RGB_base64"]
    pose_parse_img_pillow = conv_base64_to_pillow(pose_parse_img_base64)
    pose_parse_img_RGB_pillow = conv_base64_to_pillow(pose_parse_img_RGB_base64)
    if( args.debug ):
        pose_parse_img_pillow.save( os.path.join( "debug", "pose_parse_img.png" ) )
        pose_parse_img_RGB_pillow.save( os.path.join( "debug", "pose_parse_img_rgb.png" ) )

    #------------------------------------------
    # 手画像の取得
    #------------------------------------------
    pose_img_np = cv2.cvtColor(np.asarray(pose_img_pillow), cv2.COLOR_RGB2BGR)
    pose_parse_np = np.asarray(pose_parse_img_pillow)
    pose_densepose_np = cv2.cvtColor(np.asarray(pose_densepose_pillow), cv2.COLOR_RGB2BGR)
    pose_densepose_parse_np = pose_densepose_np[:,:,0]

    arm_mask_np = (pose_parse_np==GRAPHONOMY_NAME_TO_IDX["LeftArm"]).astype(np.int) + (pose_parse_np==GRAPHONOMY_NAME_TO_IDX["RightArm"]).astype(np.int)
    hand_mask_np = (pose_densepose_parse_np==3).astype(np.int) + (pose_densepose_parse_np==4).astype(np.int)

    hand_mask_np = arm_mask_np * hand_mask_np
    hand_img_np = hand_mask_np * pose_img_np

    hand_img_mask_pillow = Image.fromarray(hand_mask_np)
    hand_img_pillow = Image.fromarray(hand_img_np)

    #------------------------------------------
    # 送信する画像データの変換
    #------------------------------------------
    hand_img_base64 = conv_pillow_to_base64( hand_img_pillow )
    hand_img_mask_base64 = conv_pillow_to_base64( hand_img_mask_pillow )

    #------------------------------------------
    # レスポンスメッセージの設定
    #------------------------------------------
    http_status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            'hand_img_base64': hand_img_base64,
            'hand_img_mask_base64': hand_img_mask_base64,
        }
    )

    # レスポンスメッセージにヘッダーを付与（Access-Control-Allow-Origin エラー対策）
    #response.headers.add('Access-Control-Allow-Origin', '*')
    #response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    if( app.debug ):
        print( "response.headers : \n", response.headers )

    return response, http_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5000", help="ポート番号")
    parser.add_argument('--graphonomy_server_url', type=str, default="http://graphonomy_server_gpu_container:5003/graphonomy", help="Graphonomy サーバーの URL")
    parser.add_argument('--densepose_server_url', type=str, default="http://densepose_server:5005/densepose", help="DensePose サーバーの URL")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument("--gpu_ids", default="0", help="使用GPU番号")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()

    str_gpu_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_gpu_id in str_gpu_ids:
        gpu_id = int(str_gpu_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)  

    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if( args.debug ):
        if not os.path.exists("debug"):
            os.mkdir("debug")
        
    #------------------------------------
    # 実行 Device の設定
    #------------------------------------
    if( torch.cuda.is_available() ):
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
        print( "実行デバイス :", device)
        print( "GPU名 :", torch.cuda.get_device_name(device))
        print("torch.cuda.current_device() =", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
        print( "実行デバイス :", device)

    #------------------------------------
    # グローバル変数の設定
    #------------------------------------
    args = args
    device = device

    #--------------------------
    # Flask の起動
    #--------------------------
    app.debug = args.debug
    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )