# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
import cv2
import numpy as np
import itertools
from apex import amp

# flask
import flask
from flask_cors import CORS

# PyTorch
import torch
import torch.optim as optim

# Graphonomy
sys.path.append(os.path.join(os.getcwd(), 'Graphonomy'))
from networks import deeplab_xception_transfer

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64
from inference_all import inference

#======================
# グローバル変数
#======================
#-------------------
# flask 関連
#-------------------
app = flask.Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}}, methods=['POST', 'GET'])  # OPTIONS を受け取らないようにする（Access-Control-Allow-Origin エラー対策）
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

#-------------------
# graphonomy 関連
#-------------------
device = "gpu"
load_checkpoints_path = "checkpoints/universal_trained.pth"
model = None
debug = False

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
@app.route('/graphonomy', methods=['POST'])
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
    pose_img_pillow = conv_base64_to_pillow( json_data["pose_img_base64"] )
    pose_img_pillow.save( os.path.join( "tmp", "pose_img.png" ) )

    #------------------------------------------
    # Graphonomy の実行
    #------------------------------------------
    in_img_path = os.path.join( "tmp", "pose_img.png" )
    pose_parse_img_np, pose_parse_img_RGB_pillow = inference( net=model, img_path=in_img_path, device=device )
    pose_parse_img_pillow = Image.fromarray( np.uint8(pose_parse_img_np.transpose(0,1)) , 'L')

    pose_parse_img_pillow.save( os.path.join( "tmp", "pose_parse_img.png" ) )
    pose_parse_img_RGB_pillow.save( os.path.join( "tmp", "pose_parse_img_vis.png" ) )

    #------------------------------------------
    # 送信する画像データの変換
    #------------------------------------------
    pose_parse_img_base64 = conv_pillow_to_base64( pose_parse_img_pillow )
    pose_parse_img_RGB_base64 = conv_pillow_to_base64( pose_parse_img_RGB_pillow )

    #------------------------------------------
    # レスポンスメッセージの設定
    #------------------------------------------
    http_status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            'pose_parse_img_base64': pose_parse_img_base64,
            'pose_parse_img_RGB_base64': pose_parse_img_RGB_base64,
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
    #parser.add_argument('--host', type=str, default="localhost", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5001", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--load_checkpoints_path', default='checkpoints/universal_trained.pth', type=str, help="学習済みモデルのチェックポイントへのパス")
    parser.add_argument('--use_amp', action='store_true', help="AMP [Automatic Mixed Precision] の使用有効化")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
        
    # グローバル変数に引数を反映
    device = args.device
    load_checkpoints_path = args.load_checkpoints_path
    debug = args.debug

    #--------------------------
    # 実行 Device の設定
    #--------------------------
    if( device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    #--------------------------
    # モデルの定義
    #--------------------------
    model = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
        n_classes=20,
        hidden_layers=128,
        source_classes=7, 
    ).to(device)
    if not load_checkpoints_path == '':
        if( device == "gpu" ):
            model.load_state_dict( torch.load(load_checkpoints_path), strict=False )
        else:
            model.load_state_dict( torch.load(load_checkpoints_path, map_location="cpu"), strict=False )
    else:
        print('no model load !!!!!!!!')
        raise RuntimeError('No model!!!!')

    #-------------------------------
    # AMP の適用（使用メモリ削減効果）
    #-------------------------------
    if( args.use_amp ):
        # dummy の optimizer
        optimizer = optim.Adam( params = model.parameters(), lr = 0.0001, betas = (0.5,0.999) )

        # amp initialize
        model, optimizer = amp.initialize(
            model, 
            optimizer, 
            opt_level = args.opt_level,
            num_losses = 1
        )

    #--------------------------
    # Flask の起動
    #--------------------------
    app.debug = args.debug
    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
    