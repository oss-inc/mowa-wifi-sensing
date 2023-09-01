import socketserver
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
import pandas as pd
from os.path import exists
from runner.utils import get_config
from model.vit import ViT

config = get_config('config.yaml')


def realTimeHPDec(share_value):
    HOST = config['server_ip']
    PORT = config['server_port']

    mac = config['client_mac_address']
    global P_COUNT
    P_COUNT = 0
    window_size = config['window_size']
    batch_size = config['batch_size']
    total_size = window_size * batch_size
    num_sub = config[config['bandwidth']]
    #SUB_NUM = '_30'



    columns = []
    for i in range(0, num_sub):
        columns.append('_' + str(i))

    # 64에 대해서만 처리중.. 추후 40MHz 이상에 대한 널 처리를 해줘야함
    null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]

    # Load pretrained model
    print('======> Load model')
    model = ViT(
            in_channels=config["in_channels"],
            patch_size=(config["patch_size"], config[config["bandwidth"]]),
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_dim=config["mlp_dim"],
            num_classes=len(config["activity_labels"]),
            in_size=[config["window_size"], config[config["bandwidth"]]]
            )
    model.summary()
    model.load_state_dict(torch.load(config['save_model_path']))

    if config['cuda']:
        model.to(config['gpu_ids'][0])
    print('======> Success')


    mac_dict = {}
    mac_dict[mac] = pd.DataFrame(columns=columns)
    #mac_dict[mac].drop(null_pilot_col_list, axis=1, inplace=True)

    class MyTcpHandler(socketserver.BaseRequestHandler):

        def handle(self):
            #print('{0} is connected'.format(self.client_address[0]))
            buffer = self.request.recv(2048)  # receive data
            buffer = buffer.decode()
            global P_COUNT
            P_COUNT += 1

            if not buffer:
                print("Fail to receive!")
                return
            else:
                recv_csi = [list(map(float, buffer.split(' ')))]
                csi_df = pd.DataFrame(recv_csi, columns=columns)

                share_value.send(-1)

                '''
                    1. Remove null & pilot subcarrier
                    2. Keep window_size 50. If 25 packets changed, choose 1 subcarrier and run model.
                '''
                # 1. Remove null & pilot subcarrier
                #csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

                # 2. Keeping window_size. If half packets changed, choose 1 subcarrier and run model
                try:
                    mac_dict[mac] = pd.concat([mac_dict[mac], csi_df], ignore_index=True)
                    if len(mac_dict[mac]) == total_size and P_COUNT == total_size:
                        c_data = np.array(mac_dict[mac])
                        print(c_data)
                        # TODO: 데이터 cuda
                        c_data = c_data.reshape(-1, 50, 1)

                        pred = model(c_data)

                        print('Predict result: {}'.format(pred))
                        thres = self.__selectThreshold(pred)

                        share_value.send(thres)

                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                        P_COUNT = 0

                    elif len(mac_dict[mac]) == 50 and P_COUNT == 25:
                        c_data = np.array(mac_dict[mac][SUB_NUM].to_list())
                        c_data = c_data.reshape(-1, 50, 1)

                        pred = model.predict(c_data)

                        print('Predict result: {}'.format(pred))
                        thres = self.__selectThreshold(pred)

                        share_value.send(thres)

                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                        P_COUNT = 0

                    elif len(mac_dict[mac]) == 50:
                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                    elif len(mac_dict[mac]) > 50:
                        print("Error!")



                except Exception as e:
                    print('Error', e)

        def __selectThreshold(self, predict):
            if predict[0][0] < 0.5:
                return 0.7
            else:
                return 0.2



    def runServer(HOST, PORT):
        print('==== Start Edge Server ====')
        print('==== Exit with Ctrl + C ====')

        try:
            server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
            server.serve_forever()  # server_forever()메소드를 호출하면 클라이언트의 접속 요청을 받을 수 있음

        except KeyboardInterrupt:
            print('==== Exit Edge server ====')


    runServer(HOST, PORT)