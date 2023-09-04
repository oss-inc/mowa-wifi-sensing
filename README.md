# mowa-wifi-sensing

The mowa-wifi-sensing module performs real-time Wi-Fi CSI-based human activity recognition. CSI collected from the [Nexmon extractor](https://github.com/seemoo-lab/nexmon_csi) is delivered to the server using socket communication, and the server uses window-size-CSI-data as an input value for the trained activity classification model.

 >**※ Notice ※**  
>*The current version provides a supervised learning model, and will be replaced with a model applying meta-learning techniques in the future. (Expected to solve the problem of **Fall activity class** data sampling.)*

## Activity Classes
- Empty (default)
- Fall
- Sit
- Stand
- Walk

## Getting Started
Clone this repository on CSI extractor and server:
```bash
git clone https://github.com/oss-inc/mowa-wifi-sensing.git
```

### 1. Server
**Computing environment**
- Ubuntu 20.04
- Intel(R) i9-9900KF
- GeForce RTX 2080 Ti 11GB
- Python 3.8

### Installation
---
1. Move to server directory
```bash
cd server
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place the [downloaded dataset](https://drive.google.com/drive/u/0/folders/1dXykY81SBoQ46fosNJpN_Nr77TPog7AL) and [pre-trained models](https://drive.google.com/drive/u/0/folders/1INjUy_ZHfMEcDxMFnzcsFSUSBBRrfym2) as follows:
```
server
   |——————csi_dataset
   |        └——————domain_A
   |                └——————empty.csv
   |                └——————sit.csv
   |                └——————stand.csv
   |                └——————walk.csv
   |                └——————fall.csv
   |——————checkpoint
   |        └——————svl_vit
   |                └——————49.tar
   |——————dataloader
   |——————model
   |——————plot
   └——————runner
```

### Usage
---
1. You can customize the server and client configurations by modifying the `config.yaml` file.

Example:
```yaml
# Server
server_ip: 'xxx.xxx.xxx.xxx'
server_port: xxxx

# Client
client_ip: 'xxx.xxx.xxx.xxx'
client_port: xxxx
client_mac_address: 'xxxxxxxxx'
```

2. Running socket server for real-time activity recognition:
```bash
python run.py
```
---
**Using customized model**
1. After collecting CSI data, prepare a csv file for each activity according to the above directory structure.(If you want to easily extract the `.csv` file by activity class from the `.pcap` file, use `extract_activity.py` in [this repository](https://github.com/cheeseBG/pcap-to-csv.git).)

2. Model train:
```bash
python main.py train
```
3. Model evaluation:
```bash
python main.py test
```

### 2. Extractor
This module is based on CSI extracted with [Nexmon CSI Extractor](https://github.com/seemoo-lab/nexmon_csi)(Raspberry Pi, Wi-Fi chip: bcm43455c0). Therefore, the Nexmon CSI extractor installation must be preceded.

 >**※ Notice ※**  
>*Additional WLAN cards must be installed for socket communication.*

### Installation
1. Move to extractor directory
```bash
cd extractor
```

2. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

### Usage
1. Modify the `HOST` and `PORT` values at the top to match the server information in `client.py`.

2. Running socket client for real-time CSI transmission
```bash
python3 client.py
```

## Referenced Projects

This project takes inspiration from the following open-source project:
- **Nexmon**: The Nexmon project provides firmware patches for collecting CSI on Broadcom Wi-Fi chips. For more information about this project, please visit the [Nexmon GitHub repository](https://github.com/seemoo-lab/nexmon_csi).