# mowa-wifi-sensing

mowa-wifi-sensing 모듈은 MOWA 프로젝트의 Wi-Fi Sensing 기능을 담당하며, 실시간 Wi-Fi CSI(채널 상태 정보)를 기반으로 한 인간 행동 인식을 수행합니다. [Nexmon extractor](https://github.com/seemoo-lab/nexmon_csi)는 Wi-Fi CSI를 수집하기 위한 도구로, 라즈베리파이에 펌웨어를 설치하여 CSI를 수집합니다. 수집된 CSI는 소켓 통신을 통해 서버로 전송되며, 서버는 사전에 정의된 윈도우 크기로 CSI 데이터를 전처리하여 훈련된 활동 분류 모델의 입력 값으로 사용합니다.

 >**※ Notice ※**  
>*현재 버전은 **supervised learning**과 **meta-learning** 방식을 모두 지원합니다.*

## README 영어 버전
🌎 [README.md 영어 버전](https://github.com/oss-inc/.github/blob/main/profile/README.md)  
<br/>

## 행동 클래스
- Empty (default)
- Fall
- Sit
- Stand
- Walk

## 시작하기
서버 및 CSI 추출기(라즈베리파이)에서 repo 복제:
```bash
git clone https://github.com/oss-inc/mowa-wifi-sensing.git
```

### 1. 서버
**컴퓨팅 환경**
- Ubuntu 20.04
- Intel(R) i9-9900KF
- GeForce RTX 2080 Ti 11GB
- Python 3.8

### 설치
---
1. 서버 디렉토리로 이동
```bash
cd server
```

2. 필수 종속성 설치
```bash
pip install -r requirements.txt
```

3. [다운로드한 데이터셋](https://drive.google.com/drive/u/0/folders/1dXykY81SBoQ46fosNJpN_Nr77TPog7AL)과 [사전 훈련된 모델](https://drive.google.com/drive/u/0/folders/1INjUy_ZHfMEcDxMFnzcsFSUSBBRrfym2)을 다음과 같이 위치:
```
server
   |——————csi_dataset
   |        └——————domain_A
   |        |       └——————empty.csv
   |        |       └——————sit.csv
   |        |       └——————stand.csv
   |        |       └——————walk.csv
   |        |       └——————fall.csv
   |        └——————domain_B
   |        └——————realtime
   |
   |——————checkpoint
   |        └——————svl_vit
   |        |       └——————svl_best_model.pt
   |        └——————few_vit
   |                └——————fsl_best_model.pt
   |——————dataloader
   |——————model
   |——————plot
   └——————runner
```
4. `realtime` 폴더에는 다른 도메인에서 수집한 새로운 데이터를 위치시킵니다.
   - 이 데이터는 메타러닝을 통해 훈련된 모델을 사용하여 **프로토타입을 생성**하는데 사용됩니다.
   - 따라서, 각 클래스에 대한 서포트 세트를 구성할 수 있는 ***few-shot***의 데이터만 필요로합니다.
   - 데모를 위해 실행할 때, `domain_A` 또는 `domain_B` 데이터 중 하나를 사용할 수 있습니다.

### 사용법
---
1. `config.yaml` 파일을 수정하여 서버 및 클라이언트 구성을 사용자 정의할 수 있습니다.

예시:
```yaml
# 서버
server_ip: 'xxx.xxx.xxx.xxx'
server_port: xxxx

# 클라이언트
client_mac_address: 'xxxxxxxxx'
```

2. 실시간 행동 인식을 위한 소켓 서버 실행:
```bash
# Supervised learning 기반 모델 사용
python run_SVL.py
```
```bash
# Meta-learning(few-shot learning) 기반 모델 사용
python run_FSL.py
```
---
**사용자 정의 모델 사용**
1. CSI 데이터 수집 후, 위의 디렉토리 구조에 따라 각 행동에 대한 CSV 파일을 준비하세요.(만약 `.pcap` 파일로부터 각 행동 클래스별로 `.csv` 파일을 쉽게 추출하고 싶다면, 이 [repo](https://github.com/cheeseBG/pcap-to-csv.git)의 `extract_activity.py`를 사용하세요.

2. 모델 학습:
```bash
# Supervised learning
python main.py --learning SVL --mode train
```
```bash
# Meta-learning
python main.py --learning FSL --mode train
```
3. 모델 평가:
```bash
# Supervised learning
python main.py --learning SVL --mode test
```
```bash
# Meta-learning
python main.py --learning FSL --mode test
```

### 2. 추출기
[Nexmon CSI Extractor](https://github.com/seemoo-lab/nexmon_csi)(Raspberry Pi, Wi-Fi chip: bcm43455c0)을 사용하여 CSI를 추출 합니다. 따라서 Nexmon CSI 추출기의 설치가 선행되어야 합니다.

 >**※ Notice ※**  
>*소켓 통신을 위해 추가 WLAN 카드가 설치되어야 합니다.*

### 설치
1. 추출기 디렉토리로 이동
```bash
cd extractor
```

2. 필수 종속성 설치
```bash
pip3 install -r requirements.txt
```

### Usage
1. 서버 정보와 일치하도록 `client.py`의 맨 위 `HOST` 및 `PORT` 값을 수정하세요. 

2. 실시간 CSI 전송을 위한 소켓 클라이언트 실행
```bash
python3 client.py
```

## 참고 프로젝트

이 프로젝트는 다음의 오픈 소스 프로젝트에서 영감을 얻었습니다:
- **Nexmon**: Nexmon 프로젝트는 Broadcom Wi-Fi 칩에서 CSI 수집을 위한 펌웨어 패치를 제공합니다. 이 프로젝트에 대한 자세한 정보는 [Nexmon GitHub repository](https://github.com/seemoo-lab/nexmon_csi)에서 확인할 수 있습니다.