# mowa-wifi-sensing

1. 서버, 라즈베리파이에서 깃클론
2. 서버에서 ./server/config.yaml 내에 사용할 ip, port 값 수정
3. python main.py train 으로 모델 학습
4. python run.py 실행

위 과정 이후 라즈베리파이에서 아래 절차대로 수행

1. nexmon세팅 -> 혹시 LAN이 아닌 WLAN을 사용해서 socket 통신을 할 경우 nexmon 세팅하는 과정에서는 WLAN 카드 미리 꼽지 않기!!  
   -> wlan0을 csi값 받아오는 용도로 써야하는데 가끔 꼽아놓은 WLAN이 wlan0로 인식되는 경우가 있어서 꼬임  
   -> 따라서 nexmon 세팅 이후 추가적인 WLAN 카드를 꼽도록....  
2. client.py 상단에 ip, port 값을 서버값과 동일하게 세팅
3. python3 client.py 실행

!!주의할점!!
실행 종료할때는 client를 먼저 종료해야함 -> 클라이언트 돌아가는데 서버 먼저 종료하면 nexmon 깨짐
