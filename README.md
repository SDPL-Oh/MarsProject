# 강화학습 기반 FPSO 스캔틀링 최적화 


## 사용 방법

1. 저장소 클론 및 패키지 설치

```bash
git clone https://github.com/SDPL-Oh/MarsProject.git
pip install -r requirements.txt
```

2. MARS2000 설치

* https://marine-offshore.bureauveritas.com/mars-20002d-ship-structural-assessment-software
* 해당 사이트로 접속하여 로그인 후 mars2000 다운로드

---
## 코드 설명
1. 경로 설정(```data/config.json```)

* Mars2000 설치된 시스템 폴더에서 batch 파일을 실행하고 저장하므로, 해당 경로에 ```TestCases``` 이하 폴더를 수동으로 생성해야 합니다.
* 강화학습을 실행하기 전 Mars2000 결과파일이 필요하므로 수동으로 batch 파일을 실행해서 결과(```pnu_ 1_S_BV RULES.txt```)를 생성해야 합니다.
* 원본 데이터를 복사하여 ```temp.ma2```로 수동으로 저장해야 합니다.
```json
{
 "mars_path": "C:/BVeritas/Mars2000/TestCases",
  "batch_path": "C:/BVeritas/Mars2000/TestCases/Batch/run.bat",
  "input_path": "C:/BVeritas/Mars2000/TestCases/InputData/pnu.ma2",
  "output_path": "C:/BVeritas/Mars2000/TestCases/Output/pnu_ 1_S_BV RULES.txt",
  "temp_path": "C:/BVeritas/Mars2000/TestCases/InputData/temp.ma2",
  "angle": "F:/MarsProject/data/angle.CSV",
  "tbar": "F:/MarsProject/data/tbar.CSV",
  "strake": "F:/MarsProject/data/strake.CSV",
  "flat": "F:/MarsProject/data/flat.CSV",
  "bulb": "F:/MarsProject/data/bulb.CSV"
}
```
2. 학습 실행
```python
python run.py
```
