import subprocess
import os

# 배치 파일 경로
bat_file = r"C:\BVeritas\Mars2000\TestCases\Batch\test_run.bat"

# 실행
process = subprocess.run(bat_file, shell=True, capture_output=True, text=True)

# 실행 로그 확인
print("STDOUT:", process.stdout)
print("STDERR:", process.stderr)
