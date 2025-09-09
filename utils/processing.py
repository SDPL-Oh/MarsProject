import re
from collections import defaultdict

def parse_ma2_sections(file_path):
    sections = defaultdict(list)
    current_section = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            match = re.match(r"-+\s*(\w+)\s*-+", line)
            if match:
                current_section = match.group(1).lower()  # 예: bsd, main, panels, nodes, strakes
                continue
            if current_section:
                sections[current_section].append(line)

    # 문자열 리스트를 다시 합쳐 저장
    return {k: "\n".join(v).strip() for k, v in sections.items()}

parsed = parse_ma2_sections("example.txt")

# 예시 출력
print("BSD Section Preview:\n", parsed.get("BSD", "")[:300])
print("MAIN Section Preview:\n", parsed.get("MAIN", "")[:300])
print("PANELS Section Preview:\n", parsed.get("PANELS", "")[:300])
print("NODES Section Preview:\n", parsed.get("NODES", "")[:300])
print("STRAKES Section Preview:\n", parsed.get("STRAKES", "")[:300])
