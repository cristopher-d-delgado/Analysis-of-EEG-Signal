import re
from pathlib import Path
from collections import defaultdict

def parse_experiments(raw_path):
    pattern_with_session = re.compile(r"s(\d+)_ex(\d+)_s(\d+)")
    pattern_no_session = re.compile(r"s(\d+)_ex(\d+)")

    experiments = defaultdict(list)

    raw = Path(raw_path)

    for file in raw.glob("*.txt"):
        filename = file.stem

        match = pattern_with_session.search(filename)

        if match:
            subject, experiment, session = match.groups()
        else:
            match = pattern_no_session.search(filename)
            if match:
                subject, experiment = match.groups()
                session = None
            else:
                continue

        ex_key = f"ex{int(experiment):02d}"

        experiments[ex_key].append({
            "subject": int(subject),
            "session": int(session) if session is not None else None,
            "path": file
        })

    return experiments