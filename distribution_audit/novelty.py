"""區分影像量測值與分箱，唯讀支援原版重複欄名。"""

import csv

from distribution_audit.compare import COMBO_FIELDS
from distribution_audit.features import FEATURES

COMBO_COLUMNS = tuple(f"{field}_bin" for field in COMBO_FIELDS)
MEASUREMENT_PREFIX = ("id", "actual_format", "width", "height", "extension_match", *FEATURES)


def read_novelty(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {path}")
        start = len(MEASUREMENT_PREFIX)
        stop = start + len(COMBO_FIELDS)
        # 原版的數字在前，分箱在後。只接受已知順序，其他重複欄位一律拒絕。
        if (tuple(header[:start]) == MEASUREMENT_PREFIX and len(header) >= stop
                and all(value in (old, new) for value, old, new in
                        zip(header[start:stop], COMBO_FIELDS, COMBO_COLUMNS, strict=True))):
            header[start:stop] = COMBO_COLUMNS
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate CSV columns with unknown layout: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Wrong CSV field count in {path}:{line}")
            yield dict(zip(header, row, strict=True))
