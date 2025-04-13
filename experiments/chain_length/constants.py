from typing import List

SECOND = 1000 * 1000
SAMPLE_SIZES = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150]
HEAD_SAMPLINGS: List[str] = [f"HeadBasedTraces{s}" for s in SAMPLE_SIZES]
ALL_SAMPLINGS: List[str] = HEAD_SAMPLINGS + ["ErrorBasedTraces", "DurationBasedTraces"]