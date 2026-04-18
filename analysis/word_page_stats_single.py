from __future__ import annotations

import sys
from pathlib import Path

import pythoncom
import win32com.client


WD_STAT_PAGES = 2
WD_STAT_WORDS = 0
WD_STAT_CHARS = 3


def get_stats(path: Path) -> None:
    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(path), ReadOnly=True)
        pages = doc.ComputeStatistics(WD_STAT_PAGES)
        words = doc.ComputeStatistics(WD_STAT_WORDS)
        chars = doc.ComputeStatistics(WD_STAT_CHARS)
        print(f"{path.name}: pages={pages}, words={words}, chars={chars}")
    finally:
        try:
            if doc is not None:
                doc.Close(False)
        except Exception as e:
            print(f"doc close warning: {e}")
        try:
            if word is not None:
                word.Quit()
        except Exception as e:
            print(f"word quit warning: {e}")
        pythoncom.CoUninitialize()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python word_page_stats_single.py <docx>")
    p = Path(sys.argv[1]).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    get_stats(p)


if __name__ == "__main__":
    main()
