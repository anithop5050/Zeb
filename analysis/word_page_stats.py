from __future__ import annotations

from pathlib import Path

import pythoncom
import win32com.client


FILES = [Path("reference_template.docx").resolve(), Path("anith.docx").resolve()]


def main() -> None:
    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        for f in FILES:
            if not f.exists():
                print(f"MISSING: {f}")
                continue
            doc = word.Documents.Open(str(f), ReadOnly=True)
            pages = doc.ComputeStatistics(2)  # wdStatisticPages
            words = doc.ComputeStatistics(0)  # wdStatisticWords
            chars = doc.ComputeStatistics(3)  # wdStatisticCharacters
            print(f"{f.name}: pages={pages}, words={words}, chars={chars}")
            doc.Close(False)
    finally:
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
