from __future__ import annotations

from pathlib import Path

import pythoncom
import win32com.client


PDF_PATH = Path("sfinal.pdf").resolve()
OUT_DOCX = Path("reference_template_wordcom.docx").resolve()


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pythoncom.CoInitialize()
    word = None
    doc = None
    pvw = None

    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        try:
            doc = word.Documents.Open(str(PDF_PATH), ReadOnly=True)
        except Exception:
            doc = None

        if doc is None or word.Documents.Count == 0:
            pvw = word.ProtectedViewWindows.Open(str(PDF_PATH))
            pvw.Edit()

            if word.Documents.Count > 0:
                doc = word.Documents.Item(1)

        if doc is None:
            raise RuntimeError("Word did not expose an editable Document from PDF")

        doc.SaveAs2(str(OUT_DOCX), FileFormat=16)
        print(f"Converted via Word COM: {OUT_DOCX}")

    finally:
        try:
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass
        try:
            if pvw is not None:
                pvw.Close()
        except Exception:
            pass
        try:
            if word is not None:
                word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
