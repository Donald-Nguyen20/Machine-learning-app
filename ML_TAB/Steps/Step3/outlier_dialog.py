# ML_TAB/Steps/Step3/outlier_dialog.py
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QLabel, QPushButton
)
import pandas as pd

class OutlierResultsDialog(QDialog):
    """
    Hiển thị 3 tab kết quả outlier (IQR / Z-score / IsolationForest)
    Mỗi tab là QTableWidget với các cột:
    row_index | timestamp | column | value | score | method
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Outlier Detection — Results")
        self.resize(1000, 620)

        outer = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Phát hiện điểm bất thường theo 3 phương pháp."), 1)
        self.btnClose = QPushButton("Close")
        self.btnClose.clicked.connect(self.accept)
        header.addWidget(self.btnClose, 0, Qt.AlignRight)
        outer.addLayout(header)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

    def add_tab(self, name: str, df: pd.DataFrame):
        widget = QWidget(self)
        v = QVBoxLayout(widget)

        base_headers = ["row_index", "timestamp", "column", "value", "score", "method"]
        headers = base_headers.copy()

        # Nếu là tab Isolation Forest và có cột causes -> hiển thị thêm
        has_causes = (df is not None) and ("causes" in df.columns)
        if has_causes and ("isolation" in name.lower() or "iso" in name.lower()):
            headers.append("causes")

        table = QTableWidget(widget)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)

        if df is not None and not df.empty:
            table.setRowCount(len(df))
            for r, (_, row) in enumerate(df.iterrows()):
                vals = []
                for h in headers:
                    if h == "score":
                        val = row.get("score", "")
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            vals.append(f"{val:.6f}")
                        else:
                            vals.append("" if pd.isna(val) else str(val))
                    elif h == "value":
                        val = row.get("value", "")
                        vals.append("" if pd.isna(val) else str(val))
                    elif h == "timestamp":
                        vals.append(str(row.get("timestamp", "")))
                    else:
                        vals.append(row.get(h, ""))

                for c, val in enumerate(vals):
                    item = QTableWidgetItem(str(val))
                    if headers[c] in ("row_index", "score"):
                        item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(r, c, item)

            table.resizeColumnsToContents()
        else:
            table.setRowCount(0)

        v.addWidget(table)
        self.tabs.addTab(widget, name)

