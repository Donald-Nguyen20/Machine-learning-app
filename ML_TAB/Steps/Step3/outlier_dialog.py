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

        table = QTableWidget(widget)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["row_index", "timestamp", "column", "value", "score", "method"]
        )

        if df is not None and not df.empty:
            table.setRowCount(len(df))
            for r, (_, row) in enumerate(df.iterrows()):
                vals = [
                    row.get("row_index", ""),
                    str(row.get("timestamp", "")),
                    row.get("column", ""),
                    "" if pd.isna(row.get("value", None)) else str(row.get("value", "")),
                    "" if pd.isna(row.get("score", None)) else f"{row.get('score', ''):.6f}" if isinstance(row.get("score", None),(int,float)) else str(row.get("score","")),
                    row.get("method", ""),
                ]
                for c, val in enumerate(vals):
                    item = QTableWidgetItem(str(val))
                    if c in (0,4):  # row_index, score
                        item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(r, c, item)
            table.resizeColumnsToContents()
        else:
            table.setRowCount(0)

        v.addWidget(table)
        self.tabs.addTab(widget, name)
