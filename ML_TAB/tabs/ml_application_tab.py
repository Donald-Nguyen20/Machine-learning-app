# tabs/ml_application_tab.py
from __future__ import annotations
from typing import Optional
import os, traceback
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout,
    QFileDialog, QMessageBox, QFrame
)
from ML_TAB.widgets.step_card import StepCard
from ML_TAB.Steps.Step7.Load_and_Deployment import predict_from_model
from ML_TAB.Steps.Step1.data_collection import load_rawdata
from ML_TAB.Steps.Step2.profile_report import generate_profile_json
# from ML_TAB.Steps.Step2.dashboard_widget import ProfileDashboard
from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QPushButton
from ML_TAB.Steps.Step3.outlier_tools import (detect_outliers_iqr, detect_outliers_zscore, detect_outliers_isoforest, detect_outliers_lof,)
from ML_TAB.Steps.Step3.outlier_dialog import OutlierResultsDialog



class MLApplicationTab(QWidget):
    """
    Tab ML: bố trí các StepCard theo hàng ngang với QScrollArea (scroll ngang).
    Màu sắc & style lấy từ QSS (ThemeManager), không set inline ở đây.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MLApplicationTab")

        # Root + Scroll
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("mlScroll")
        scroll.setFrameShape(QFrame.NoFrame) 
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root.addWidget(scroll)

        # Container bên trong Scroll (viewport sẽ được QSS tô nền)
        container = QWidget()
        container.setObjectName("mlContainer")
        scroll.setWidget(container)
        # 👇 cấu hình viewport (chỉ cần 1 lần setObjectName)
        vp = scroll.viewport()
        vp.setObjectName("mlViewport")
        vp.setAttribute(Qt.WA_StyledBackground, True)  # Quan trọng để nền QSS có hiệu lực

        self.Rawdata = None   # <-- THÊM: lưu DataFrame cho các bước sau dùng

        # HBox chứa các StepCard
        self.h = QHBoxLayout(container)
        self.h.setContentsMargins(16, 4, 16, 16)
        self.h.setSpacing(16)

        self._add_step_cards()
        self.h.addStretch(1)  # đẩy cụm card sát trái

        # Font chung nhẹ nhàng (màu/viền do QSS quyết định)
        base_font = QFont()
        base_font.setPointSize(10)
        self.setFont(base_font)

    def _add_step_cards(self):
        steps = [
            dict(step=1, title="Data collection",    sub="", role="step1"),
            dict(step=2, title="Statistics",         sub="", role="step2"),
            dict(step=3, title="Data preprocessing", sub="", role="step3"),
            dict(step=4, title="Data visualization", sub="", role="step4"),
            dict(step=5, title="Model building",     sub="", role="step5"),
            dict(step=6, title="Model evaluation",   sub="", role="step6"),
            dict(step=7, title="Model deployment",   sub="", role="step7"),
        ]
        CARD_W, CARD_H = 220, 110
        self.cards: list[StepCard] = []

        for cfg in steps:
            card = StepCard(cfg["step"], cfg["title"], cfg["sub"], parent=self)
            card.setProperty("variant", cfg["role"])
            card.setFixedSize(CARD_W, CARD_H)

            if cfg["step"] == 3:
                # === CỘT STEP 3: card ở trên, NÚT CON ở dưới (cùng size) ===
                box = QFrame(self)
                vlay = QVBoxLayout(box)
                vlay.setContentsMargins(0, 0, 0, 0)
                vlay.setSpacing(10)

                # 3.1) Step 3 card (giữ như cũ)
                vlay.addWidget(card, 0, Qt.AlignTop)

                # 3.2) Nút con “Detect Outlier” — CHỈ THIẾT KẾ, CHƯA GẮN GÌ
                btn = QPushButton("Detect Outlier", box)
                btn.setObjectName("btnDetectOutlier")
                btn.setFixedSize(CARD_W, CARD_H)     # kích thước BẰNG Step 3
                vlay.addWidget(btn, 0, Qt.AlignTop)
                btn.clicked.connect(self._on_detect_outlier)

                # Đưa CỘT Step 3 (card + nút con) vào hàng ngang self.h
                self.h.addWidget(box, 0, Qt.AlignTop)

                # Giữ hành vi click của Step 3 card như cũ
                card.clicked.connect(self._on_step_clicked)

                # (tuỳ chọn) lưu tham chiếu nếu cần dùng sau
                self.btnDetectOutlier = btn

            else:
                # Các step khác giữ nguyên: chỉ có card
                self.h.addWidget(card, 0, Qt.AlignTop)
                card.clicked.connect(self._on_step_clicked)

            self.cards.append(card)


    def _on_step_clicked(self, step_no: int):
        # === STEP 1: Data collection ===
        if step_no == 1:
            try:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Chọn file dữ liệu (CSV/Excel)",
                    os.path.abspath("."),
                    "CSV/Excel Files (*.csv *.xlsx *.xls)"
                )
                if not path:
                    return

                # Đọc dữ liệu về DataFrame
                self.Rawdata = load_rawdata(path)

                # Thông báo kết quả (5 dòng đầu, shape)
                head_info = self.Rawdata.head(5).to_string(index=False)
                QMessageBox.information(
                    self, "Đã nạp dữ liệu",
                    f"File: {os.path.basename(path)}\n"
                    f"Shape: {self.Rawdata.shape}\n\n"
                    f"Preview 5 dòng đầu:\n{head_info}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi nạp dữ liệu", str(e))
            return  # đã xử lý step 1, kết thúc
        # --- STEP 2: Statistics / Profiling (HTML full fidelity) ---
        if step_no == 2:
            if getattr(self, "Rawdata", None) is None:
                QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy chạy Step 1 để nạp Rawdata trước.")
                return
            try:
                json_path, html_path = generate_profile_json(
                    self.Rawdata,
                    out_dir="reports",
                    html=True,         # đảm bảo có file HTML
                    minimal=True       # True: nhanh; False: đầy đủ hơn nhưng lâu hơn
                )
                # dash = ProfileDashboard(html_path, parent=self)
                # dash.show()
            except Exception as e:
                QMessageBox.critical(self, "Lỗi Step 2", str(e))
            return

        # === CÁC STEP KHÁC (mặc định như cũ) ===
        if step_no != 7:
            print(f"[UI] Step {step_no} clicked")
            return

        # === STEP 7: Model deployment (giữ nguyên của anh) ===
        try:
            models_dir = os.path.abspath("models")
            model_path, _ = QFileDialog.getOpenFileName(
                self, "Chọn file model", models_dir,
                "Model files (*.pkl *.sav);;All files (*.*)"
            )
            if not model_path:
                return

            data_dir = os.path.abspath(".")
            data_path, _ = QFileDialog.getOpenFileName(
                self, "Chọn file dữ liệu (.csv)", data_dir,
                "CSV (*.csv);;All files (*.*)"
            )
            if not data_path:
                return

            out_path, nrows = predict_from_model(model_path, data_path)
            QMessageBox.information(
                self, "Hoàn tất",
                f"✅ Dự đoán xong {nrows} dòng.\n💾 Lưu tại: {out_path}"
            )
        except Exception:
            QMessageBox.critical(self, "Lỗi", traceback.format_exc())
    def _on_detect_outlier(self):
        if getattr(self, "Rawdata", None) is None:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy chạy Step 1 để nạp dữ liệu trước.")
            return

        df = self.Rawdata
        try:
            # bạn có thể điều chỉnh tham số tại đây:
            iqr_df   = detect_outliers_iqr(df, factor=1.5)              # per-column outliers
            zs_df    = detect_outliers_zscore(df, z=3.0)                 # per-column outliers
            iso_df   = detect_outliers_isoforest(df, contamination=0.05) # row-level outliers
            lof_df   = detect_outliers_lof(df, n_neighbors=20, contamination=0.05)


            dlg = OutlierResultsDialog(self)
            dlg.add_tab("IQR", iqr_df)
            dlg.add_tab("Z-score", zs_df)
            dlg.add_tab("IsolationForest", iso_df)
            dlg.add_tab("LOF", lof_df) 

            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi Detect Outlier", str(e))