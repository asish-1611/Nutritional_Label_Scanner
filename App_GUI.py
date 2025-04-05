import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTextEdit, QWidget, QMessageBox,
    QGroupBox, QProgressBar, QSplitter, QComboBox,
    QGraphicsOpacityEffect, QTabWidget, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from nutrition_label_scanner_app import NutritionAnalyzer


class NutritionLabelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nutrition Vision - Smart Food Analysis")
        self.setGeometry(100, 100, 1400, 900)  # Increased window size
        # Set application-wide font
        app = QApplication.instance()
        app.setFont(QFont('Segoe UI', 10))
        self.setup_ui()
        self.setup_animations()
        self.current_image = None
        self.setAcceptDrops(True)

    def setup_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)  # Increased margins
        main_layout.setSpacing(20)  # Increased spacing

        # Enhanced header with gradient and shadow
        header = QLabel("Nutrition Vision - Smart Food Analysis")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #2980b9, stop:1 #3498db);
                color: white;
                font: bold 32px 'Segoe UI';
                padding: 30px;
                border-radius: 15px;
                margin: 10px;
            }
        """)
        # Add drop shadow effect to header
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(Qt.black)
        shadow.setOffset(0, 2)
        header.setGraphicsEffect(shadow)
        main_layout.addWidget(header)

        # Main content
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Input controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Image selection
        self.setup_image_selection(left_layout)
        
        # Analysis type selection
        self.setup_analysis_options(left_layout)

        # Update process button styling
        self.process_btn = QPushButton("Analyze Food Label")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                font: bold 16px 'Segoe UI';
                padding: 15px 30px;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #216694;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.process_btn.clicked.connect(self.process_image)
        left_layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Right panel - Results with tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tabs for different types of results
        self.tabs = QTabWidget()
        
        # Tab 1: Nutritional Facts
        self.nutrition_tab = QWidget()
        nutrition_layout = QVBoxLayout(self.nutrition_tab)
        self.nutrition_display = QTextEdit()
        self.nutrition_display.setReadOnly(True)
        nutrition_layout.addWidget(self.nutrition_display)
        self.tabs.addTab(self.nutrition_tab, "Nutritional Facts")
        
        # Tab 2: Health Recommendations
        self.recommendations_tab = QWidget()
        recommendations_layout = QVBoxLayout(self.recommendations_tab)
        self.recommendations_display = QTextEdit()
        self.recommendations_display.setReadOnly(True)
        recommendations_layout.addWidget(self.recommendations_display)
        self.tabs.addTab(self.recommendations_tab, "Health Recommendations")
        
        # Tab 3: Raw Text
        self.raw_text_tab = QWidget()
        raw_text_layout = QVBoxLayout(self.raw_text_tab)
        self.raw_text_display = QTextEdit()
        self.raw_text_display.setReadOnly(True)
        raw_text_layout.addWidget(self.raw_text_display)
        self.tabs.addTab(self.raw_text_tab, "Raw Text")
        
        right_layout.addWidget(self.tabs)

        # Add panels to main layout
        content_layout.addWidget(left_panel, 40)
        content_layout.addWidget(right_panel, 60)
        main_layout.addWidget(content_widget)

    def setup_image_selection(self, layout):
        group = QGroupBox("Food Label Image")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 16px 'Segoe UI';
                border: 2px solid #bdc3c7;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }
            QGroupBox::title {
                color: #2c3e50;
                padding: 0 15px;
            }
        """)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(15)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setText("Drag & drop an image or click 'Browse'")
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 3px dashed #3498db;
                border-radius: 15px;
                background-color: #f8f9fa;
                color: #7f8c8d;
                font: 16px 'Segoe UI';
                padding: 25px;
            }
            QLabel:hover {
                background-color: #edf2f7;
                border-color: #2980b9;
            }
        """)

        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #3498db, stop:1 #2980b9);
                color: white;
                font: bold 16px 'Segoe UI';
                padding: 15px 30px;
                border-radius: 8px;
                min-width: 150px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #2980b9, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: #216694;
                padding: 17px 30px 13px 30px;
            }
        """)
        browse_btn.clicked.connect(self.browse_image)

        group_layout.addWidget(self.image_preview)
        group_layout.addWidget(browse_btn, alignment=Qt.AlignCenter)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def setup_analysis_options(self, layout):
        group = QGroupBox("Analysis Options")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 16px 'Segoe UI';
                border: 2px solid #bdc3c7;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }
            QGroupBox::title {
                color: #2c3e50;
                padding: 0 15px;
            }
            QComboBox {
                padding: 12px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background: white;
                min-width: 200px;
                font: 14px 'Segoe UI';
            }
            QComboBox:hover {
                border: 2px solid #3498db;
                background: #f8f9fa;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #2c3e50;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                selection-background-color: #3498db;
                selection-color: white;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px;
                min-height: 25px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #edf2f7;
            }
            QLabel {
                font: 14px 'Segoe UI';
                color: #2c3e50;
                margin-right: 15px;
            }
        """)
        group_layout = QVBoxLayout()
        
        # Analysis detail level
        detail_layout = QHBoxLayout()
        detail_label = QLabel("Analysis Detail:")
        self.detail_combo = QComboBox()
        self.detail_combo.addItems(["Basic", "Detailed", "Comprehensive"])
        detail_layout.addWidget(detail_label)
        detail_layout.addWidget(self.detail_combo)
        
        # Health focus
        focus_layout = QHBoxLayout()
        focus_label = QLabel("Health Focus:")
        self.focus_combo = QComboBox()
        self.focus_combo.addItems(["General", "Weight Loss", "Fitness", "Heart Health", "Diabetes"])
        focus_layout.addWidget(focus_label)
        focus_layout.addWidget(self.focus_combo)
        
        group_layout.addLayout(detail_layout)
        group_layout.addLayout(focus_layout)
        group.setLayout(group_layout)
        layout.addWidget(group)

    def setup_animations(self):
        self.opacity_effect = QGraphicsOpacityEffect()
        self.tabs.setGraphicsEffect(self.opacity_effect)

        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in.setDuration(500)
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Food Label Image", "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.current_image = file_path
            pixmap = QPixmap(file_path)
            self.image_preview.setPixmap(
                pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def process_image(self):
        if not self.current_image:
            QMessageBox.warning(self, "Error", "Please select an image first")
            return

        # Get analysis options
        detail_level = self.detail_combo.currentText().lower()
        health_focus = self.focus_combo.currentText().lower()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.process_btn.setEnabled(False)
        
        # Clear previous results
        self.nutrition_display.clear()
        self.recommendations_display.clear()
        self.raw_text_display.clear()

        # Start processing
        self.worker = ProcessingThread(self.current_image, detail_level, health_focus)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_results(self, result):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Display nutritional facts
        self.nutrition_display.setHtml(result['nutrition_html'])
        
        # Display recommendations
        self.recommendations_display.setHtml(result['recommendations_html'])
        
        # Display raw text
        self.raw_text_display.setPlainText(result['raw_text'])
        
        # Switch to recommendations tab
        self.tabs.setCurrentIndex(1)
        
        # Run animation
        self.fade_in.start()

    def show_error(self, message):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Processing Error", message)


class ProcessingThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(self, image_path, detail_level, health_focus):
        super().__init__()
        self.image_path = image_path
        self.detail_level = detail_level
        self.health_focus = health_focus
        self.analyzer = NutritionAnalyzer()

    def run(self):
        try:
            # Extract text with OCR
            self.progress_update.emit(20)
            text = self.analyzer.extract_text(self.image_path)
            self.progress_update.emit(40)
            
            # Analyze nutrition
            nutrition_data = self.analyzer.analyze_nutrition(text)
            self.progress_update.emit(60)
            
            # Get health recommendations
            recommendations = self.analyzer.get_health_recommendations(
                nutrition_data, 
                detail_level=self.detail_level,
                health_focus=self.health_focus
            )
            self.progress_update.emit(80)
            
            # Format results
            result = {
                'nutrition_html': self.format_nutrition_html(nutrition_data),
                'recommendations_html': self.format_recommendations_html(recommendations),
                'raw_text': text
            }
            self.progress_update.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def format_nutrition_html(self, nutrition_data):
        html = """
        <div style='font-family: Segoe UI, Arial; padding: 20px;'>
            <h2 style='color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>
                Nutritional Information
            </h2>
            <table style='width:100%; border-collapse:collapse; margin-top: 20px;'>
            <tr style='background-color:#2980b9; color:white;'>
                <th style='padding:12px; text-align:left; border-radius:5px 0 0 0;'>Nutrient</th>
                <th style='padding:12px; text-align:right; border-radius:0 5px 0 0;'>Amount</th>
            </tr>
        """

        for nutrient, value in nutrition_data.items():
            unit = ""
            if "protein" in nutrient.lower() or "fat" in nutrient.lower() or "carb" in nutrient.lower():
                unit = "g"
            elif "calorie" in nutrient.lower():
                unit = "kcal"
            elif "sodium" in nutrient.lower():
                unit = "mg"

            html += f"""
            <tr style='border-bottom:1px solid #eee; background-color:#f8f9fa;'>
                <td style='padding:12px;'>{nutrient}</td>
                <td style='padding:12px; text-align:right; font-weight:bold;'>{value}{unit}</td>
            </tr>
            """

        html += "</table></div>"
        return html

    def format_recommendations_html(self, recommendations):
        html = """
        <div style='font-family: Segoe UI, Arial; padding: 20px;'>
            <h2 style='color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;'>
                Health Recommendations
            </h2>
        """

        for category, items in recommendations.items():
            html += f"""
            <div style='margin-top: 20px; background-color: #f8f9fa; border-radius: 5px; padding: 15px;'>
                <h3 style='color: #2980b9; margin-top: 0;'>{category}</h3>
                <ul style='list-style-type: none; padding-left: 0;'>
            """
            for item in items:
                html += f"""
                <li style='margin: 10px 0; padding: 10px; background-color: white; 
                          border-left: 4px solid #3498db; border-radius: 4px;'>
                    {item}
                </li>
                """
            html += "</ul></div>"

        html += "</div>"
        return html


# Add styling for tabs (add this at the end of setup_ui method)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background: #f8f9fa;
                border: 1px solid #bdc3c7;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #2980b9;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #bdc3c7;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font: 13px 'Segoe UI';
                background-color: #ffffff;
            }
        """)

        # Style the progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #2980b9;
                border-radius: 3px;
            }
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = NutritionLabelApp()
    window.show()
    sys.exit(app.exec_())