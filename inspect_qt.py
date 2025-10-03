#!/usr/bin/env python3
"""
GUI application to inspect problematic questions with both text and image answer options.
Allows visual inspection of question images, extracted text, and answer options.
"""
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, QScrollArea,
                            QGroupBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import base64
from io import BytesIO
import sys
import os

class ProblematicQuestionInspector(QMainWindow):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.df = pd.read_parquet(dataset_path)
        
        # Identify problematic questions
        self.problematic_questions = []
        for idx, row in self.df.iterrows():
            mixed_options = []
            for letter in ['A', 'B', 'C', 'D', 'E']:
                text_option = row[f'sol_{letter}']
                image_option = row[f'sol_{letter}_image_bin']
                
                has_text = pd.notna(text_option) and str(text_option) != '' and str(text_option) != 'nan'
                has_image = pd.notna(image_option) and image_option is not None
                
                if has_text and has_image:
                    mixed_options.append(letter)
            
            if mixed_options:
                self.problematic_questions.append({
                    'index': idx,
                    'row': row,
                    'mixed_options': mixed_options
                })
        
        print(f"Found {len(self.problematic_questions)} problematic questions")
        
        # Current question index
        self.current_idx = 0
        
        # Setup GUI
        self.init_ui()
        
        # Display first question if any exist
        if self.problematic_questions:
            self.show_question(0)
    
    def init_ui(self):
        self.setWindowTitle("Problematic Questions Inspector")
        self.setGeometry(100, 100, 1200, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Navigation bar
        nav_layout = QHBoxLayout()
        
        self.nav_label = QLabel(f"Question 1 of {len(self.problematic_questions)}")
        nav_layout.addWidget(self.nav_label)
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_question)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_question)
        nav_layout.addWidget(self.next_btn)
        
        # Info label
        self.info_label = QLabel("")
        nav_layout.addWidget(self.info_label)
        
        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)
        
        # Question info
        info_widget = QGroupBox("Question Information")
        info_layout = QVBoxLayout()
        
        self.id_label = QLabel("ID:")
        self.year_group_label = QLabel("Year/Group:")
        self.problem_num_label = QLabel("Problem #:")
        
        info_layout.addWidget(self.id_label)
        info_layout.addWidget(self.year_group_label)
        info_layout.addWidget(self.problem_num_label)
        info_widget.setLayout(info_layout)
        main_layout.addWidget(info_widget)
        
        # Question text
        self.question_text = QTextEdit()
        self.question_text.setMaximumHeight(100)
        self.question_text.setReadOnly(True)
        question_text_widget = QGroupBox("Question Text")
        question_text_layout = QVBoxLayout()
        question_text_layout.addWidget(self.question_text)
        question_text_widget.setLayout(question_text_layout)
        main_layout.addWidget(question_text_widget)
        
        # Question image
        self.question_img_label = QLabel()
        self.question_img_label.setAlignment(Qt.AlignCenter)
        self.question_img_label.setMinimumHeight(150)
        question_img_widget = QGroupBox("Question Image")
        question_img_layout = QVBoxLayout()
        question_img_layout.addWidget(self.question_img_label)
        question_img_widget.setLayout(question_img_layout)
        main_layout.addWidget(question_img_widget)
        
        # Scroll area for answer options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        options_widget = QWidget()
        self.options_layout = QVBoxLayout(options_widget)
        
        # Create option groups for A-E
        self.option_widgets = {}
        for letter in ['A', 'B', 'C', 'D', 'E']:
            option_group = QGroupBox(f"Option {letter}")
            option_layout = QVBoxLayout()
            
            # Text label
            text_label = QLabel("Text:")
            text_label.setStyleSheet("font-weight: bold;")
            option_layout.addWidget(text_label)
            
            text_display = QLabel()
            text_display.setWordWrap(True)
            option_layout.addWidget(text_display)
            
            # Image label
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumHeight(100)
            option_layout.addWidget(img_label)
            
            option_group.setLayout(option_layout)
            self.options_layout.addWidget(option_group)
            
            self.option_widgets[letter] = {
                'text_display': text_display,
                'img_label': img_label
            }
        
        # Add stretch to push content to top
        self.options_layout.addStretch()
        
        scroll.setWidget(options_widget)
        main_layout.addWidget(scroll)
    
    def show_question(self, idx):
        if not (0 <= idx < len(self.problematic_questions)):
            return
        
        self.current_idx = idx
        prob_q = self.problematic_questions[idx]
        row = prob_q['row']
        
        # Update navigation
        self.nav_label.setText(f"Question {idx + 1} of {len(self.problematic_questions)}")
        
        # Update question info
        self.id_label.setText(f"ID: {row.get('id', 'N/A')}")
        self.year_group_label.setText(f"Year: {row.get('year', 'N/A')}, Group: {row.get('group', 'N/A')}")
        self.problem_num_label.setText(f"Problem #: {row.get('problem_number', 'N/A')}")
        self.info_label.setText(f"Answer: {row.get('answer', 'N/A')} | Mixed Options: {', '.join(prob_q['mixed_options'])}")
        
        # Update question text
        self.question_text.setPlainText(str(row.get('problem_statement', '')))
        
        # Show question image if available
        q_img_bin = row.get('question_image')
        if pd.notna(q_img_bin) and q_img_bin is not None:
            pixmap = self.bytes_to_pixmap(q_img_bin, 600, 200)
            if pixmap:
                self.question_img_label.setPixmap(pixmap)
            else:
                self.question_img_label.setText("No image available")
        else:
            self.question_img_label.setText("No image available")
        
        # Update answer options
        for letter in ['A', 'B', 'C', 'D', 'E']:
            text_option = row.get(f'sol_{letter}', '')
            image_option = row.get(f'sol_{letter}_image_bin')
            
            # Update text
            if pd.notna(text_option) and str(text_option) != '' and str(text_option) != 'nan':
                self.option_widgets[letter]['text_display'].setText(f"{text_option}")
            else:
                self.option_widgets[letter]['text_display'].setText("(No text)")
            
            # Update image
            if pd.notna(image_option) and image_option is not None:
                pixmap = self.bytes_to_pixmap(image_option, 400, 150)
                if pixmap:
                    self.option_widgets[letter]['img_label'].setPixmap(pixmap)
                    self.option_widgets[letter]['img_label'].setText("")
                else:
                    self.option_widgets[letter]['img_label'].setText("No image")
            else:
                self.option_widgets[letter]['img_label'].setText("(No image)")
    
    def bytes_to_pixmap(self, img_bytes, max_width=None, max_height=None):
        try:
            if isinstance(img_bytes, (bytes, bytearray)):
                image_bytes = img_bytes
            elif isinstance(img_bytes, str):
                try:
                    image_bytes = base64.b64decode(img_bytes)
                except:
                    return None
            else:
                return None
            
            image = QImage.fromData(image_bytes)
            if image.isNull():
                return None
            
            pixmap = QPixmap.fromImage(image)
            
            if max_width and max_height:
                pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            return pixmap
        except Exception as e:
            print(f"Error converting bytes to pixmap: {e}")
            return None
    
    def prev_question(self):
        if self.current_idx > 0:
            self.show_question(self.current_idx - 1)
    
    def next_question(self):
        if self.current_idx < len(self.problematic_questions) - 1:
            self.show_question(self.current_idx + 1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_qt.py <path_to_dataset.parquet>")
        print("Using default: datasets/dataset_full.parquet")
        dataset_path = "datasets/dataset_full.parquet"
    else:
        dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    inspector = ProblematicQuestionInspector(dataset_path)
    inspector.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()