import os
import torchvision
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.viewer = QViewer()
        self.setCentralWidget(self.viewer)
        self.setWindowTitle("Remote Sense Extraction")
        self.setWindowState(Qt.WindowMaximized)  # 启动最大化

        # 打开按钮
        open_button = QPushButton("Open Image")
        open_button.clicked.connect(self.open_image)

        # 保存按钮
        save_button = QPushButton("Save Annotation")
        save_button.clicked.connect(self.viewer.save)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(open_button)
        layout.addWidget(save_button)
        layout.addWidget(self.viewer)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 记录上一张图片路径
        self.previous_image_path = None

        # 标记是否保存过图片
        self.image_saved = False

    def open_image(self):
        # 检查是否有未保存的图片
        if self.viewer.scene.items() and self.previous_image_path is not None and not self.image_saved:
            reply = QMessageBox.question(self, 'Save Confirmation',
                                         "Do you want to save the annotation?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.viewer.save()
            elif reply == QMessageBox.No:
                self.clear_and_open_image()
            elif reply == QMessageBox.Cancel:
                return
        else:
            self.clear_and_open_image()

    def clear_and_open_image(self):
        # 清理当前
        self.viewer.scene.clear()
        self.previous_image_path = None
        # 重置保存标志
        self.image_saved = False
        # 打开新图片
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:  # 检查是否选择了文件
                file_path = file_paths[0]
                if os.path.exists(file_path):  # 检查文件路径是否存在
                    try:
                        self.viewer.detect_and_draw(file_path)  # 调用 detect_and_draw 方法
                        self.previous_image_path = file_path
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to open image: {str(e)}")
                else:
                    QMessageBox.critical(self, "Error", "File not found.")

    def set_image_saved(self, value):
        self.image_saved = value


class QViewer(QGraphicsView):
    def __init__(self):
        super(QViewer, self).__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.image_item = QGraphicsPixmapItem()

    def wheelEvent(self, event):
        # 使用鼠标滚轮进行缩放
        delta = event.angleDelta().y() / 120
        factor = 1.1 ** delta
        self.scale(factor, factor)

    def detect_and_draw(self, image_path):
        # 加载模型
        model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval()

        # 执行图像检测
        image = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)

        # 提取检测到的边界框
        boxes = predictions[0]['boxes'].cpu().numpy()

        # 在图像上绘制边界框
        annotated_image = self.draw_boxes(image.copy(), boxes)

        # 显示带有边界框的图像
        pixmap = self.convert_to_qpixmap(annotated_image)

        # 创建新的 QGraphicsPixmapItem 对象并设置 pixmap
        new_image_item = QGraphicsPixmapItem(pixmap)

        # 将新的图片项添加到场景中
        self.scene.clear()
        self.scene.addItem(new_image_item)

        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def draw_boxes(self, image, boxes, color=(255, 0, 0), thickness=2):
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=thickness)
        return image

    def convert_to_qpixmap(self, image):
        if image.mode == 'RGB':
            r, g, b = image.split()
            image = Image.merge("RGB", (b, g, r))
        elif image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGBA", (b, g, r, a))
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.size[0], image.size[1], QImage.Format_ARGB32)
        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap

    def save(self):
        # 获取场景中的图像项
        items = self.scene.items()

        # 确保存在图像项
        if not items:
            return

        # 获取第一个图像项
        image_item = items[0]

        # 获取图像
        pixmap = image_item.pixmap()

        # 确保图像存在
        if pixmap.isNull():
            return

        # 提示用户选择保存位置
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg *.jpeg)")
        file_dialog.setDefaultSuffix("png")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            # 保存图像
            if file_path:
                pixmap.save(file_path)
                QMessageBox.information(self, "Saved", "Annotation saved successfully.")
                # 更新保存标志
                mainWindow.set_image_saved(True)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
