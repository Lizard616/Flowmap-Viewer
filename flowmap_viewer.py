import sys, time, math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout,
    QPushButton, QWidget, QHBoxLayout, QDoubleSpinBox, QLabel, QTabWidget
)
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;

void main()
{
    TexCoord = aTexCoord;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform sampler2D flowMap;
uniform float time;
uniform float flowSpeed;

void main()
{
    vec2 flow = texture(flowMap, TexCoord).rg * 2.0 - 1.0;
    vec2 offset = flow * mod(time, 4.0) * flowSpeed;
    FragColor = texture(diffuseMap, TexCoord + offset);
}
"""

def load_texture(path):
    img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(img, dtype=np.uint8)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return tex_id

def create_sphere(radius, sectors, stacks):
    vertices, texcoords, indices = [], [], []

    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * math.pi / stacks
        xy = radius * math.cos(stack_angle)
        z = radius * math.sin(stack_angle)

        for j in range(sectors + 1):
            sector_angle = j * 2 * math.pi / sectors
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            vertices.extend([x, y, z])
            texcoords.extend([j / sectors, i / stacks])

    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(texcoords, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )

class FlowmapSphere(QGLWidget):
    def __init__(self):
        super().__init__()
        self.shader = None
        self.diffuse_tex = None
        self.flow_tex = None
        self.vao = None
        self.index_count = 0
        self.time_start = time.time()
        self.flow_speed = 0.3

        # Orbit camera
        self.last_mouse_pos = None
        self.yaw, self.pitch = 0.0, 0.0
        self.distance = 3.0

    def initializeGL(self):
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        vertices, texcoords, indices = create_sphere(1.0, 64, 32)
        self.index_count = len(indices)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(2)
        ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
        glBufferData(GL_ARRAY_BUFFER, texcoords.nbytes, texcoords, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        elapsed = time.time() - self.time_start

        # Orbit camera
        
        # Start with position at a fixed distance on Z
        eye = np.array([0.0, 0.0, self.distance])

        # Apply pitch rotation around X axis
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(self.pitch), -math.sin(self.pitch)],
            [0, math.sin(self.pitch), math.cos(self.pitch)]
        ])

        # Apply yaw rotation around Y axis
        rot_y = np.array([
            [math.cos(self.yaw), 0, math.sin(self.yaw)],
            [0, 1, 0],
            [-math.sin(self.yaw), 0, math.cos(self.yaw)]
        ])

        # Combine rotations and apply to the eye vector
        rotation = rot_y @ rot_x
        eye = rotation @ eye
        center = np.array([0.0, 0.0, 0.0])

        up = np.array([0.0, 1.0, 0.0])
        f = (center - eye)
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)

        view = np.identity(4, dtype=np.float32)
        view[:3, 0] = s
        view[:3, 1] = u
        view[:3, 2] = -f
        view[3, :3] = -eye @ np.array([s, u, -f])

        # Projection matrix
        aspect = self.width() / self.height()
        fov = math.radians(45)
        near, far = 0.1, 100.0
        f = 1.0 / math.tan(fov / 2)
        projection = np.array([
            [f/aspect, 0.0, 0.0,                           0.0],
            [0.0,      f,   0.0,                           0.0],
            [0.0,      0.0, (far+near)/(near-far),        -1.0],
            [0.0,      0.0, (2*far*near)/(near-far),       0.0]
        ], dtype=np.float32)

        model = np.identity(4, dtype=np.float32)

        # Send uniforms
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection)
        glUniform1f(glGetUniformLocation(self.shader, "time"), elapsed % 4.0)
        glUniform1f(glGetUniformLocation(self.shader, "flowSpeed"), self.flow_speed)

        # Textures
        if self.diffuse_tex:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.diffuse_tex)
            glUniform1i(glGetUniformLocation(self.shader, "diffuseMap"), 0)

        if self.flow_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.flow_tex)
            glUniform1i(glGetUniformLocation(self.shader, "flowMap"), 1)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.yaw += dx * 0.01
            self.pitch += dy * 0.01
            self.pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.pitch))
            self.last_mouse_pos = event.pos()

    def set_diffuse(self, path):
        self.diffuse_tex = load_texture(path)

    def set_flow(self, path):
        self.flow_tex = load_texture(path)

    def set_flow_speed(self, value):
        self.flow_speed = value


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flowmap 3D Sphere Viewer")
        self.setGeometry(100, 100, 900, 700)

        self.glWidget = FlowmapSphere()
        self.tabs = QTabWidget()

        # Viewer tab
        viewer_tab = QWidget()
        v_layout = QVBoxLayout()
        load_diffuse = QPushButton("Load Diffuse Map")
        load_flow = QPushButton("Load Flow Map")
        load_diffuse.clicked.connect(self.load_diffuse)
        load_flow.clicked.connect(self.load_flow)
        v_layout.addWidget(self.glWidget)
        v_layout.addWidget(load_diffuse)
        v_layout.addWidget(load_flow)
        viewer_tab.setLayout(v_layout)

        # Settings tab
        settings_tab = QWidget()
        s_layout = QHBoxLayout()
        self.speed_slider = QDoubleSpinBox()
        self.speed_slider.setDecimals(2)
        self.speed_slider.setSingleStep(0.05)
        self.speed_slider.setValue(0.3)
        self.speed_slider.setRange(0.0, 5.0)
        self.speed_slider.valueChanged.connect(self.glWidget.set_flow_speed)
        s_layout.addWidget(QLabel("Flow Speed:"))
        s_layout.addWidget(self.speed_slider)
        settings_tab.setLayout(s_layout)

        self.tabs.addTab(viewer_tab, "Viewer")
        self.tabs.addTab(settings_tab, "Settings")
        self.setCentralWidget(self.tabs)

    def load_diffuse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Diffuse Map", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.glWidget.set_diffuse(path)

    def load_flow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Flow Map", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.glWidget.set_flow(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
