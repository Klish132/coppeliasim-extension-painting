import time

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QSurfaceFormat, QTextCursor, QIcon
from PyQt5.QtWidgets import QStatusBar, QSizePolicy, QWidget, QMainWindow, QApplication, QVBoxLayout, QOpenGLWidget, \
    QPushButton, QTextEdit, QLabel, QScrollArea, QFileDialog
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread

from zmqRemoteApi import RemoteAPIClient

import numpy as np
import pyrr
import sys
import argparse

from wavefront_object import WfObject
from camera import Camera
from path_generator import PathGenerator

sel_color = [0.749, 0.749, 0.749]
sel_triangle_colors = np.array(sel_color * 3, dtype="float32")

unsel_color = [0.407, 0.407, 0.611]
unsel_triangle_colors = np.array(unsel_color * 3, dtype="float32")

vertex_src = """
# version 330


layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;
layout(location = 3) in vec3 a_id;
layout(location = 4) in vec3 a_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out mat4 v_pvm;
out vec3 v_position;
out vec2 v_texture;
flat out vec3 v_id;
flat out vec3 v_color;

void main()
{
    v_pvm = projection * view * model;
    v_position = a_position;
    v_texture = a_texture;
    v_id = a_id;
    v_color = a_color;
}
"""

geometry_src = """
# version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in mat4 v_pvm[];
in vec3 v_position[];
in vec2 v_texture[];
flat in vec3 v_id[];
flat in vec3 v_color[];

out vec3 g_bary_pos;
out vec3 g_position;
out vec2 g_texture;
flat out vec3 g_id;
flat out vec3 g_color;

void main() {
    for (int i = 0; i < 3; ++i) {
        vec4 point_3dp = gl_in[i].gl_Position;
        gl_Position = v_pvm[i] * vec4(v_position[i], 1.0);
        g_bary_pos = vec3(i == 0, i == 1, i == 2);
        g_position = v_position[i];
        g_texture = v_texture[i];
        g_id = v_id[i];
        g_color = v_color[i];
        
        EmitVertex();
    }
    
    EndPrimitive();
}
"""

fragment_src = """
# version 330

in vec3 g_bary_pos;
in vec3 g_position;
in vec2 g_texture;
flat in vec3 g_id;
flat in vec3 g_color;

out vec4 out_color;

uniform sampler2D s_texture;
uniform float camera_x;
uniform float camera_y;
uniform float camera_z;
// 0 - obj, 1 - grid
uniform int frag_switcher;
// 0 - default, 1 - color picking
uniform int buffer_switcher;

void main()
{
    float dist = length(vec3(camera_x, camera_y, camera_z) - g_position);
    float alpha = clamp(dist / 80, 0.0, 1.0);
    
    // Main buffer
    if (buffer_switcher == 0) {
        // Object
        if (frag_switcher == 0) {
            bool showWire = (g_bary_pos.x < 0.02
                          || g_bary_pos.y < 0.02
                          || g_bary_pos.z < 0.02);
            if (showWire) {
                out_color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                //out_color = texture(s_texture, g_texture) * vec4(0.1, 0.1, 0.1, 1.0);
                out_color = vec4(g_color, 1.0);
            }
        }
        // Grid
        else if (frag_switcher == 1) {
            float size = 1.0/2.0;   // size of the tile
            bool x_center = (g_position.x > -0.0015 && g_position.x < 0.015);
            bool z_center = (g_position.z > -0.0015 && g_position.z < 0.015);
            float edge = (x_center || z_center ? 4.0 : 4.0) / g_texture.y; // size of the edge
            vec2 uv = sign(vec2(edge) - mod(g_texture, size));
            vec4 color = vec4(sign(uv.x + uv.y + 2));
            color.xyz = (x_center ? vec3(1.0, 0.0, 0.0) : (z_center ? vec3(0.0, 0.0, 1.0) : color.xyz * 0.0));
            color.w = color.w * (0.5 - alpha);
            //color.w = 0.5;
            out_color = color;
        }
    }
    // Color selection buffer
    else if (buffer_switcher == 1) {
        // Object
        if (frag_switcher == 0) {
            out_color = vec4(g_id, 1.0);
        }
        // Grid
        else if (frag_switcher == 1) {
            out_color = vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
}
"""


class GeneratorThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(object)

    def __init__(self, triangles, width, height, pos, orient):
        super().__init__()
        self.triangles = triangles
        self.width = width
        self.height = height
        #self.width = 0.2
        #self.height = 0.4
        self.pos = pos
        self.orient = orient

    def run(self):
        self.progress.emit("[Path generation]: Starting path generation...")
        generator = PathGenerator(self.triangles)
        #path = generator.do_main_routine(self.width, self.height * 0.8, self.pos, self.orient)

        start_time = time.time()
        axes = generator.find_main_axes()
        axes_time = time.time() - start_time
        self.progress.emit(f"[Path generation]: Found main axes in {round(axes_time, 2)} s.")
        groups = generator.group_triangles_by_orthogonal_axes()
        groups_time = time.time() - start_time - axes_time
        self.progress.emit(f"[Path generation]: Formed groups in {round(groups_time, 2)} s.")
        patches = generator.form_patches(groups)
        patches_time = time.time() - start_time - axes_time - groups_time
        self.progress.emit(f"[Path generation]: Formed patches in {round(patches_time, 2)} s.")
        paths = generator.create_paths_from_patches(patches, self.width, self.height * 0.8)
        final_path = generator.connect_all_paths(paths, self.pos, self.orient)
        path_time = time.time() - start_time - axes_time - groups_time - patches_time
        self.progress.emit(f"[Path generation]: Generated path in {round(path_time, 2)} s.")
        generation_time = time.time() - start_time
        self.progress.emit(f"[Path generation]: DONE! Generation took {round(generation_time, 2)} s.")
        flat_final_path = generator.get_flat_path(final_path)

        self.finished.emit(flat_final_path)


class ZmqServerThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(object)

    def __init__(self, port, path, subdiv):
        super().__init__()
        self.port = port
        self.path = path
        self.subdiv = subdiv

    def run(self):
        self.progress.emit("[ZMQ Server]: Starting ZMQ Server...")
        client = RemoteAPIClient('localhost', self.port)
        self.progress.emit("[ZMQ Server]: Connected!")
        sim = client.getObject('sim')

        options = int("01001", 2)
        sim.createPath(self.path, options, self.subdiv, 0.0)
        self.progress.emit("[ZMQ Server]: Path created in CoppeliaSim!")
        self.finished.emit()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.width = 1280
        self.height = 720
        self.title = "3D Topology Viewer"

        self.widget = GLWidget(self)

        self.initUI()

        loop = QTimer(self)
        loop.setInterval(20)
        loop.timeout.connect(self.widget.timerEvent)
        loop.start()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon("textures/icon.ico"))
        self.resize(self.width, self.height)
        self.statusbar = QStatusBar()
        self.statusbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.statusbar.showMessage("Mouse wheel button (MWB) to rotate. Shift + MWB to move. Scroll to zoom. Select with left mouse button (LMB). Ctrl + LMB to select multiple.")
        self.setStatusBar(self.statusbar)

        central_widget = QWidget()
        gui_layout = QVBoxLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        gui_layout.addWidget(self.widget)


class GLWidget(QOpenGLWidget):

    def __init__(self, parent):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent)
        self.init_ui()

        # multisample
        format = QSurfaceFormat()
        format.setSamples(16)
        self.setFormat(format)

        self.args = self.parse_args()
        self.generator_thread = None
        self.zmq_thread = None

        self.generated_path = None

        obj = "exported"
        self.object = WfObject(obj)
        self.add_notification(f"[3D Viewer]: Object loaded.")
        self.camera = Camera(pyrr.Vector3([-1.0, 1.0, 0.0]), pyrr.Vector3([0.0, 0.3, 0.0]))

        self.selected_triangles = set()
        self.selected_colors = set()

        self.pressed_btn = None
        self.last_mouse_pos_middle = (0, 0)
        self.pressed_key = None

        self.picking_mode = 0
        self.last_mouse_pos_left = (0, 0)
        self.picking_pos_pixel = None
        self.picking_pos_pixels = None
        self.first_click = True

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def parse_args(self):
        CLI = argparse.ArgumentParser()
        CLI.add_argument(
            "port",
            type=int
        )
        CLI.add_argument(
            "gw",
            type=float
        )
        CLI.add_argument(
            "gh",
            type=float
        )
        CLI.add_argument(
            "--sp",
            nargs="*",
            type=float,
            default=[0.0, 0.0, 0.0]
        )
        CLI.add_argument(
            "--so",
            nargs="*",
            type=float,
            default=[0.0, 0.0, 0.0, 1.0]
        )
        return CLI.parse_args()

    def handle_generator_finish(self, result_path):
        self.generated_path = result_path
        self.apply_to_zmq_button.setEnabled(True)
        self.export_as_txt_button.setEnabled(True)

    def handle_zmq_server_finish(self):
        pass

    def handle_generator_progress(self, progress_text):
        self.add_notification(progress_text)

    def handle_zmq_progress(self, progress_text):
        self.add_notification(progress_text)

    def init_ui(self):
        self.generate_button = QPushButton(self)
        self.generate_button.setText("Generate path")
        self.generate_button.clicked.connect(self.create_generator_thread)
        self.generate_button.setToolTip("Generate path for selected faces.")
        self.generate_button.setGeometry(20, 20, 150, 30)
        self.generate_button.setEnabled(False)

        self.apply_to_zmq_button = QPushButton(self)
        self.apply_to_zmq_button.setText("Create path in CoppeliaSim")
        self.apply_to_zmq_button.clicked.connect(self.create_zmq_thread)
        self.apply_to_zmq_button.setToolTip("Create path in CoppeliaSim.")
        self.apply_to_zmq_button.setGeometry(20, 70, 150, 30)
        self.apply_to_zmq_button.setEnabled(False)

        self.export_as_txt_button = QPushButton(self)
        self.export_as_txt_button.setText("Export path as TXT")
        self.export_as_txt_button.clicked.connect(self.export_as_txt)
        self.export_as_txt_button.setToolTip("Export generated path as .txt file.")
        self.export_as_txt_button.setGeometry(20, 120, 150, 30)
        self.export_as_txt_button.setEnabled(False)


        self.noti_text_edit = QTextEdit(self)
        self.noti_text_edit.setGeometry(1280 - 400, 720 - 300, 400, 250)

        self.add_notification("[3D Viewer]: Starting application...")

        self.noti_text_edit.setReadOnly(True)
        self.noti_text_edit.setStyleSheet("""QTextEdit{ font-style: bold; background: transparent; border: none; font-size: 14px; color: white; }""")

    def add_notification(self, noti_text):
        self.noti_text_edit.append(noti_text)
        noti_cursor = self.noti_text_edit.textCursor()
        noti_cursor.movePosition(QTextCursor.End)
        self.noti_text_edit.setTextCursor(noti_cursor)
        self.noti_text_edit.ensureCursorVisible()

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.initialize_grid()
        self.generate_triangle_colors_for_selection_buffer()
        self.generate_triangle_colors()

        self.compile_shader()
        self.get_uniform_locations()
        #self.bind_model_texture()
        self.update_projection(45, self.width() / self.height())

    def initialize_grid(self):
        self.grid_vertices2 = np.array([0, 0, 0, 0, 0,
                                       1024, 0, 1024, 0, 0,
                                       1024, 1024, 1024, 0, 1024,
                                       0, 1024, 0, 0, 1024], dtype=np.float32)
        self.grid_vertices = np.array([0, 0, -1024, 0, -1024,
                                       1024, 0, 1024, 0, -1024,
                                       1024, 1024, 1024, 0, 1024,
                                       0, 1024, -1024, 0, 1024], dtype=np.float32)

    def generate_triangle_colors_for_selection_buffer(self):
        # triangle count
        triangle_count = len(self.object.vertices) // self.object.get_vertex_length() // 3

        s = []
        i = 5
        #set color to vertices of all triangles. i - triangle index
        for t in range(0, triangle_count):
            r = (i & 0x000000FF) >> 0
            g = (i & 0x0000FF00) >> 8
            b = (i & 0x00FF0000) >> 16
            s.extend([r/255.0, g/255.0, b/255.0] * 3)
            i += 5

        print("tri count: ", triangle_count)
        self.vertex_selection_colors = np.array(s, dtype=np.float32)

    def generate_triangle_colors(self):
        # vertex count
        vertex_count = len(self.object.vertices) // self.object.get_vertex_length()
        self.vertex_colors = np.array(unsel_color * vertex_count, dtype=np.float32)

    def compile_shader(self):
        self.shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                     compileShader(geometry_src, GL_GEOMETRY_SHADER),
                                     compileShader(fragment_src, GL_FRAGMENT_SHADER))
        # VAO, VBO
        self.VAO = glGenVertexArrays(1)

        # Bind obj VAO, VBO, EBO
        glBindVertexArray(self.VAO)

        self.VBOs = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[0])
        glBufferData(GL_ARRAY_BUFFER, self.object.vertices.nbytes, self.object.vertices, GL_STATIC_DRAW)

        p = 0
        # # obj textures
        # if self.object.has_uvs():
        #     glEnableVertexAttribArray(1)
        #     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * self.object.get_vertex_length(), ctypes.c_void_p(0))
        #     p += 8

        # obj normals
        if self.object.has_normals():
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 4 * self.object.get_vertex_length(), ctypes.c_void_p(p))
            p += 12

        # obj vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * self.object.get_vertex_length(), ctypes.c_void_p(p))

        # Object colors VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[1])
        glBufferData(GL_ARRAY_BUFFER, self.vertex_colors.nbytes, self.vertex_colors, GL_DYNAMIC_DRAW)

        # Object colors
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))

        # Object color selection colors VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[2])
        glBufferData(GL_ARRAY_BUFFER, self.vertex_selection_colors.nbytes, self.vertex_selection_colors, GL_STATIC_DRAW)

        # Object color selection colors
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))

        # Grid VAO
        self.grid_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.grid_VAO)

        # Grid Vertex Buffer Object
        grid_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, grid_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.grid_vertices.nbytes, self.grid_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 5, ctypes.c_void_p(0))

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * 5, ctypes.c_void_p(8))

        # Color picking texture
        self.picking_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.picking_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width(), self.height(), 0, GL_RGB, GL_FLOAT, None)
        # Color picking FBO
        depth_buff = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buff)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width(), self.height())

        self.FBO = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.FBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.picking_texture, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buff)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        glUseProgram(self.shader)

        self.grid_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))

    def get_uniform_locations(self):
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.model_loc = glGetUniformLocation(self.shader, "model")

        self.frag_switcher_loc = glGetUniformLocation(self.shader, "frag_switcher")
        self.buffer_switcher_loc = glGetUniformLocation(self.shader, "buffer_switcher")

        self.camera_x = glGetUniformLocation(self.shader, "camera_x")
        self.camera_y = glGetUniformLocation(self.shader, "camera_y")
        self.camera_z = glGetUniformLocation(self.shader, "camera_z")

    def bind_model_texture(self):
        self.object_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.object_texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.object.texture.width, self.object.texture.height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, self.object.texture_data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.update_projection(45, width / height)

    def update_projection(self, fov, aspect, near=0.1, far=100):
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(fov, aspect, near, far)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)

    def paintGL(self):
        glClearColor(0.060, 0.103, 0.154, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # update look-at matrix
        self.look_at = self.camera.get_look_at_view()
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.look_at)

        # set camera position uniforms for grid fog
        glUniform1f(self.camera_x, self.camera.eye[0])
        glUniform1f(self.camera_y, self.camera.eye[1])
        glUniform1f(self.camera_z, self.camera.eye[2])

        # set switcher uniform to main buffer
        glUniform1i(self.buffer_switcher_loc, 0)

        # set switcher uniform to draw object
        glUniform1i(self.frag_switcher_loc, 0)


        # draw object
        glBindVertexArray(self.VAO)
        #glBindTexture(GL_TEXTURE_2D, self.object_texture)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.object.model)
        glDrawArrays(GL_TRIANGLES, 0, len(self.object.indices))

        # set switcher uniform to draw grid
        glUniform1i(self.frag_switcher_loc, 1)

        # draw grid
        glBindVertexArray(self.grid_VAO)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.grid_pos)
        glDrawArrays(GL_QUADS, 0, len(self.grid_vertices))

        # if selecting color
        if self.picking_pos_pixel is not None or self.picking_pos_pixels is not None:
            # set switcher uniform to color selection buffer
            glUniform1i(self.buffer_switcher_loc, 1)

            # bind color selection buffer
            glBindFramebuffer(GL_FRAMEBUFFER, self.FBO)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # set switcher uniform to draw object
            glUniform1i(self.frag_switcher_loc, 0)

            # draw object
            glBindVertexArray(self.VAO)
            glBindTexture(GL_TEXTURE_2D, self.picking_texture)
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.object.model)
            glDrawArrays(GL_TRIANGLES, 0, len(self.object.indices))

            # select color
            if self.picking_pos_pixel is not None:
                self.pick_color_pixel()
            if self.picking_pos_pixels is not None:
                self.pick_color_pixels()

            # bind main buffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def create_generator_thread(self):

        # # !!!TEST!!!
        # triangle_count = len(self.object.vertices) // self.object.get_vertex_length() // 3
        # s = []
        # for t in range(0, triangle_count):
        #     s.append(t)
        #
        # triangles = []
        # for idx in s:
        #     triangles.append(self.get_triangle_by_index(idx))

        triangles = []
        for idx in self.selected_triangles:
            triangles.append(self.get_triangle_by_index(idx))

        if len(triangles) != 0:
            self.generator_thread = GeneratorThread(triangles, self.args.gw, self.args.gh, self.args.sp, self.args.so)
            self.generator_thread.progress.connect(self.handle_generator_progress)
            self.generator_thread.finished.connect(self.handle_generator_finish)
            self.generator_thread.finished.connect(self.generator_thread.quit)
            self.generator_thread.finished.connect(self.generator_thread.deleteLater)
            self.generator_thread.start()

    def create_zmq_thread(self):

        self.zmq_thread = ZmqServerThread(self.args.port, self.generated_path, 100)
        self.zmq_thread.progress.connect(self.handle_zmq_progress)
        self.zmq_thread.finished.connect(self.handle_zmq_server_finish)
        self.zmq_thread.finished.connect(self.zmq_thread.quit)
        self.zmq_thread.finished.connect(self.zmq_thread.deleteLater)
        self.zmq_thread.start()

    def export_as_txt(self):
        name, _ = QFileDialog().getSaveFileName(self, "Export path", "", "Text files (*.txt)")
        if name:
            file = open(name, "w")
            text = str(self.generated_path)
            file.write(text)
            file.close()

    def get_triangle_offsets_by_idx(self, idx):
        l = self.object.get_vertex_length()
        vert_offset = idx * 3 * l + (l - 3)
        off1 = (vert_offset, vert_offset + 3)
        off2 = (vert_offset + 3 + (l - 3), vert_offset + 6 + (l - 3))
        off3 = (vert_offset + 6 + 2 * (l - 3), vert_offset + 9 + 2 * (l - 3))
        return off1, off2, off3

    def get_triangle_by_index(self, idx):
        off1, off2, off3 = self.get_triangle_offsets_by_idx(idx)
        p1 = self.object.vertices[off1[0]: off1[1]]
        p2 = self.object.vertices[off2[0]: off2[1]]
        p3 = self.object.vertices[off3[0]: off3[1]]
        return [p1, p2, p3]

    def add_triangle_to_selected(self, idx):
        if idx not in self.selected_triangles:
            self.selected_triangles.add(idx)

            glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[1])
            glBufferSubData(GL_ARRAY_BUFFER, 4 * idx * 9, 4 * 9, sel_triangle_colors)

    def select_triangle(self, idx):
        if idx not in self.selected_triangles:
            self.selected_triangles.add(idx)
            if not self.generate_button.isEnabled():
                self.generate_button.setEnabled(True)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[1])
            glBufferSubData(GL_ARRAY_BUFFER, 4 * idx * 9, 4 * 9, sel_triangle_colors)

    def deselect_triangle(self, idx):
        if idx in self.selected_triangles:
            self.selected_triangles.remove(idx)
            if len(self.selected_triangles) == 0:
                self.generate_button.setEnabled(False)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[1])
            glBufferSubData(GL_ARRAY_BUFFER, 4 * idx * 9, 4 * 9, unsel_triangle_colors)

    def get_selection_colors(self, read_pixels):
        colors = set()
        length = len(read_pixels) // 3
        i = 0
        while i != length:
            idx = i * 3
            colors.add((read_pixels[idx], read_pixels[idx + 1], read_pixels[idx + 2]))
            i += 1
        return colors

    def read_pixels(self, pos, w, h):
        x, y = pos
        y = self.height() - y
        data = glReadPixels(x - w, y - h, w, h, GL_RGB, GL_UNSIGNED_BYTE)

        return data

    def pick_color_pixel(self):
        data = self.read_pixels(self.picking_pos_pixel, 1, 1)
        idx = int((data[0] + data[1] * 256 + data[2] * 256 * 256) / 5) - 1

        if idx in self.selected_triangles:
            if idx != -1:
                self.deselect_triangle(idx)
            if self.first_click:
                self.first_click = False
                self.picking_mode = 0
        else:
            if idx != -1:
                self.select_triangle(idx)
            if self.first_click:
                self.first_click = False
                self.picking_mode = 1

        self.picking_pos_pixel = None

    def pick_color_pixels(self):
        data = self.read_pixels(self.picking_pos_pixels, 12, 12)
        colors = self.get_selection_colors(data)

        ctr_data = self.read_pixels(self.picking_pos_pixels, 1, 1)
        ctr_idx = int((ctr_data[0] + ctr_data[1] * 256 + ctr_data[2] * 256 * 256) / 5) - 1

        if ctr_idx in self.selected_triangles:
            if self.first_click:
                self.first_click = False
                self.picking_mode = 0
        else:
            if self.first_click:
                self.first_click = False
                self.picking_mode = 1

        for c in colors:
            idx = int((c[0] + c[1] * 256 + c[2] * 256 * 256) / 5) - 1
            if idx != -1:
                if self.picking_mode == 0:
                    self.deselect_triangle(idx)
                else:
                    self.select_triangle(idx)

        self.picking_pos_pixels = None

    def mousePressEvent(self, event: QMouseEvent):
        self.pressed_btn = event.button()
        if self.pressed_btn == Qt.MouseButton.MidButton:
            self.last_mouse_pos_middle = (event.x(), event.y())
        elif self.pressed_btn == Qt.MouseButton.LeftButton and self.pressed_key == Qt.Key_Control:
            self.picking_pos_pixels = (event.x(), event.y())
            self.last_mouse_pos_left = (event.x(), event.y())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.pressed_btn == Qt.MouseButton.LeftButton and not self.pressed_key == Qt.Key_Control:
            self.picking_pos_pixel = (event.x(), event.y())
        self.pressed_btn = None
        self.first_click = True

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pressed_btn == Qt.MouseButton.MidButton:
            x_off = event.x() - self.last_mouse_pos_middle[0]
            y_off = event.y() - self.last_mouse_pos_middle[1]
            self.last_mouse_pos_middle = (event.x(), event.y())

            if self.pressed_key is None:
                self.camera.process_mouse_rotation(x_off, y_off, self.width(), self.height())
            elif self.pressed_key == Qt.Key_Shift:
                self.camera.process_mouse_movement(x_off, y_off, self.width(), self.height())

        elif self.pressed_btn == Qt.MouseButton.LeftButton and self.pressed_key == Qt.Key_Control:
            len = np.sqrt((event.x() - self.last_mouse_pos_left[0]) ** 2 + (event.y() - self.last_mouse_pos_left[1]) ** 2)
            if len > 12:
                self.last_mouse_pos_left = (event.x(), event.y())
                self.picking_pos_pixels = (event.x(), event.y())

    def keyPressEvent(self, event: QKeyEvent):
        self.pressed_key = event.key()
        if self.pressed_key == Qt.Key_Slash:
            self.camera.reset()
        elif self.pressed_key == Qt.Key_G:
            self.create_generator_thread()
        elif self.pressed_key == Qt.Key_H:
            self.create_zmq_thread()

    def keyReleaseEvent(self, event: QKeyEvent):
        self.pressed_key = None

    def wheelEvent(self, event: QWheelEvent):
        degrees = event.angleDelta().y() / 8
        steps = degrees / 15
        self.camera.process_wheel_input(steps)

    def timerEvent(self):
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
