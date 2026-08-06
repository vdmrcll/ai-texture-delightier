"""Small embedded OpenGL OBJ texture viewer used by the GUI."""

import math
import os
import ctypes

# pyopengltk embeds a GLX context in Tk. On Wayland desktops PyOpenGL can
# otherwise select EGL first, which makes pyopengltk fail before the widget is
# created. Tk uses XWayland here, so GLX is the compatible backend.
if os.name != "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
from PIL import Image

try:
    from OpenGL import GL
    from OpenGL.GL import shaders
    from pyopengltk import OpenGLFrame
except Exception:  # Keep the main application importable when GL is unavailable.
    GL = None
    shaders = None
    OpenGLFrame = object

OPENGL_AVAILABLE = GL is not None and shaders is not None and OpenGLFrame is not object


class OBJMesh:
    """Load only OBJ geometry and UVs. MTL files are intentionally ignored."""

    def __init__(self, path):
        self.path = path
        positions = []
        texcoords = []
        position_indices = []
        uv_indices = []

        # OBJ is a text format, but decoding every line to Unicode is
        # unnecessary for its numeric data. Keeping the parser byte-based
        # also avoids building expanded Python vertex tuples for every face.
        with open(path, "rb") as handle:
            for raw_line in handle:
                parts = raw_line.split()
                if not parts or parts[0].startswith(b"#"):
                    continue
                if parts[0] == b"v" and len(parts) >= 4:
                    positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == b"vt" and len(parts) >= 3:
                    texcoords.append((float(parts[1]), float(parts[2])))
                elif parts[0] == b"f" and len(parts) >= 4:
                    face = []
                    for token in parts[1:]:
                        fields = token.split(b"/")
                        if len(fields) < 2 or not fields[0] or not fields[1]:
                            raise ValueError("The OBJ must provide UV coordinates for every face.")

                        position_index = int(fields[0])
                        uv_index = int(fields[1])
                        position_index = position_index - 1 if position_index > 0 else len(positions) + position_index
                        uv_index = uv_index - 1 if uv_index > 0 else len(texcoords) + uv_index
                        if not (0 <= position_index < len(positions) and 0 <= uv_index < len(texcoords)):
                            raise ValueError("The OBJ contains an out-of-range vertex or UV index.")
                        face.append((position_index, uv_index))

                    # Triangulate polygons as a fan, storing only indices until
                    # all source vertices have been parsed.
                    for index in range(1, len(face) - 1):
                        for position_index, uv_index in (face[0], face[index], face[index + 1]):
                            position_indices.append(position_index)
                            uv_indices.append(uv_index)

        if not position_indices:
            raise ValueError("The OBJ does not contain any faces.")
        if len(uv_indices) != len(position_indices):
            raise ValueError("The OBJ does not contain usable UV coordinates.")

        source_positions = np.asarray(positions, dtype=np.float32)
        source_uvs = np.asarray(texcoords, dtype=np.float32)
        points = source_positions[np.asarray(position_indices, dtype=np.int32)]
        uvs = source_uvs[np.asarray(uv_indices, dtype=np.int32)]
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        radius = float(np.max(np.linalg.norm(points - center, axis=1)))
        if radius <= 1e-8:
            raise ValueError("The OBJ has zero-sized geometry.")
        points = (points - center) / radius
        self.vertex_data = np.concatenate((points, uvs), axis=1)
        self.vertex_count = len(position_indices)

    @staticmethod
    def _parse_vertex(token, position_count, uv_count):
        fields = token.split("/")
        if len(fields) < 2 or not fields[1]:
            raise ValueError("The OBJ must provide UV coordinates for every face.")

        def resolve(value, count):
            index = int(value)
            return index - 1 if index > 0 else count + index

        position_index = resolve(fields[0], position_count)
        uv_index = resolve(fields[1], uv_count)
        if not (0 <= position_index < position_count and 0 <= uv_index < uv_count):
            raise ValueError("The OBJ contains an out-of-range vertex or UV index.")
        return position_index, uv_index


class TextureViewport(OpenGLFrame):
    """Unlit, UV-textured OBJ viewport with orbit and zoom controls."""

    VERTEX_SHADER = """
        #version 120
        attribute vec3 position;
        attribute vec2 texcoord;
        uniform mat4 transform;
        varying vec2 uv;
        void main() { uv = texcoord; gl_Position = transform * vec4(position, 1.0); }
    """
    FRAGMENT_SHADER = """
        #version 120
        uniform sampler2D texture_map;
        varying vec2 uv;
        void main() { gl_FragColor = texture2D(texture_map, uv); }
    """

    def __init__(self, master, **kwargs):
        if GL is None:
            raise RuntimeError("OpenGL preview dependencies are not installed.")
        super().__init__(master, **kwargs)
        self.mesh = None
        self.mesh_path = None
        self.texture_id = None
        self.program = None
        self.vertex_buffer = None
        self.yaw = 25.0
        self.pitch = 15.0
        self.distance = 3.0
        self.fov = 50.0
        self._drag_start = None
        self.bind("<ButtonPress-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._end_drag)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda event: self._zoom(-1))
        self.bind("<Button-5>", lambda event: self._zoom(1))

    def initgl(self):
        GL.glClearColor(0.035, 0.047, 0.065, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program = shaders.compileProgram(
            shaders.compileShader(self.VERTEX_SHADER, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
        )

    def load_mesh(self, path):
        normalized_path = os.path.abspath(path)
        if self.mesh is not None and self.mesh_path == normalized_path and self.vertex_buffer is not None:
            # Preview texture changes should not reparse the OBJ or reset the
            # camera. The mesh is independent from the texture bound below.
            return

        self.mesh = OBJMesh(path)
        self.vertex_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.mesh.vertex_data.nbytes, self.mesh.vertex_data, GL.GL_STATIC_DRAW)
        self.mesh_path = normalized_path
        self.reset_view()
        self.redraw()

    def load_texture(self, path):
        if not path or not os.path.isfile(path):
            self._delete_texture()
            return
        with Image.open(path) as source:
            image = source.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            width, height = image.size
            pixels = image.tobytes()
        if self.texture_id is None:
            self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
        self.redraw()

    def _delete_texture(self):
        if self.texture_id is not None:
            GL.glDeleteTextures([self.texture_id])
            self.texture_id = None
        self.redraw()

    def reset_view(self):
        self.yaw, self.pitch, self.distance = 25.0, 15.0, 3.0
        self.redraw()

    def set_fov(self, fov):
        self.fov = max(20.0, min(120.0, float(fov)))
        self.redraw()

    def _begin_drag(self, event):
        self._drag_start = (event.x, event.y, self.yaw, self.pitch)

    def _drag(self, event):
        if self._drag_start:
            x, y, yaw, pitch = self._drag_start
            self.yaw = yaw - (event.x - x) * 0.5
            self.pitch = max(-89.0, min(89.0, pitch + (event.y - y) * 0.5))
            self.redraw()

    def _end_drag(self, _event):
        self._drag_start = None

    def _wheel(self, event):
        self._zoom(-1 if event.delta > 0 else 1)

    def _zoom(self, direction):
        self.distance = max(1.2, min(12.0, self.distance * (0.88 if direction < 0 else 1.14)))
        self.redraw()

    def redraw(self):
        if not self.winfo_exists() or not self.program:
            return
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.mesh is None or self.texture_id is None:
            self.tkSwapBuffers()
            return

        aspect = width / height
        projection = self._perspective(math.radians(self.fov), aspect, 0.1, 100.0)
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        eye = np.array([
            self.distance * math.cos(pitch) * math.sin(yaw),
            self.distance * math.sin(pitch),
            self.distance * math.cos(pitch) * math.cos(yaw),
        ], dtype=np.float32)
        view = self._look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        transform = projection @ view

        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "transform"), 1, GL.GL_TRUE, transform)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "texture_map"), 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
        stride = 5 * 4
        position = GL.glGetAttribLocation(self.program, "position")
        texcoord = GL.glGetAttribLocation(self.program, "texcoord")
        GL.glEnableVertexAttribArray(position)
        GL.glVertexAttribPointer(position, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(texcoord)
        GL.glVertexAttribPointer(texcoord, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.mesh.vertex_count)
        GL.glDisableVertexAttribArray(position)
        GL.glDisableVertexAttribArray(texcoord)
        GL.glUseProgram(0)
        self.tkSwapBuffers()

    @staticmethod
    def _perspective(fov, aspect, near, far):
        f = 1.0 / math.tan(fov / 2.0)
        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2 * far * near) / (near - far)
        result[3, 2] = -1.0
        return result

    @staticmethod
    def _look_at(eye, target, up):
        forward = target - eye
        forward /= np.linalg.norm(forward)
        side = np.cross(forward, up)
        side /= np.linalg.norm(side)
        true_up = np.cross(side, forward)
        result = np.eye(4, dtype=np.float32)
        result[0, :3], result[1, :3], result[2, :3] = side, true_up, -forward
        result[0, 3], result[1, 3], result[2, 3] = -np.dot(side, eye), -np.dot(true_up, eye), np.dot(forward, eye)
        return result
