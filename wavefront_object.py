import pywavefront
import numpy as np
import pyrr
from os.path import exists
from PIL import Image


class WfObject:
    def __init__(self, filename):
        self.vertices, self.indices, self.format = self.load_mesh("{0}.obj".format(filename))
        #self.texture, self.texture_data = self.load_texture("{0}.png".format(filename))
        #self.model = pyrr.matrix44.create_from_x_rotation(np.pi / 2)
        self.model = pyrr.matrix44.create_from_eulers([np.pi / 2, np.pi, 0.0])

    def load_mesh(self, file):
        scene = pywavefront.Wavefront(file, create_materials=True, collect_faces=True)

        mesh = scene.mesh_list[0]
        # vt, vn, v
        vertices = mesh.materials[0].vertices
        # flattened indices
        indices = [vert for triangle in mesh.faces for vert in triangle]

        return np.array(vertices, dtype="float32"), np.array(indices, dtype="uint32"), mesh.materials[0].vertex_format

    # def load_texture(self, filename):
    #     texture = Image.open("textures/missing.png")
    #     if exists(filename):
    #         texture = Image.open(filename)
    #     texture = texture.transpose(Image.FLIP_TOP_BOTTOM)
    #     texture_data = texture.convert("RGBA").tobytes()
    #     return texture, texture_data

    def has_uvs(self):
        return "T2F" in self.format

    def has_normals(self):
        return "N3F" in self.format

    def get_vertex_length(self):
        return 3 + self.has_uvs() * 2 + self.has_normals() * 3


