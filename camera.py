from pyrr import Vector3, vector, vector3, vector4, matrix44
from math import pi


class Camera:
    def __init__(self, pos, focus_pos):
        self.eye = self.def_eye = pos
        self.target = self.def_target = focus_pos
        self.up = Vector3([0.0, 1.0, 0.0])
        self.right = Vector3([1.0, 0.0, 0.0])

        self.look_at_matrix = matrix44.create_look_at(self.eye, self.target, self.up)

    def reset(self):
        self.update_view(self.def_eye, self.def_target, self.up)
        #self.update_view([0, 1.0, 1.0], [0, 1.0, 0.0], self.up)

    def process_mouse_rotation(self, x_off, y_off, width, height):
        delta_x = 2 * pi / width
        delta_y = pi / height

        # handle the dir_vector = up_vector problem
        cos_angle = vector.dot(self.get_dir_vector(), self.up)
        if (cos_angle < -0.99 and -y_off > 0) or (cos_angle > 0.99 and -y_off < 0):
            delta_y = 0

        x_angle = -x_off * delta_x
        y_angle = -y_off * delta_y

        self.rotate(x_angle, y_angle)

    def process_mouse_movement(self, x_off, y_off, width, height):
        length_multiplier = vector.length(self.target - self.eye) / 50
        x_dist = x_off / width * 30 * length_multiplier
        y_dist = y_off / height * 30 * length_multiplier

        eye_hor = self.eye + self.get_right_vector() * -x_dist
        eye_final = eye_hor + self.get_up_vector() * y_dist
        target_hor = self.target + self.get_right_vector() * -x_dist
        target_final = target_hor + self.get_up_vector() * y_dist

        self.update_view(eye_final, target_final, self.up)

    def process_wheel_input(self, steps):
        new_eye = self.eye + self.get_dir_vector() * -steps * 0.5
        if vector.squared_length(new_eye - self.target) >= 0.2:
            self.update_view(new_eye, self.target, self.up)

    def rotate(self, x_angle, y_angle):
        vec4_eye = vector4.create_from_vector3(self.eye, 1)
        vec4_target = vector4.create_from_vector3(self.target, 1)

        rotX = matrix44.create_identity()
        rotX = matrix44.multiply(rotX, matrix44.create_from_axis_rotation(self.up, x_angle))
        vec4_eye = matrix44.apply_to_vector(rotX, (vec4_eye - vec4_target)) + vec4_target

        rotY = matrix44.create_identity()
        rotY = matrix44.multiply(rotY, matrix44.create_from_axis_rotation(self.get_right_vector(), y_angle))
        final_position = matrix44.apply_to_vector(rotY, (vec4_eye - vec4_target)) + vec4_target

        e, ew = vector3.create_from_vector4(final_position)
        t, tw = vector3.create_from_vector4(vec4_target)
        self.update_view(e, t, self.up)

    def update_view(self, eye, target, up):
        self.eye = eye
        self.target = target
        self.up = up
        self.look_at_matrix = matrix44.create_look_at(self.eye, self.target, self.up)

    def get_right_vector(self):
        v, w = vector3.create_from_vector4(self.look_at_matrix.T[0])
        return v

    def get_dir_vector(self):
        v, w = vector3.create_from_vector4(self.look_at_matrix.T[2])
        return v

    def get_up_vector(self):
        v, w = vector3.create_from_vector4(self.look_at_matrix.T[1])
        return v

    def get_look_at_view(self):
        return self.look_at_matrix

