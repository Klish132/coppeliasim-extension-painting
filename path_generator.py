import math
import pyrr.vector3
from pyrr import vector3, Vector3, vector, matrix33, quaternion
import numpy as np
from itertools import chain

from wavefront_object import WfObject


def transform_points(m, points):
    return [matrix33.apply_to_vector(m, p) for p in points]


class Triangle:
    def __init__(self, points):
        self.p1 = Vector3(points[0])
        self.p2 = Vector3(points[1])
        self.p3 = Vector3(points[2])

        self.cross = self.get_cross()
        self.normal = self.get_normal()
        self.area = self.get_area()

    def get_points(self):
        return self.p1, self.p2, self.p3

    def get_edge(self, num):
        if num == 0:
            return self.p2 - self.p1
        elif num == 1:
            return self.p3 - self.p1
        elif num == 2:
            return self.p3 - self.p2
        else:
            return Vector3([0, 0, 0])

    def get_cross(self):
        return vector3.cross(self.p2 - self.p1, self.p3 - self.p1)

    def get_normal(self):
        cross = self.get_cross()
        if vector.length(cross) == 0:
            return cross
        return vector.normalise(self.get_cross())

    def get_area(self):
        return vector.length(self.get_cross()) / 2

    def compare_vertices(self, v, t):
        #return all(np.isclose(v, t.p1, rtol=0.0000001)) or all(np.isclose(v, t.p2, rtol=0.0000001)) or all(np.isclose(v, t.p3, rtol=0.0000001))
        return np.array_equal(v, t.p1) or np.array_equal(v, t.p2) or np.array_equal(v, t.p3)

    def shares_edge(self, t):
        p1 = self.compare_vertices(self.p1, t)
        p2 = self.compare_vertices(self.p2, t)
        p3 = self.compare_vertices(self.p3, t)
        return p1 if (p1 == p2) else p3

    def vert2string(self, v):
        return np.array2string(v, suppress_small=True, separator=",")

    def __str__(self):
        return f"({self.vert2string(self.p1)},{self.vert2string(self.p2)},{self.vert2string(self.p3)}),"

    def __repr__(self):
        # return "T()"
        return self.__str__()


class TrianglePatch:
    def __init__(self, triangle_list):
        self.triangles = triangle_list
        self.average_normal = self.get_weighted_average_normal()
        self.axes = self.get_orthogonal_axes()

    def transform_triangles(self, triangles, axes):
        new_triangles = []
        for t in triangles:
            new_t = Triangle(transform_points(axes, t.get_points()))
            new_triangles.append(new_t)
        return new_triangles

    def get_weighted_average_normal(self):
        area_normal_sum = 0
        area_sum = 0
        for t in self.triangles:
            area_sum += t.area
            area_normal_sum += t.area * t.normal
        v = area_normal_sum / area_sum
        n = pyrr.vector.normalise(v)
        return n

    def get_orthogonal_axes(self):
        x = self.average_normal
        up = [0.0, 0.0, 1.0] if not np.array_equal(x, [0.0, 0.0, 1.0]) and \
                                not np.array_equal(x, [0.0, 0.0, -1.0]) else [1.0, 0.0, 0.0]
        z = vector.normalise(pyrr.vector3.cross(x, up))
        y = vector.normalise(pyrr.vector3.cross(x, z))
        return np.array([x, y, z])

    def get_plane_point(self, axis, plane_point, test_point):
        x = axis[0] * (test_point[0] - plane_point[0])
        y = axis[1] * (test_point[1] - plane_point[1])
        z = axis[2] * (test_point[2] - plane_point[2])
        x = 0 if math.isclose(x, 0, abs_tol=0.01) else x
        y = 0 if math.isclose(y, 0, abs_tol=0.01) else y
        z = 0 if math.isclose(z, 0, abs_tol=0.01) else z
        res = 0 if math.isclose(x + y + z, 0, abs_tol=0.01) else x + y + z
        return res

    def get_triangle_plane_intersections(self, t, right_off):
        right_axis = self.axes[1]
        point_on_axis = right_axis * right_off
        p1_sign = np.sign(self.get_plane_point(right_axis, point_on_axis, t.p1))
        p2_sign = np.sign(self.get_plane_point(right_axis, point_on_axis, t.p2))
        p3_sign = np.sign(self.get_plane_point(right_axis, point_on_axis, t.p3))

        # equal signs means all points are on the same side
        if p1_sign == p2_sign == p3_sign:
            return []

        # if any points are on plane< return them
        on_plane = []
        for s in zip([p1_sign, p2_sign, p3_sign], [t.p1, t.p2, t.p3]):
            if s[0] == 0:
                on_plane.append(s[1])
        if len(on_plane) != 0:
            return on_plane

        p1p2 = p1p3 = False
        points = []
        # p1p2 edge intersected
        if p1_sign != p2_sign:
            p1p2 = True
            points.append(self.get_edge_plane_intersection(t.p1, t.p2, right_off))
        # p1p3 edge intersected
        if p1_sign != p3_sign:
            p1p3 = True
            points.append(self.get_edge_plane_intersection(t.p1, t.p3, right_off))
        # if either of previous two edges isn't intersected, means p2p3 is intersected
        if p1p2 * p1p3 == False:
            points.append(self.get_edge_plane_intersection(t.p2, t.p3, right_off))
        return points

    def get_edge_plane_intersection(self, p1, p2, right_off):
        plane_normal = self.axes[1]
        plane_point = plane_normal * right_off
        return p1 + (vector.dot(plane_normal, plane_point - p1) / vector.dot(plane_normal, p2 - p1)) * (p2 - p1)

    def get_bounding_sides(self, axis, triangles):
        max_right = -np.inf
        min_right = np.inf
        for t in triangles:
            p1, p2, p3 = t.get_points()
            # [0, 1, 0] self.axes[1]
            max_right = max(self.get_axis_maximum(axis, p1, p2, p3), max_right)
            min_right = min(self.get_axis_minimum(axis, p1, p2, p3), min_right)
        return max_right, min_right

    def get_axis_maximum(self, axis, p1, p2, p3):
        return max(vector.dot(axis, p1), vector.dot(axis, p2), vector.dot(axis, p3))

    def get_axis_minimum(self, axis, p1, p2, p3):
        return min(vector.dot(axis, p1), vector.dot(axis, p2), vector.dot(axis, p3))

    def intersect_with_cutting_planes(self, step):
        max_right, min_right = self.get_bounding_sides(self.axes[1], self.triangles)
        box_length = np.abs(min_right - max_right)
        step = step if step < box_length else box_length / 2
        curr_right = max_right

        intersection_points = []
        i = 0
        while curr_right > min_right:
            points_on_this_plane = []
            for t in self.triangles:
                new_points = self.get_triangle_plane_intersections(t, curr_right)
                if len(new_points) == 0:
                    continue
                for new_p in new_points:
                    is_in = False
                    for in_p in points_on_this_plane:
                        if all(np.isclose(new_p, in_p, rtol=0.000001)):
                            is_in = True
                            break
                    if is_in:
                        continue
                    points_on_this_plane.append(new_p)

            points_on_this_plane = sorted(points_on_this_plane,
                                          key=lambda p: vector.dot(self.axes[2], p),
                                          reverse=(False + i % 2))
            for p in points_on_this_plane:
                intersection_points.append(p)
            curr_right = curr_right - step
            i += 1

        return intersection_points

    def intersect_triangle_with_ray(self, t, ro, rd):
        det = - vector.dot(rd, t.normal)
        inv_det = 1.0 / det
        p1_ro = ro - t.p1
        rd_p1_ro = vector3.cross(p1_ro, rd)
        u = vector.dot(t.get_edge(1), rd_p1_ro) * inv_det
        v = -vector.dot(t.get_edge(0), rd_p1_ro) * inv_det
        t = vector.dot(p1_ro, t.normal) * inv_det
        return (det >= 1e-6 and t >= 0.0 and u >= 0.0 and v >= 0.0 and (u + v) <= 1.0), t

    def get_paint_points(self, step):
        inter_points = []
        right, left = self.get_bounding_sides(self.axes[1], self.triangles)
        top, bottom = self.get_bounding_sides(self.axes[2], self.triangles)
        front, back = self.get_bounding_sides(self.axes[0], self.triangles)
        print("right, left: ", right, left)
        print("top, bottom: ", top, bottom)

        curr_y, curr_z = left, bottom
        direction = 1

        while True:
            print("y, z: ", curr_y, curr_z)
            if curr_z <= top:
                if left <= curr_y <= right:
                    for triangle in self.triangles:
                        ray_origin = Vector3([front, curr_y, curr_z])
                        is_inter, t = self.intersect_triangle_with_ray(triangle, ray_origin, -self.axes[0])
                        if is_inter:
                            inter_points.append(ray_origin + t * -self.axes[0])
                    curr_y += step * direction
                else:
                    curr_z = curr_z + step * math.sin(np.pi / 3 * (1 if direction < 0 else 2))
                    curr_y = curr_y + step * math.cos(np.pi / 3 * (1 if direction < 0 else 2))
                    direction = -1 * direction
            else:
                break
        print(len(inter_points))
        return inter_points

    def __str__(self):
        stri = ""
        for t in self.triangles:
            stri += str(t)
        return stri

    def __repr__(self):
        return self.__str__()


class Path:
    def __init__(self, points, axes, gun_height):
        self.points = points
        self.axes = axes
        self.quaternions = self.create_quaternions()
        self.gun_height = gun_height
        self.move_points_up()

    def create_quaternions(self):
        quats = []
        for p in self.points:
            x_ident = Vector3([1.0, 0.0, 0.0])
            x_self = -self.axes[0]
            k_cos_theta = vector.dot(x_ident, x_self)
            k = math.sqrt(vector.length(x_ident) * vector.length(x_self))
            x_cross = vector3.cross(x_ident, x_self)

            if k_cos_theta / k == -1:
                q = quaternion.create_from_axis_rotation(self.axes[2], np.pi)
                quats.append(q)
            else:
                q = pyrr.quaternion.create(x_cross[0], x_cross[1], x_cross[2], k_cos_theta + k)
                q = pyrr.quaternion.normalise(q)
                quats.append(q)
        return quats

    def move_points_up(self):
        self.points = [p + self.axes[0] * self.gun_height for p in self.points]

    def transform_points(self, axes):
        self.points = transform_points(axes, self.points)
        qm = pyrr.quaternion.create_from_matrix(axes)
        self.quaternions = [pyrr.quaternion.cross(q, qm) for q in self.quaternions]

    def connect_other_path(self, other):
        self.points += other.points
        self.quaternions += other.quaternions

    def get_start_point(self):
        return self.points[0]

    def get_end_point(self):
        return self.points[len(self.points) - 1]

    def __str__(self):
        flat_zipped = [item for sublist in list(zip(self.points, self.quaternions)) for item in sublist]
        str = ""
        for item in flat_zipped:
            for el in item:
                str += f"{el},"
        return str

    def __repr__(self):
        return self.__str__()


class PathGenerator:
    def __init__(self, triangle_points_list):
        # Triangles
        self.triangles = []
        for points in triangle_points_list:
            t = Triangle(points)
            if not t.area == 0:
                self.triangles.append(t)

        # Orthonormal coordinate system
        self.axes = matrix33.create_identity()
        self.opposite_axes = self.axes * -1
        self.rotation = (0, 0, 0)

    def do_main_routine(self, spray_width, gun_height, start_pos, start_orient):
        self.find_main_axes()
        groups = self.group_triangles_by_orthogonal_axes()
        patches = self.form_patches(groups)
        paths = self.create_paths_from_patches(patches, spray_width, gun_height)
        final_path = self.connect_all_paths(paths, start_pos, start_orient)
        return self.get_flat_path(final_path)

    def create_paths_from_patches(self, patches, spray_width, gun_height):
        paths = []
        for patch in patches:
            points = patch.intersect_with_cutting_planes(spray_width * 2 / 3)
            if not len(points) == 0:
                path = Path(points, patch.axes, gun_height)
                paths.append((patch, path))
        paths = sorted(paths, key=lambda p: p[1].get_start_point()[2])
        return paths

    def connect_all_paths(self, paths, start_pos, start_orient):
        f = paths.pop(0)
        final_path = f[1]
        final_path.points.insert(0, np.array(start_pos))
        final_path.quaternions.insert(0, np.array(start_orient))
        for path in paths:
            final_path.connect_other_path(path[1])
        return final_path

    def get_flat_path(self, path):
        return list(chain.from_iterable(chain.from_iterable(zip(path.points, path.quaternions))))

    def get_m(self, axes):
        sum = 0
        for triangle in self.triangles:
            sum += max(vector.dot(axes[0], triangle.normal),
                       vector.dot(axes[1], triangle.normal),
                       vector.dot(axes[2], triangle.normal)) * triangle.area
        return sum

    def find_main_axes(self):
        current_m = 0
        step = np.pi / 6

        # self.axes = np.array([np.array([-4.33012702e-01, -8.66025404e-01,  2.50000000e-01]),
        #                      np.array([-5.00000000e-01,  5.30287619e-17, -8.66025404e-01]),
        #                      np.array([ 7.50000000e-01, -5.00000000e-01, -4.33012702e-01])])
        # self.rotation = (1.5707963267948966, 5.759586531581287, 4.1887902047863905)
        # 1728
        for p in range(0, 12):
            for r in range(0, 12):
                for y in range(0, 12):
                    axes = matrix33.create_from_eulers([p * step, y * step, r * step])
                    m = self.get_m(axes)
                    if m > current_m:
                        current_m = m
                        self.axes = axes
                        self.rotation = (p * step, y * step, r * step)
        self.opposite_axes = self.axes * -1

    def group_triangles_by_orthogonal_axes(self):
        groups = {}
        for triangle in self.triangles:
            applied_normal = matrix33.apply_to_vector(self.axes, triangle.normal)
            #applied_normal = triangle.normal
            max_idx = max(range(len(applied_normal)), key=np.abs(applied_normal).__getitem__)
            if applied_normal[max_idx] >= 0:
                axis_as_tuple = tuple(self.axes[max_idx])
                if axis_as_tuple not in groups:
                    groups[axis_as_tuple] = []
                groups[axis_as_tuple].append(triangle)
            else:
                axis_as_tuple = tuple(self.opposite_axes[max_idx])
                if axis_as_tuple not in groups:
                    groups[axis_as_tuple] = []
                groups[axis_as_tuple].append(triangle)
        return groups

    def split_group_into_patches(self, group):
        # for naming clarity
        disconnected = group
        connected = [disconnected.pop(0)]

        conn_new = True
        while conn_new is not False:
            conn_new = False
            idxes_to_pop = []
            i = 0
            for triangle in disconnected:
                for conn_triangle in connected:
                    if triangle.shares_edge(conn_triangle):
                        connected.append(triangle)
                        idxes_to_pop.append(i)
                        conn_new = True
                        break
                i += 1
            rem = 0
            for idx in idxes_to_pop:
                disconnected.pop(idx - rem)
                rem += 1

        result = [connected]
        if len(disconnected) > 1:
            for res in self.split_group_into_patches(disconnected):
                result.append(res)
        elif len(disconnected) == 1:
            result.append(disconnected)
        return result

    def form_patches(self, groups):
        patches = []
        for group in groups.values():
            result = self.split_group_into_patches(group)
            for res in result:
                patches.append(TrianglePatch(res))
        return patches


def rad_angle(a, b):
    return np.arccos(vector.dot(a, b) / (vector.length(a) * vector.length(b)))


def deg_angle(a, b):
    return np.arccos(vector.dot(a, b) / (vector.length(a) * vector.length(b))) * 57.2958


def project_vector(v, plane):
    return v * plane


# obj = WfObject("binoculars3")
# triangles = []
#
# l = obj.get_vertex_length()
# count = len(obj.vertices) // obj.get_vertex_length() // 3
# for x in range(0, count):
#     vert_idx = x * 3 * l + (l - 3)
#     t = []
#     t.append(obj.vertices[vert_idx: vert_idx + 3])
#     t.append(obj.vertices[vert_idx + 3 + (l - 3): vert_idx + 6 + (l - 3)])
#     t.append(obj.vertices[vert_idx + 6 + 2 * (l - 3): vert_idx + 9 + 2 * (l - 3)])
#     triangles.append(t)
#
# gen = PathGenerator(triangles)
# width = 0.2
# height = 0.2
# gen.do_main_routine(width, height, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
