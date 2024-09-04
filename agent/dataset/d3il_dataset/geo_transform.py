import itertools

import numpy as np

"""
From OpenAIGym Please see there under mujoco/Robots
"""

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    :param curr_quat: current quaternion
    :param des_quat: desired quaternion
    :return: difference between current quaternion and desired quaternion
    """
    quatError = np.zeros((3,))

    quatError[0] = (
        curr_quat[0] * des_quat[1]
        - des_quat[0] * curr_quat[1]
        - curr_quat[3] * des_quat[2]
        + curr_quat[2] * des_quat[3]
    )

    quatError[1] = (
        curr_quat[0] * des_quat[2]
        - des_quat[0] * curr_quat[2]
        + curr_quat[3] * des_quat[1]
        - curr_quat[1] * des_quat[3]
    )

    quatError[2] = (
        curr_quat[0] * des_quat[3]
        - des_quat[0] * curr_quat[3]
        - curr_quat[2] * des_quat[1]
        + curr_quat[1] * des_quat[2]
    )

    return quatError


def euler2mat(euler):
    """Convert Euler Angles to Rotation Matrix.  See rotation.py for notes"""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """Convert Euler Angles to Quaternions.  See rotation.py for notes"""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    return mat2euler(quat2mat(quat))


def subtract_euler(e1, e2):
    assert e1.shape == e2.shape
    assert e1.shape[-1] == 3
    q1 = euler2quat(e1)
    q2 = euler2quat(e2)
    q_diff = quat_mul(q1, quat_conjugate(q2))
    return quat2euler(q_diff)


def quat2mat(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_rot_vec(q, v0):
    q_v0 = np.array([0, v0[0], v0[1], v0[2]])
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[1:]
    return v


def quat_identity():
    return np.array([1, 0, 0, 0])


def quat2axisangle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def euler2point_euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 3
    _euler_sin = np.sin(_euler)
    _euler_cos = np.cos(_euler)
    return np.concatenate([_euler_sin, _euler_cos], axis=-1)


def point_euler2euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 6
    angle = np.arctan(_euler[..., :3] / _euler[..., 3:])
    angle[_euler[..., 3:] < 0] += np.pi
    return angle


def quat2point_quat(quat):
    # Should be in qw, qx, qy, qz
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 4
    angle = np.arccos(_quat[:, [0]]) * 2
    xyz = _quat[:, 1:]
    xyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (xyz / np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([np.sin(angle), np.cos(angle), xyz], axis=-1)


def point_quat2quat(quat):
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 5
    angle = np.arctan(_quat[:, [0]] / _quat[:, [1]])
    qw = np.cos(angle / 2)

    qxyz = _quat[:, 2:]
    qxyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (qxyz * np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([qw, qxyz], axis=-1)


def normalize_angles(angles):
    """Puts angles in [-pi, pi] range."""
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -np.pi - 1e-6 <= angles.min() and angles.max() <= np.pi + 1e-6
    return angles


def round_to_straight_angles(angles):
    """Returns closest angle modulo 90 degrees"""
    angles = np.round(angles / (np.pi / 2)) * (np.pi / 2)
    return normalize_angles(angles)


def get_parallel_rotations():
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 24
    return parallel_rotations


def posRotMat2TFMat(pos, rot_mat):
    """Converts a position and a 3x3 rotation matrix to a 4x4 transformation matrix"""
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


def mat2posQuat(mat):
    """Converts a 4x4 rotation matrix to a position and a quaternion"""
    pos = mat[:3, 3]
    quat = mat2quat(mat[:3, :3])
    return pos, quat


def wxyz_to_xyzw(quat):
    """Converts WXYZ Quaternions to XYZW Quaternions"""
    return np.roll(quat, -1)


def xyzw_to_wxyz(quat):
    """Converts XYZW Quaternions to WXYZ Quaternions"""
    return np.roll(quat, 1)
