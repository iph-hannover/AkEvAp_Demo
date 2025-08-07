import numpy as np

JOINTS = {'3D':
              ['root', 'right_hip', 'right_knee', 'right_ankle',
               'left_hip', 'left_knee', 'left_ankle',
               'spine', 'center_shoulder', 'center_head', 'top_head',
               'left_shoulder', 'left_elbow', 'left_wrist',
               'right_shoulder', 'right_elbow', 'right_wrist'],
          '2D':
              ['head_joint', 'neck_1_joint',
               'right_shoulder_1_joint', 'right_forearm_joint', 'right_hand_joint', 'left_shoulder_1_joint', 'left_forearm_joint', 'left_hand_joint',
               'root', 'right_upLeg_joint', 'right_leg_joint', 'right_foot_joint', 'left_upLeg_joint', 'left_leg_joint', 'left_foot_joint',
               'right_eye_joint', 'left_eye_joint', 'right_ear_joint', 'left_ear_joint']
          }
BONES = {'3D': [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
                [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]],
         '2D': [[0, 1], [1, 8],
                [1, 2], [2, 3], [3, 4],                 # right arm
                [1, 5], [5, 6], [6, 7],                 # left arm
                [8, 9], [9, 10], [10, 11],              # right leg
                [8, 12], [12, 13], [13, 14],            # left leg
                [0, 15], [0, 16], [15, 17], [16, 18]],  # head
         }
COLORS = {
    'r': (255, 0, 0),
    'b': (0, 255, 0),
    'g': (0, 0, 255),
    'y': (255, 255, 0),
    'c': (0, 255, 255),
}

def get_bones_named(type):
    return [[JOINTS[type][j1], JOINTS[type][j2]] for j1, j2 in BONES[type]]

def print_bones(type):
    bone_name_pairs = get_bones_named(type)
    print([f"{j1} <-> {j2}" for j1, j2 in bone_name_pairs])

def get_bone_colors(type):
    """
    Get a list of matplotlib color-characters that describes whether a bone belongs to the right (r) or left (g) side of the body or to the middle (b)
    :return: list of matplotlib color-characters
    """
    def color_map(j1: str, j2: str) -> str:
        if j1.lower().startswith('l') or j2.lower().startswith('l'):
            return 'g' if type == '2D' else 'b'
        elif (j1.lower().startswith('r') and j1.lower() != 'root') or (j2.lower().startswith('r') and j2.lower() != 'root'):
            return 'r' if type == '2D' else 'c'
        else:
            return 'b' if type == '2D' else 'y'
    return [color_map(*bone) for bone in get_bones_named(type)]

def get_joint_colors(type):
    """
    Get a list of matplotlib color-characters that describes whether a joint belongs to the right (r) or left (g) side of the body or to the middle (b)
    :return: list of matplotlib color-characters
    """
    def color_map(j: str) -> str:
        if j.lower().startswith('l'):
            return 'g' if type == 'default' else 'b'
        elif j.lower().startswith('r') and j.lower() != 'root':
            return 'r' if type == 'default' else 'c'
        else:
            return 'b' if type == 'default' else 'y'
    return [color_map(joint) for joint in JOINTS[type]]


def plot(points, type):
    from matplotlib import pyplot as plt
    # import matplotlib as mpl
    # mpl.use('Qt5Agg')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    mask = np.logical_not(np.all(points == 0, axis=1))
    colors = get_bone_colors(type)
    n_bones = len(BONES[type])
    n_joints = n_bones + 1

    def get_limits(x, y, z):
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        return Xb, Yb, Zb

    assert points.size == 3*n_joints

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x[mask], y[mask], z[mask])

    for i, b in enumerate(BONES[type]):
        if b[0] in np.nonzero(mask)[0] and b[1] in np.nonzero(mask)[0]:
            ax.plot(x[b], y[b], z[b], color=colors[i])

    # Create cubic bounding box to simulate equal aspect ratio
    Xb, Yb, Zb = get_limits(x, y, z)

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #ax.axis('equal')
    #ax.axis('off')
    plt.show()
    return
