import numpy as np

# produces an ordered set of corners
# which, when plotted, create a parallelogram
corners_xyz_coeffs = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1,-1], # front face
                               [-1,-1, 1],
                               [-1, 1, 1],
                               [-1, 1,-1], # left face
                               [-1, 1, 1],
                               [ 1, 1, 1],
                               [ 1, 1,-1], # top face
                               [ 1, 1, 1],
                               [ 1,-1, 1],
                               [ 1,-1,-1], # right face
                               [ 1,-1, 1],
                               [-1,-1, 1]], dtype=float)# bottom & back face

def make_rotation_matrix(pitch=0., yaw=0., roll=0., radians=False):
    if not radians:
        b, a, g = np.radians([pitch, yaw, roll])
    else:
        b, a, g = pitch, yaw, roll
        
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sg, cg = np.sin(g), np.cos(g)
        
    # as reported in: https://en.wikipedia.org/wiki/Rotation_matrix
    rot = np.array([[ca*cb, ca*sb*sg-sa*cg, ca*sb*cg-sa*sg],
                    [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg],
                    [-sb, cb*sg, cb*cg]])
        
    return rot

# note that it expects points and camera intrinsics with the left-handed z-up convention
# meaning that the points height (and not the distance from the sensor) is encoded in the z position
# on the other hand, the y-direction corresponds to the left-right shift and the x is the distance
# returns the raw projection, so that we can feed it into a z-buffer
def project_bounding_box(center, extent, rotation, ego_position, ego_rotation, camera_position, camera_rotation, camera_intrinsics):
    ego_inverse = np.linalg.inv(make_rotation_matrix(**ego_rotation))
    camera_inverse = np.linalg.inv(make_rotation_matrix(**camera_rotation))
    
    corners = corners_xyz_coeffs*extent
    corners = np.matmul(make_rotation_matrix(**rotation), corners.T).T
    corners = center+corners-ego_position
    corners = np.matmul(ego_inverse, corners.T).T - camera_position
    corners = np.matmul(camera_inverse, corners.T)
    
    proj = np.matmul(camera_intrinsics, corners).T
    return proj
    
def project_boxes(bboxes, wdata, sdata, id_map=None):
    s_tr, s_rot, K = np.array(sdata['translation']), sdata['rotation'], np.array(sdata['camera_intrinsic'])
    im_w, im_h = 2*K[0,0], 2*K[1,0]
    w_tr, w_rot = np.array([wdata['Location'][k] for k in wdata['Location']]), wdata['Rotation']
    
    to_render = []
    for bbox in bboxes:
        ext = np.array([bbox['extent'][k] for k in bbox['extent']])
        loc = np.array([bbox['location'][k] for k in bbox['location']])
        rot = bbox['rotation']
        lab = id_map[bbox['label']] if id_map is not None else bbox['label']
        
        proj = project_bounding_box(loc, ext, rot, w_tr, w_rot, s_tr, s_rot, K)
        # check if at least one corner is in front of the camera
        if np.any(proj[:,2]>0):
            min_z = np.min(proj[:,2])
            proj = proj[proj[:,2]>0,:]
            proj /= proj[:,2:3]
            # check if the projection is inside the 
            if np.any(np.logical_or(np.logical_and(proj[:,0]>=0,proj[:,0]<im_w), np.logical_and(proj[:,1]>=0, proj[:,1]<im_h))):
                to_render.append((min_z, proj.copy(), lab))
    to_render.sort(key=lambda e: e[0])
    to_render = [(e[1], e[2]) for e in to_render]
    return to_render, im_w, im_h