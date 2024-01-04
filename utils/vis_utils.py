import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
functions for visualization
"""

def show_mask(mask, ax, random_color=False):
    if random_color:
        rgb = np.random.random(3)
        color = np.concatenate([rgb, np.array([0.65])], axis=0)
    else:
        rgb = None
        color = np.array([30/255, 144/255, 255/255, 0.65])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return rgb

def show_mask_ins(mask, ax, color, random_color=False):
    if random_color:
        rgb = np.random.random(3)
        color = np.concatenate([rgb, np.array([0.65])], axis=0)
    else:
        color = np.concatenate([color, np.array([0.65])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=20, linewidth=1.25)   
    
def show_points_color(coords, labels, ax, rgb, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=rgb, marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=20, linewidth=1.25) 
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        rgb = np.random.random(3)
        color_mask = np.concatenate([rgb, [0.65]])
        img[m] = color_mask
        # also show the corresponding point coords on each instance:
        coords = ann['point_coords']
        color_point = np.concatenate([rgb, [0.95]])  # point color is deeper than mask
        show_points_color(np.array(coords), np.array([1]), ax, rgb=color_point)
    ax.imshow(img)
    
def show_anns_sem(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns.shape[0], anns.shape[1], 4))
    img[:,:,3] = 0
    for i in range(0, 21):
        m = np.where(anns==i)
        color_mask = np.concatenate([np.random.random(3), [0.65]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_anns_ins(anns, num):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns.shape[0], anns.shape[1], 4))
    img[:,:,3] = 0
    for i in range(0, num+1):
        m = np.where(anns==i)
        color_mask = np.concatenate([np.random.random(3), [0.45]])
        img[m] = color_mask
    ax.imshow(img)
    
def cal_iou(pred, gt):
    assert pred.shape == gt.shape  # H * W
    I = np.sum(np.logical_and(pred == 1, gt == 1))
    U = np.sum(np.logical_or(pred == 1, gt == 1))
    iou = I / float(U)
    return iou

def show_iou(data_path_ins, scene_name, frame_id, ins_id, pred, image):
    # instance seg:
    data_path_ins = "sample_data/scannet_2d_allframe_label_ins/"
    label_ins = cv2.imread(os.path.join(data_path_ins, scene_name, 'label', str(frame_id) + '.png'), # TODO: try 'label' instead of 'label_filt'
                   cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

    if image.shape[0] != label_ins.shape[0] or image.shape[1] != label_ins.shape[1]:
            raise (RuntimeError("Image & label shape mismatch!"))
        
    mask_coord = np.where(label_ins == ins_id)
    other_coord = np.where(label_ins != ins_id)
    label_ins[mask_coord] = 1
    label_ins[other_coord] = 0
    
    iou = cal_iou(pred, label_ins)
    
    plt.imshow(image)
    show_mask(label_ins, plt.gca())
    plt.title(f"Instance Annotation, the IoU is: {iou:.5f}", fontsize=10)
    plt.axis('off')
    
    return iou

def rand_cmap(nlabels, type='bright', first_color_black=False, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib ranging from 0~1
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            #print(list(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])))
            randRGBcolors.append(list(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
    
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

#     return random_colormap
    return randRGBcolors


def rand_cmap_sns(nlabels):

    import seaborn as sns
    
    # palette = "husl"
    palette = "bright"
    randRGBcolors = sns.color_palette(palette, nlabels)

    return randRGBcolors


def write_ply_color_rgb(points, labels, rgb, out_filename, n_classes):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]

    np.random.seed(1)
    colors = rand_cmap(n_classes, type='bright', first_color_black=False, last_color_black=False, verbose=False)

    ignore_idx_list = np.where(labels==-1)[0] # list of ignore_idx

    for i in range(N):
        if i in ignore_idx_list:  # if ignore_idx, using original rgb value
            c = rgb[i]  
        else:  # else, using the given label rgb
            c = colors[labels[i]]
            c = [int(x * 255) for x in c]  # change rgb value from 0-1 to 0-255
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def get_color_rgb(points, labels, rgb, cmap):
    labels = labels.astype(int)
    N = points.shape[0]
    colors = cmap
    ignore_idx_list = np.where(labels==-1)[0] # list of ignore_idx
    
    c_all = []
    for i in range(N):
        if i in ignore_idx_list:  # if ignore_idx, using original rgb value
            c = rgb[i] / 255.0
        else:  # else, using the given label rgb
            c = colors[labels[i]]
            c = [x for x in c]
        c_all.append(np.array([c[0], c[1], c[2]]).reshape(1, 3))
    return c_all