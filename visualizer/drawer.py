
import numpy as np
import cv2
import assessment.quality as qual


def normalize_1ch(img):
    if img.dtype == np.uint8:
        is_uint8 = True
        img_float = img.astype(np.float32)/255.0
    elif img.dtype == np.float32:
        is_uint8 = False
        img_float = img

    max_val = np.max(img)
    min_val = np.min(img)

    img_normalized = (img_float - min_val)/(max_val - min_val)

    if is_uint8:
        img_normalized = img_normalized*255
        img_normalized = img_normalized.astype(np.uint8)

    return img_normalized


def imshow_with_zoom(wname='zoom',img=[], scale = 1.0, interpolation=cv2.INTER_CUBIC):
    if img == []:
        return
    img_zoom = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)
    cv2.imshow(wname, img_zoom)

def visualize_sparse_code(code, rows=16, cols=16, padding=5, title='code', scale=0.25):
    # visualize the feature map
    _, fmap_h, fmap_w, fmap_chs = code.shape
    vis_h = (padding + fmap_h) * rows + padding
    vis_w = (padding + fmap_w) * cols + padding
    img_vis = np.zeros(shape=(vis_h, vis_w), dtype=np.float32)

    for c in xrange(fmap_chs):
        code_ic = code[:, :, :, c]
        max_val = np.max(code_ic)
        min_val = np.min(code_ic)
        if not (min_val == max_val):
           code_ic = (code_ic - min_val) / (max_val - min_val)

        # set position
        idx_v = c / cols
        idx_h = c % cols
        pos_y = (padding + fmap_h) * idx_v
        pos_x = (padding + fmap_w) * idx_h

        # draw
        img_vis[pos_y:(pos_y + fmap_h), pos_x:(pos_x + fmap_w)] \
            = code_ic
    img_vis = img_vis * 255
    img_vis = img_vis.astype(np.uint8)
    img_vis_cmap = np.zeros(shape=(vis_h, vis_w, 3), dtype=np.uint8)
    cv2.applyColorMap(src=img_vis, colormap=cv2.COLORMAP_JET, dst=img_vis_cmap)
    imshow_with_zoom(wname=title, img=img_vis_cmap, scale=scale, interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('code', img_vis)a
    cv2.waitKey(10)

def draw_the_comparison(img, img_gt=[],
                      rows=4, cols=10, padding_row=35, padding_col=5, title='comp',
                        scale=1.0,
                        compute_psnr=True
                        ):
    # get shape
    shape_img = img.shape

    if img_gt != []:
        shape_gt = img_gt.shape
        # check if they are the same
        if shape_gt != shape_img:
            print 'the size of the two HS images are different'
            return

    # determined the size
    _, h, w, chs = img.shape
    if img_gt != []:
        vis_h = (padding_row + h + padding_row + h)*rows + padding_row
    else:
        vis_h = (padding_row + h)*rows + padding_row
    vis_w = (padding_col + w)*cols + padding_col
    img_vis = np.zeros(shape=(vis_h, vis_w), dtype=np.float32)

    # text setup
    font = cv2.FONT_HERSHEY_TRIPLEX

    if img_gt != []:
        max_val_gt = np.max(img_gt)
        max_val = np.max(img)
        ratio = max_val_gt/max_val
    else:
        ratio = 1.0
    ratio = 1.0

    for c in xrange(chs):
        img_c = img[:, :, :, c]*ratio
        # set position
        idx_v = c / cols
        idx_h = c % cols

        if img_gt != []:
            gt_c = img_gt[:, :, :, c]
            # the position for GT
            pos_gt_y = padding_row + (padding_row + h + padding_row + h) * idx_v
            pos_gt_x = padding_col + (padding_col + w) * idx_h
            # draw gt
            img_vis[pos_gt_y:(pos_gt_y + h), pos_gt_x:(pos_gt_x + w)] \
                = gt_c

        # the position for img

        if img_gt != []:
            pos_img_y = pos_gt_y + h + padding_col
        else:
            pos_img_y = padding_row + (padding_row + h) * idx_v
        pos_img_x = padding_col + (padding_col + w) * idx_h

        # draw img
        img_vis[pos_img_y:(pos_img_y + h), pos_img_x:(pos_img_x + w)] \
            = img_c



        # compute PSNR
        if img_gt != [] and compute_psnr:
            psnr_val = qual.psnr_1ch(img_c, gt_c)
            psnr_text = '%.2f'%(psnr_val)
            pos_font_x = pos_gt_x + 10
            pos_font_y = pos_img_y + h + 20
            cv2.putText(img_vis, psnr_text, fontFace=font,
                        org=(pos_font_x, pos_font_y), fontScale=0.75,
                        color=(1.0, 1.0, 1.0), bottomLeftOrigin=False)


    img_vis = np.power(img_vis, 1/2.2)
    img_vis = img_vis * 255
    img_vis = img_vis.astype(np.uint8)
    img_vis_cmap = np.zeros(shape=(vis_h, vis_w, 3), dtype=np.uint8)
    cv2.applyColorMap(src=img_vis, colormap=cv2.COLORMAP_JET, dst=img_vis_cmap)
    imshow_with_zoom(wname=title, img=img_vis, scale=1, interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('code', img_vis)a
    #cv2.waitKey(100)

