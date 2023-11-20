from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def project(angle, mirror, img):
    height, width = img.shape[:2]

    new_width = int(width * np.cos(angle * np.pi / 180.))
    added_height = int(width * np.sin(angle * np.pi / 180.))
    new_height = height + added_height

    pts_src = np.array([[1, 1],
                        [width, 1],
                        [width, height],
                        [1, height]], dtype=float) - 1
    
    pts_dst = np.array([[1, 1],
                        [new_width, added_height],
                        [new_width, new_height],
                        [1, height]], dtype=float) - 1

    projected_img = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    mask = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(int), 255, 16)

    projected_img[:, :, 3] = mask

    h = cv2.findHomography(pts_src, pts_dst)[0]
    if not mirror:
        img = img[:, ::-1]

    im_temp = cv2.warpPerspective(img, h, (new_width, new_height))
    projected_img[:, :, :3] += im_temp

    if not mirror:
        projected_img = projected_img[:, ::-1]
    return projected_img


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_img',
                        help='Path for the input image',
                        type=lambda p: Path(p))
    parser.add_argument('-a', '--angle', required=False,
                        help='Angle in degrees to project the image',
                        type=int, default=30)
    parser.add_argument('-m', '--mirror', required=False,
                        help='Horizontally mirror the resulting image',
                        action='store_true')
    parser.add_argument('-o', '--output_img', required=False,
                        help='Path for the output image',
                        type=lambda p: Path(p))
    args = parser.parse_args()

    input_img = cv2.imread(str(args.input_img))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    if args.output_img:
        output_img = args.output_img
    else:
        output_img = f'projected_{args.angle}_{args.input_img.stem}.png'

    out_img = Image.fromarray(project(args.angle, args.mirror, input_img), 'RGBA')
    out_img.save(output_img)