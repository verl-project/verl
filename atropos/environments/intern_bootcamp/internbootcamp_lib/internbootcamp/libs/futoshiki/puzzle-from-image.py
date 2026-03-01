import numpy as np
import matplotlib.pyplot as plt
import cv2

imgs = []
for i in range(11):
    img = cv2.imread("img/{}.jpg".format(i), 0)
    imgs.append(img)

# Harris corner detection
# plt.imshow(cv2.cornerHarris(img5, blockSize=5, ksize=3, k=0), "gray")
# Image pyramids
# lower_reso = cv2.pyrDown(img)


def canny_edges(img, thresh=200, ratio=2.5):
    return cv2.Canny(img, thresh, ratio * thresh)


def houghlines_and_plot(img):
    plt.figure()
    plt.subplot(121)
    edges = canny_edges(img)
    plt.imshow(edges, "gray")
    plt.subplot(122)
    rho_thresh = 3
    theta_thresh = np.pi / 180
    houghthresh = 300
    lines = cv2.HoughLines(edges, rho_thresh, theta_thresh, houghthresh)
    imgcopy = img.copy()
    for row in lines:
        # print(row)
        [rho, theta] = row[0]
        # print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imgcopy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(imgcopy, "gray")


def contour_and_plot(img):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img, "gray")
    plt.subplot(122)
    edges = canny_edges(img)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    drawncont = cv2.drawContours(img, contours, -1, (0, 128, 0), 1)
    plt.imshow(drawncont, "gray")


# https://docs.opencv2.org/master/d7/dff/tutorial_feature_homography.html

puzzle3712 = cv2.imread("img/3712crop.jpg".format(i), 0)
scene_imgs = imgs[0:3]


# -- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
# detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
detector = cv2.xfeatures2d_SIFT.create()
keypoints_obj, descriptors_obj = detector.detectAndCompute(puzzle3712, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(scene_imgs[1], None)

# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.4
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# -- Draw matches
img_matches = np.empty(
    (
        max(puzzle3712.shape[0], scene_imgs[1].shape[0]),
        puzzle3712.shape[1] + scene_imgs[1].shape[1],
        3,
    ),
    dtype=np.uint8,
)
cv2.drawMatches(
    puzzle3712,
    keypoints_obj,
    scene_imgs[1],
    keypoints_scene,
    good_matches,
    img_matches,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# -- Localize the object
obj = np.empty((len(good_matches), 2), dtype=np.float32)
scene = np.empty((len(good_matches), 2), dtype=np.float32)
for i in range(len(good_matches)):
    # -- Get the keypoints from the good matches
    obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
# -- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4, 1, 2), dtype=np.float32)
obj_corners[0, 0, 0] = 0
obj_corners[0, 0, 1] = 0
obj_corners[1, 0, 0] = puzzle3712.shape[1]
obj_corners[1, 0, 1] = 0
obj_corners[2, 0, 0] = puzzle3712.shape[1]
obj_corners[2, 0, 1] = puzzle3712.shape[0]
obj_corners[3, 0, 0] = 0
obj_corners[3, 0, 1] = puzzle3712.shape[0]
scene_corners = cv2.perspectiveTransform(obj_corners, H)
# -- Draw lines between the corners (the mapped object in the scene - image_2 )
cv2.line(
    img_matches,
    (int(scene_corners[0, 0, 0] + puzzle3712.shape[1]), int(scene_corners[0, 0, 1])),
    (int(scene_corners[1, 0, 0] + puzzle3712.shape[1]), int(scene_corners[1, 0, 1])),
    (0, 255, 0),
    4,
)
cv2.line(
    img_matches,
    (int(scene_corners[1, 0, 0] + puzzle3712.shape[1]), int(scene_corners[1, 0, 1])),
    (int(scene_corners[2, 0, 0] + puzzle3712.shape[1]), int(scene_corners[2, 0, 1])),
    (0, 255, 0),
    4,
)
cv2.line(
    img_matches,
    (int(scene_corners[2, 0, 0] + puzzle3712.shape[1]), int(scene_corners[2, 0, 1])),
    (int(scene_corners[3, 0, 0] + puzzle3712.shape[1]), int(scene_corners[3, 0, 1])),
    (0, 255, 0),
    4,
)
cv2.line(
    img_matches,
    (int(scene_corners[3, 0, 0] + puzzle3712.shape[1]), int(scene_corners[3, 0, 1])),
    (int(scene_corners[0, 0, 0] + puzzle3712.shape[1]), int(scene_corners[0, 0, 1])),
    (0, 255, 0),
    4,
)
# -- Show detected matches
# cv2.imshow("Good Matches & Object detection", img_matches)
plt.figure()
plt.imshow(img_matches)

contour_and_plot(imgs[2])
plt.figure()
plt.imshow(cv2.cornerHarris(imgs[2].copy(), blockSize=5, ksize=3, k=0), "gray")

plt.show()
