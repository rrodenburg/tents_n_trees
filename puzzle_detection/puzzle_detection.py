### See the notebook puzzle_detector for more documentation

import cv2
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class PuzzleDetector:

    def __init__(self, image_location, kernel_size=11, high_threshold=10, low_threshold=5, rho=3, vote_threshold=20):
        
        self.image = cv2.imread(image_location)
        self.image_bw = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Parameters of the algrithm
        self.kernel_size = kernel_size # Size of the gaussian kernal use to smooth the image.
        self.high_threshold = high_threshold # Intensity gradient above which pixels are certainly classified as edge.
        self.low_threshold = low_threshold # pixel with an intensity gradient below the min treshold cannot be edges, pixels between both thresholds are only classified as edges if connected to pixels above the high threshold

        self.rho = rho  # distance resolution in pixels of the Hough grid
        self.vote_threshold = vote_threshold # minimum number of votes (intersections in Hough grid cell)

        # Detected game variables
        self.dimension = None
        self.square_coordinates = None # Shape of the array is [n_blocks,[x1,x2,y1,y2]]
        self.tree_indices = None

    def detect_lines(self):

        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        min_line_length = self.image.shape[0]*0.8  # minimum number of pixels making up a line
        max_line_gap = self.image.shape[0]*0.05 # maximum gap in pixels between connectable line segments

        blur_gray = cv2.GaussianBlur(self.image_bw,(self.kernel_size, self.kernel_size),0)
        edges = cv2.Canny(blur_gray, self.low_threshold, self.high_threshold)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, self.rho, theta, self.vote_threshold, np.array([]),
                            min_line_length, max_line_gap).reshape(-1,4)

        return lines

    @staticmethod
    def deduplicate_lines(lines):
        dbscan = DBSCAN(eps=50, min_samples=1)
        cluster_ids = dbscan.fit_predict(lines.reshape(-1,4))
        deduplicated_lines = lines[np.unique(cluster_ids, return_index=True)[1]]
        return deduplicated_lines.reshape(-1,2,2)

    def find_intersections(self, deduplicated_lines):

        hor_lines = deduplicated_lines[np.diff(deduplicated_lines[:,:,0]).reshape(-1) == 0]
        ver_lines = deduplicated_lines[np.diff(deduplicated_lines[:,:,1]).reshape(-1) == 0]

        if len(hor_lines) == len(ver_lines):
            self.dimension = len(hor_lines) - 1
        else:
            print('Warning: the number of horizontal and vertical lines is not equal. The puzzle is incorrectly detected')

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return np.array([[x, y]]).astype('int')

            # Initiate an empty array with shape [n_intersections,xy_coordinates]

        n_intersections = len(hor_lines) * len(ver_lines)
        intersections = np.zeros(shape=(n_intersections,2))

        i = 0
        for hor_line in hor_lines:
            for ver_line in ver_lines:
                intersections[i] = line_intersection(hor_line, ver_line)
                i += 1

        return intersections
    
    def find_square_coordinates(self, intersections):

        squares = np.empty(shape=(self.dimension**2,4), dtype='int')

        idx = 0
        for intersection in intersections:
            intersections_diff = intersections - intersection
            
            # Exclude intersections that are on the same x or y axis
            potentials = intersections[(intersections_diff[:,0] > 0) &
                                    (intersections_diff[:,1] > 0)]
            
            # Select the closest intersection
            closest_idx = np.argsort(np.sum((potentials - intersection)**2, axis=1))
            
            if len(closest_idx) > 0:
                square = np.concatenate(([potentials[closest_idx[0]]], [intersection]), axis=0).reshape(4)
                
                # Assign and reorder indices to x1, x2, y1, y2
                squares[idx] = square[[3,1,2,0]]
                idx += 1

        self.square_coordinates = squares[np.lexsort((squares[:,2],squares[:,0]))]\
                                        .reshape(self.dimension,self.dimension,4)\

        return
    
    def detect_squares(self):
        
        lines = self.detect_lines()

        lines = self.deduplicate_lines(lines)

        intersections = self.find_intersections(lines)

        self.find_square_coordinates(intersections)

        return

    def detect_trees(self):

        def mean_intensity(square, bw_image):
            """Calculate the mean pixel intensity for a square"""
            x1, x2, y1, y2 = square
            area_of_interest = bw_image[y1:y2, x1:x2]
            return np.mean(area_of_interest).astype('int')

        # Calculate the mean pixel intesity for each square of the game.
        square_intensity = np.apply_along_axis(mean_intensity, 1,
                                               self.square_coordinates.reshape(-1,4), 
                                               self.image_bw)

        kmeans = KMeans(n_clusters = 2)
        tree_flag = kmeans.fit_predict(square_intensity.reshape(-1,1)).astype('int')

        # Ensure that trees are always cluster 1
        if sum(tree_flag == 1) > sum(tree_flag):
            idx_0 = tree_flag == 0
            idx_1 = tree_flag == 1

            tree_flag[idx_0] = 1
            tree_flag[idx_1] = 0
    
        self.tree_indices = np.argwhere(tree_flag.reshape(self.dimension,self.dimension).T==1)
        return

    def show_squares(self, square_indices, filename=None):

        # Extract squares coordinates
        squares = self.square_coordinates[square_indices.T]
        print(len(squares))

        # Initiate a empty array
        squares_image = np.copy(self.image) * 0

        # Extract coordinates for the square
        for square in squares.reshape(-1,4):
            x1,x2,y1,y2 = square
            # Set pixels to max value
            squares_image[y1:y2,x1:x2] = 255
        
        # Combine game image with blocks
        temp = cv2.addWeighted(self.image, 0.8, squares_image, 0.3, 0)
        
        if filename:
            cv2.imwrite(f'./examples/{filename}.png', temp)
    
        plt.imshow(temp)
        return



