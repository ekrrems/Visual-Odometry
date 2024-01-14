print('ekrem')
import pandas as pd
import numpy as np
import cv2 
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div



# use images until 370
# dir_path = Path(r'NTSD-complete-v1.0.1\NewTsukubaStereoDataset\illumination\lamps')
dir_path = Path(r'KITTI_sequence_2\image_l')
left_imgs = [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith('L')][:15]


class MonocularOdometry():
    def __init__(self, directory) -> None:
        self.K = self._form_cam_matrix()
        self.images = self._load_images(directory)
        self.P = np.matmul(self.K, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

        self.main()


    @staticmethod
    def _load_images(directory):
        # img_paths = [file for file in directory.iterdir() if file.is_file() and file.name.startswith('L')][:140]
        img_paths = [file for file in directory.iterdir() if file.is_file()]
        print(img_paths[-1])
        
        return [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in img_paths]
    

    @staticmethod
    def _form_cam_matrix():
        focal_length = [615, 615]  
        principal_point = [320, 240]  
        image_size = (480, 640)  

        camera_matrix = np.array([[focal_length[0], 0, principal_point[0]],
                                  [0, focal_length[1], principal_point[1]],
                                  [0, 0, 1]], dtype=np.float32)
        
        return camera_matrix
    

    @staticmethod
    def _transf_matrix(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    

    def get_features(self, i, plot=False):
        """
        This function detects and computes keypoints and descriptors from the i-1'th and i'th image using ORB.

        Parameters
        ----------
        i (int): The current frame
        plot (bool): Whether to plot the matches

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        orb = cv2.ORB_create(3000) 
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)   

        # Detect keypoints and compute descriptors for the (i-1)'th image
        keypoints1, descriptors1 = orb.detectAndCompute(self.images[i - 1], None)

        # Detect keypoints and compute descriptors for the i'th image
        keypoints2, descriptors2 = orb.detectAndCompute(self.images[i], None)

        # Use FLANN to perform knn matching
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test to select good matches
        good = []
        for m, n in matches:
            if m.distance < 0.60 * n.distance:
                good.append(m)

        # Extract the keypoints' coordinates for the good matches
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])

        # Plot matches if requested
        if plot:
            img_matches = cv2.drawMatches(
                self.images[i - 1], keypoints1,
                self.images[i], keypoints2,
                good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Display the image with matches
            cv2.imshow('Matches', img_matches)
            cv2.waitKey(1)
            

        return q1, q2
    

    def get_pose(self, feat_0, feat_1):
        Essential, mask = cv2.findEssentialMat(feat_0, feat_1, self.K)
        R, t = self.decomp_essential_mat(Essential, feat_0, feat_1)

        return self._transf_matrix(R,t)
    

    def decomp_essential_mat(self, E, q1, q2):
        """"
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._transf_matrix(R1,np.ndarray.flatten(t))
        T2 = self._transf_matrix(R2,np.ndarray.flatten(t))
        T3 = self._transf_matrix(R1,np.ndarray.flatten(-t))
        T4 = self._transf_matrix(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate((self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_feat_0 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)
        
    
    def visualize_paths(self, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
        output_file(file_out, title=html_tile)
        pred_path = np.array(pred_path)

        tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

        pred_x, pred_z = pred_path.T
        source = ColumnDataSource(data=dict(px=pred_path[:, 0], pz=pred_path[:, 1]))

        fig = figure(title="Predicted Path (2D)", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                    x_axis_label="x", y_axis_label="z", height=500, width=800,
                    output_backend="webgl")

        fig.line("px", "pz", source=source, line_width=2, line_color="green", legend_label="Pred")
        fig.circle("px", "pz", source=source, size=8, color="green", legend_label="Pred")

        show(layout([Div(text=f"<h1>{title}</h1>"),
                    [fig],
                    ], sizing_mode='scale_width'))


  
    def main(self):
        estimated_path = []
        cur_pose = np.eye(4)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(0, 0, marker='o', c='r', label='Origin')
        for i, img in enumerate(tqdm(self.images, unit="pose")):
            if i == 0:
                cur_pose = np.eye(4)

            else:
                try:
                    q1, q2 = self.get_features(i, True)
                    transf = self.get_pose(q1, q2)
                    cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
                except (cv2.error, ValueError) as e:
                    print(f"Error in processing frame {i}: {e}")
                    continue
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        self.visualize_paths(estimated_path, "Visual Odometry")

if __name__ == '__main__':
    MonocularOdometry(dir_path)
