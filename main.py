import numpy as np
import cv2 
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div



# dir_path = Path(r'NTSD-complete-v1.0.1\NewTsukubaStereoDataset\illumination\lamps')
dir_path = Path(r'KITTI_sequence_2\image_l')
left_imgs = [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith('L')]


class MonocularOdometry():
    def __init__(self, directory) -> None:
        self.K = self._form_cam_matrix()
        self.images = self._load_images(directory)
        self.P = np.matmul(self.K, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

        self.main()


    @staticmethod
    def _load_images(directory):
        """
        Load grayscale images from the specified directory.

        Parameters
        ----------
        directory : pathlib.Path
            The directory containing the images.

        Returns
        -------
        list
            A list of grayscale images.
        """
        # img_paths = [file for file in directory.iterdir() if file.is_file() and file.name.startswith('L')]
        img_paths = [file for file in directory.iterdir() if file.is_file()]
        print(img_paths[-1])
        
        return [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in img_paths]
    

    @staticmethod
    def _form_cam_matrix():
        """
        Form the camera matrix based on focal length, principal point, and image size.

        Returns
        -------
        np.ndarray
            The camera matrix.
        """
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

        keypoints1, descriptors1 = orb.detectAndCompute(self.images[i - 1], None)

        keypoints2, descriptors2 = orb.detectAndCompute(self.images[i], None)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.60 * n.distance:
                good.append(m)

        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])
       
        if plot:
            img_matches = cv2.drawMatches(
                self.images[i - 1], keypoints1,
                self.images[i], keypoints2,
                good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            cv2.imshow('Matches', img_matches)
            cv2.waitKey(1)

        return q1, q2
    

    def get_pose(self, feat_0, feat_1):
        """
        Estimate the pose (rotation and translation) between two sets of feature points.

        Parameters
        ----------
        feat_0 : np.ndarray
            Feature points in the first image.
        feat_1 : np.ndarray
            Feature points in the second image.

        Returns
        -------
        tuple
            A tuple containing the rotation matrix (R) and translation vector (t).
        """

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
        

        K = np.concatenate((self.K, np.zeros((3,1)) ), axis = 1)


        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
    
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if (max == 2):
    
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):

            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
    
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
    
            return R2, np.ndarray.flatten(t)
        
    
    def visualize_paths(self, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
        """
        Visualize the predicted path in 2D.

        Parameters
        ----------
        pred_path : np.ndarray
            Array containing the predicted path.
        html_tile : str, optional
            Title for the HTML output, by default "".
        title : str, optional
            Title for the visualization, by default "VO exercises".
        file_out : str, optional
            Output file name for the plot, by default "plot.html".
        """

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
        """
        Main function for visual odometry.
        """
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
