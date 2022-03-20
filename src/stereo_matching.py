import numpy as np
from scipy.ndimage.filters import *
from scipy.sparse import diags
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from numba import jit


def str_comma(number):
    """
    Create a string from a number and replace all dots by a comma
      
        @param number: any number that should be converted to a string

        @return:       a string of the number with 2 decimals where all dots are replaced by commas
    """

    return str(round(number, 2)).replace('.',',')


def export_img(export_image, name, error_measure, matching_method, D, R, accX = 0):
    """
    Export image to disk with an approriate file name
      
        @param export_image:    the image data that has to be exported as numpy array
        @param name:            name and sub-directory of the image
        @param error_measure:   the error measure used (e.g. SSD, SAD, NCC)
        @param matching_method: the measure used for matching point (e.g. WTA, SGM)
        @param D:               maximum disparity
        @param R:               filter radius
        @param accX:            accX measure for evaluation (if available)

        @return:                none
    """

    filename = name + "_" + error_measure + "_" + matching_method + "_D" + str_comma(D) + "_R" + str_comma(R)

    if accX != 0:
        filename += "_accX" + str_comma(accX)

    imsave(filename + ".jpg", img_as_ubyte(export_image), quality = 100)
    return
        

def normalise_img(image, groundtruth_image = None):
    """
    Normalise image with ground truth to floating number points in interval 0..1
      
        @param image:             non-normalised image
        @param groundtruth_image: ground-truth

        @return:                  image normalised with ground truth or maximimum distance
    """
    
    if groundtruth_image is not None:
        assert(np.max(groundtruth_image) > 0)
        image = image/np.max(groundtruth_image)

    assert(np.max(image) > 0)
    return image/np.max(image)


def compute_wta(cv):
    '''
    Compute the best disparity on a scan line using winner-takes-it-all
        
        @param cv: a given cost volume (H,W,D)
        
        @return    a disparity image (H,W)
    '''
    
    assert(cv.ndim == 3)
    return np.argmin(cv, axis=2)


def compute_accX(prediction_image, groundtruth_image, mask_image, X = 3):
    '''
    Compute the accX accuracy measure [0..1]
        
        @param prediction_image:  the stereo image as reconstructed by an algorithm
        @param groundtruth_image: the ground truth stereo image
        @param mask_image:        the mask for excluding invalid pixels such as occluded areas
        @param X:                 threshold disparity measure
        
        @return                   the accX measure of the reconstructed stereo image
    '''
    
    Z = np.sum(mask_image)
    assert(Z > 0)
    assert(prediction_image.shape == groundtruth_image.shape == mask_image.shape)
    
    mask_rel = mask_image*(np.absolute(prediction_image - groundtruth_image) <= X)
    return 1/Z*np.sum(mask_rel)


@jit(nopython = True, parallel = True, cache = True)
def compute_costvolume_sad(left_image, right_image, D, R):
    """
    Compute a cost volume with maximum disparity D considering a neighbourhood R with Sum of Absolute Differences (SAD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param D:           maximum disparity to be considered
        @param R:           radius of the filter

        @return:            cost volume of size (H,W,D)
    """
    
    assert(left_image.shape == right_image.shape)
    assert(D > 0)
    assert(R > 0)
    
    (H,W) = left_image.shape
    cv    = np.zeros((H,W,D))
    
    # Loop over internal image
    for y in range(R, H - R):
        for x in range(R, W - R):
            
            # Loop over window
            for v in range(-R, R + 1):
                for u in range(-R, R + 1):
                    
                    # Loop over all possible disparities
                    for d in range(0, D):
                        cv[y,x,d] += np.absolute(left_image[y+v, x+u] - right_image[y+v, x+u-d])
        
    return cv


@jit(nopython = True, parallel = True, cache = True)
def compute_costvolume_ssd(left_image, right_image, D, R):
    """
    Compute a cost volume with maximum disparity D considering a neighbourhood R with Sum of Squared Differences (SSD)
    
        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param D:           maximum disparity
        @param R:           radius of the filter
    
        @return:            cost volume of size (H,W,D)
    """
    
    assert(left_image.shape == right_image.shape)
    assert(D > 0)
    assert(R > 0)
    
    (H,W) = left_image.shape
    cv    = np.zeros((H,W,D))
    
    # Loop over internal image
    for y in range(R, H - R):
        for x in range(R, W - R):
            
            # Loop over window
            for v in range(-R, R + 1):
                for u in range(-R, R + 1):
                    
                    # Loop over all possible disparities
                    for d in range(0, D):
                        cv[y,x,d] += (left_image[y+v, x+u] - right_image[y+v, x+u-d])**2
        
    return cv


@jit(nopython = True, parallel = True, cache = True)
def compute_costvolume_ncc(left_image, right_image, D, R):
    """
    Compute a cost volume with maximum disparity D considering a neighbourhood R with Normalized Cross Correlation (NCC)
    
        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param D:           maximum disparity
        @param radius:      radius of the filter
        
        @return:            cost volume of size (H,W,D)
    """
    
    assert(left_image.shape == right_image.shape)
    assert(D > 0)
    assert(R > 0)
    
    (H,W) = left_image.shape
    cv    = np.zeros((D,H,W))
    
    # Loop over all possible disparities
    for d in range(0, D):
        
        # Loop over image
        for y in range(R, H - R):
            for x in range(R, W - R):
                
                l_mean = 0
                r_mean = 0
                n      = 0
                
                # Loop over window
                for v in range(-R, R + 1):
                    for u in range(-R, R + 1):
                        
                        # Calculate cumulative sum
                        l_mean += left_image[y+v, x+u]
                        r_mean += right_image[y+v, x+u-d]
                        n      += 1

                l_mean = l_mean/n
                r_mean = r_mean/n
                
                l_r   = 0
                l_var = 0
                r_var = 0
            
                for v in range(-R, R + 1):
                    for u in range(-R, R + 1):
                        
                        # Calculate terms
                        l = left_image[y+v, x+u]    - l_mean
                        r = right_image[y+v, x+u-d] - r_mean
                        
                        l_r   += l*r
                        l_var += l**2
                        r_var += r**2
                        
                        
                # Assemble terms
                cv[d,y,x] = -l_r/np.sqrt(l_var*r_var)
    
    return np.transpose(cv, (1, 2, 0))


def get_f(D, L1 = 0.025, L2 = 0.5):
    """
    Get pairwise cost matrix for semi-global matching
    
        @param D:  disparities, number of possible choices
        @param L1: parameter for setting cost for jumps between two layers of depth
        @param L2: cost for jumping more than one layer of depth
    
        @return: pairwise_costs of shape (D,D)
    """
    
    return np.full((D, D), L2) + diags([L1 - L2, -L2, L1 - L2], [-1, 0, 1], (D, D)).toarray()


# For some reason @jit(nopython = True, parallel = True, cache = True) does not work here!
@jit
def compute_message(cv, f):
    """
    Compute the messages in one particular direction for semi-global matching
    
        @param cv: cost volume of shape (H,W,D)
        @param f:  pairwise costs of shape (D,D)
    
        @return:   messages for all H in positive direction of W with possible options D (H,W,D)
    """
    
    (H,W,D) = cv.shape
    mes     = np.zeros((H,W,D))
    
    # Loop over passive direction
    for y in range(0, H):
        
        # Loop over forward direction
        for x in range(0, W - 1):
            
            # Loop over all possible nodes
            for t in range(0, D):
                
                # Loop over all possible connections
                buffer = np.zeros(D)
                for s in range(0, D):
                    # Input messages + unary cost + binary cost
                    buffer[s] = mes[y,x,s] + cv[y,x,s] + f[t,s]
                
                # Choose path of least effort
                mes[y, x+1, t] = np.min(buffer)
                
    return mes
    

def compute_sgm(cv, f):
    """
    Compute semi-global matching by message passing in four directions
    
        @param cv: cost volume of shape (H,W,D)
        @param f:  pairwise costs of shape (H,W,D,D)
    
        @return:   pixel-wise disparity map of shape (H,W)
    """
    # Messages for every single spatial direction and collect in single message
    (H,W,D) = cv.shape
    mes     = np.zeros((H,W,D))
    
    # Positive W
    mes += compute_message(cv, f)
    
    # Negative W
    mes_buffer  = np.zeros((H,W))
    mes_buffer  = compute_message(np.flip(cv, axis=1), f)
    mes        += np.flip(mes_buffer, axis=1)
    
    # Positive H
    mes_buffer  = compute_message(np.transpose(cv, (1, 0, 2)), f)
    mes        += np.transpose(mes_buffer, (1, 0, 2))
    
    # Negative H
    mes_buffer  = compute_message(np.flip(np.transpose(cv, (1, 0, 2)), axis=1), f)
    mes        += np.transpose(np.flip(mes_buffer, axis=1), (1, 0, 2))
    
    # Choose best believe from all messages
    disp_map = np.zeros((H,W))
    for y in range(0, H):
        for x in range(0, W):
            # Minimum argument of unary cost and messages
            disp_map[y,x] = np.argmin(cv[y,x,:] + mes[y,x,:])
    
    return disp_map


def main():
    # Load input images
    im0 = imread("../data/Adirondack_left.png")
    im1 = imread("../data/Adirondack_right.png")

    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    # Plot input images
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(1,2,2), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()

    # Use either SAD, NCC or SSD to compute the cost volume
    cv = compute_costvolume_ncc(im0g, im1g, 64, 5)

    # Compute pairwise costs
    (H,W,D) = cv.shape
    f = get_f(D, 0.025, 0.5)
    # Compute SGM
    disp = compute_sgm(cv, f)

    # Plot result
    plt.figure()
    plt.imshow(disp)
    plt.show()


if __name__== "__main__":
    main()
