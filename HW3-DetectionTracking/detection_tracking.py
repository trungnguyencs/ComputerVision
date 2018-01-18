import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

Debug_laptop = 1
def skeleton_tracker(v, file_name):
    # Open output file
    # output_name = sys.argv[3] + file_name
    output_name = file_name
    output = open(output_name,"w")
    frameCounter = 0
    # read first frame
    if Debug_laptop == 0:
        ret,frame = v.read()
        if ret == False:
            return
    else:
        if len(v) == 0: return
        frame = v[0]            # first frame

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    faces = (c,r,w,h)
    # Write track point for first frame
    pt = (0,c+w/2,r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    if output_name == "output_camshift.txt":
        def CAMshift(faces, frame):
            # cap = cv2.VideoCapture(0)
            c, r, w, h = faces
            track_window = faces
            # set up the ROI for tracking
            roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))  # this is provided for you
            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            dst_draw = cv2.polylines(dst, [pts], True, 255, 2)
            # cv2.imshow("BackProject", dst_draw)
            cv2.waitKey(10)
            return track_window
        while (1):
            if Debug_laptop == 0:
                ret, frame = v.read()
                if ret == False: return
            else:
                if len(v) == 0: return
                if frameCounter == len(v): break
                frame = v[frameCounter]  # read frame
            ## CAMshift
            c, r, w, h = CAMshift(faces,frame)
            pt = (frameCounter, c + w / 2, r + h / 2)
            output.write("%d,%d,%d\n" % pt)  # Write as frame_index,pt_x,pt_y
            frameCounter = frameCounter + 1

    if output_name == "output_kalman.txt":
        ### Kalman Filter
        c, r, w, h = faces
        state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position

        kalman = cv2.KalmanFilter(4, 2, 0)  # 4 state/hidden, 2 measurement, 0 control
        kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                            [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
        kalman.measurementMatrix = 1. * np.eye(2, 4)  # you can tweak these to make the tracker
        kalman.processNoiseCov = 1e-5 * np.eye(4, 4)  # respond faster to change and be less smooth, try 1e-3
        kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        kalman.errorCovPost = 1e-1 * np.eye(4, 4)
        kalman.statePost = state

        while (1):
            if Debug_laptop == 0:
                ret, frame = v.read()
                if ret == False: return
            else:
                if len(v) == 0: return
                if frameCounter == len(v): break
                frame = v[frameCounter]  # read frame

            # perform the tracking
            # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
            prediction = kalman.predict()
            pt = (frameCounter, prediction[0], prediction[1])
            # print "prediction: ", prediction

            # generate measurement

            measurement = detect_one_face(frame)
            a, b, c, d = measurement
            # print "face detected: ", measurement
            # face_detected = np.array([a + c / 2, b + d / 2])

            # generate measurement
            if tuple(measurement) != (0, 0, 0, 0):  # e.g. face found
                x = np.array([a + c / 2, b + d / 2], dtype='float64')  # initial position
                posterior = kalman.correct(x)
                pt = (frameCounter,posterior[0], posterior[1])
                frame = cv2.rectangle(frame, (a, b), (a + c, b + d), 255, 2)


            # show image with location
            img = cv2.circle(frame, (prediction[0], prediction[1]), 3, (0,255,0), -1)
            cv2.imshow('Kalman', img)
            cv2.waitKey(10)

            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
            frameCounter = frameCounter + 1
            
        cv2.destroyAllWindows()


    if output_name == "output_particle.txt":
        # a function that, given a particle position, will return the particle's "fitness"
        def particleevaluator(back_proj, particle):
            return back_proj[particle[1], particle[0]]

        # hist_bp: obtain using cv2.calcBackProject and the HSV histogram

        roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))  # this is provided for you
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        n_particles = 200       # 200 particles
        init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
        particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init same position
        f = particleevaluator(hist_bp, init_pos) * np.ones(n_particles)  # Evaluate appearance model
        weights = np.ones(n_particles) / n_particles  # weights are uniform (at first)



        while (1):
            if Debug_laptop == 0:
                ret, frame = v.read()
                if ret == False: return
            else:
                if len(v) == 0: return
                if frameCounter == len(v): break
                frame = v[frameCounter]  # read frame

                # Tracking:
            stepsize = 50
            # Particle motion model: uniform step (TODO: find a better motion model)
            np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

            # Clip out-of-bounds particles
            # print "image shape: ", frame.shape
            im_h, im_w = frame.shape[0], frame.shape[1]

            particles = particles.clip(np.zeros(2), np.array((im_w, im_h)) - 1).astype(int)

            f = particleevaluator(hist_bp, particles.T)  # Evaluate particles
            weights = np.float32(f.clip(1))  # Weight ~ histogram response
            weights /= np.sum(weights)  # Normalize w
            pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average
            # print "pos: ", pos

            if 1. / np.sum(weights ** 2) < n_particles / 2.:  # If particle cloud degenerate:
                particles = particles[resample(weights), :]  # Resample particles according to weights
                # resample() function is provided for you

            # show image with location
            for pt in particles:
                frame = cv2.circle(frame, (pt[0], pt[1]), 2, (0,0,255), -1)

            pt = (frameCounter,pos[0], pos[1])
            cv2.imshow('particle', frame)
            cv2.waitKey(25)

            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
            frameCounter = frameCounter + 1

        cv2.destroyAllWindows()

    if output_name == "output_of.txt":

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.05,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # read first frame
        old_frame = v[0]
        frameCounter = 0

        # detect face in first frame
        c,r,w,h = detect_one_face(old_frame)
        faces = (c,r,w,h)
        pt = (frameCounter, c+w/2, r+h/2)
        output.write("%d,%d,%d\n" % pt)  
        frameCounter = frameCounter + 1

        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        roi = np.zeros(old_gray.shape)
        roi[r:r+h,c:c+w] = 1

        p0 = cv2.goodFeaturesToTrack(old_gray, mask = np.uint8(roi), **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while (1):
            if frameCounter == len(v): break
            
            frame = v[frameCounter]  # read frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points    
            good_old = p0[st==1]
            good_new = p1[st==1]
            average_point = np.uint8(np.average(good_new, axis=0))
            pt = (frameCounter, average_point[0], average_point[1])
            output.write("%d,%d,%d\n" % pt)  
                
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
                
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            frameCounter = frameCounter + 1
            
        cv2.destroyAllWindows()

        output.close()

# output_camshift.txt
# output_particle.txt
# output_kalman.txt
# Bonus: output_of.txt

filename = "02-1.avi"
if Debug_laptop:
    import skvideo.io
    v = skvideo.io.vread(filename)
else:
    v = cv2.VideoCapture(filename)

skeleton_tracker(v, "output_camshift.txt")
skeleton_tracker(v, "output_kalman.txt")
skeleton_tracker(v, "output_particle.txt")
skeleton_tracker(v, "output_of.txt")

# if __name__ == '__main__':
#     question_number = -1

#     # Validate the input arguments
#     if (len(sys.argv) != 4):
#         help_message()
#         sys.exit()
#     else:
#         question_number = int(sys.argv[1])
#         if (question_number > 4 or question_number < 1):
#             print("Input parameters out of bound ...")
#             sys.exit()

#     # read video file
#     video = cv2.VideoCapture(sys.argv[2]);

#     if (question_number == 1):
#         skeleton_tracker(video, "output_camshift.txt")
#     elif (question_number == 2):
#         skeleton_tracker(video, "output_particle.txt")
#     elif (question_number == 3):
#         skeleton_tracker(video, "output_kalman.txt")
#     elif (question_number == 4):
#         skeleton_tracker(video, "output_of.txt")
