import cv2
import os

def create_folder(base_dir, folder_name):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def capture_photos(base_dir, folder_name, num_photos=10):
    cap = cv2.VideoCapture(0)
    folder_path = create_folder(base_dir, folder_name)
    
    count = 1
    while count <= num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Capture', frame)
        
        # Save the frame as an image file
        image_path = os.path.join(folder_path, f'pic{count}.jpg')
        cv2.imwrite(image_path, frame)
        
        count += 1
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_dir = r"M:\FinalYearProject\project\students"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    folder_name = input("Enter the folder name for this session: ")
    num_photos = int(input("Enter the number of photos to capture: "))
    capture_photos(base_dir, folder_name, num_photos)


