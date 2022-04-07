from imutils import paths # Do łatwego tworzenia listy plików
import cv2 # Biblioteka OpenCV do operacji na obrazach
           # pip install opencv-python
import numpy as np

# Nazwy plików z obrazami z katalogu 'images'
images_paths = list(paths.list_images('images'))

image_data = [] # Pusta lista na obrazy
for image_path in images_paths:
    print("Wczytuję dane z pliku",image_path)
    # Załadowanie zdjęcia z pliku (OpenCV > 2.2)
    # Normalizacja do przedziału [0, 1]
    image = cv2.imread(image_path)/255.0
    # Dodanie kolejnego obrazu do listy obrazów
    image_data.append(image)
    
# Końcowa konwersja na ndarray
data = np.array(image_data)

