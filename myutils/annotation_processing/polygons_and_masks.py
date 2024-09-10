# Note these aren't currently used.


def masks_to_polygons(binary_mask):
    """
    Example of converting binary masks to polygons

    :param binary_mask: Binary mask image where shapes are represented as white (255) regions on a black (0) background.

    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    cv2.rectangle(binary_mask, (30, 30), (70, 70), 1, -1)  # Draw a filled rectangle

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the polygons
    polygons_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Draw the contours (polygons)
    for contour in contours:
        cv2.polylines(polygons_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert contours to polygons (list of points)
    polygons = [contour.reshape(-1, 2).tolist() for contour in contours]

    # Display the original binary mask and the polygons
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Binary Mask')
    plt.imshow(binary_mask, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Polygons')
    plt.imshow(polygons_image)

    plt.show()

    # Print the polygon coordinates
    for i, polygon in enumerate(polygons):
        print(f"Polygon {i + 1}: {polygon}")


def polygons_to_masks(polygons):
    """
    Example of converting polygons to binary masks

    :param polygons: A list of polygons, where each polygon is represented as a list of points (e.g., numpy array).

    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the image size
    height, width = 100, 100

    # Create an empty binary mask
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the polygons on the binary mask
    for polygon in polygons:
        cv2.fillPoly(binary_mask, [polygon], 1)

    # Display the binary mask
    plt.figure(figsize=(6, 6))
    plt.title('Binary Mask from Polygons')
    plt.imshow(binary_mask, cmap='gray')
    plt.show()


if __name__ == '__main__':
    import numpy as np

    # Create a binary mask (for demonstration purposes)
    bin_mask = np.zeros((100, 100), dtype=np.uint8)
    masks_to_polygons(bin_mask)

    # Define polygons (example polygons)
    example_polygons = [
        np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32),  # Rectangle
        np.array([[10, 10], [20, 10], [15, 20]], dtype=np.int32)  # Triangle
    ]
    polygons_to_masks(example_polygons)
