import matplotlib.pyplot as plt
from proccess import preprocess_image, deprocess_image


img1 = preprocess_image("./原始图片.png")
img1 = deprocess_image(img1)
img2 = preprocess_image("./result_2.png")
img2 = deprocess_image(img2)
img3 = preprocess_image("image_data/fangao.jpg")
img3 = deprocess_image(img3)
plt.subplot(221)
plt.title("origin")
plt.imshow(img1)
plt.subplot(222)
plt.title("1")
plt.imshow(img2)
plt.subplot(223)
plt.title("2")
plt.imshow(img3)
plt.show()