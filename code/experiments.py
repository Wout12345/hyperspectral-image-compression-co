from tools import *
import st_hosvd

# Hoofdstuk 3: Methodologie

def save_cuprite_image():
	data = load_cuprite()
	image = np.sum(data, axis=2)
	imsave("../tekst/images/cuprite_sum.png", np.rint(image/np.amax(image)*255).astype(int))

# Hoofdstuk 4: De Tuckerdecompositie

def randomized_svd():
	
	pass

save_cuprite_image()
