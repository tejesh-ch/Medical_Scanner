from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        fill_mode = 'nearest')

for j in range (826):
    glioma_scan = j + 1
    img = load_img(f'Training/glioma_tumor/gg ({str(glioma_scan)}).jpg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0

    generated_data = datagen.flow(x, batch_size=1,
                                  save_to_dir='Augmented/glioma_augmented', save_prefix='glioma_tumor', save_format='jpeg')
    for batch in generated_data:
        i += 1
        if i > 20:
            brea
