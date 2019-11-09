# testing set

test_set = KangarooDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()

# load image
image_id = 0
img = train_set.load_image(image_id)

# load mask
mask, class_id = train_set.load_mask(image_id)

# plt.imshow(img)
# plt.imshow(mask[:, :, 0], cmap='jet', alpha=0.5)

for i in range(9):
    plt.subplot(330 + 1 + i)
    img = train_set.load_image(i)
    plt.imshow(img)
    mask, _ = train_set.load_mask(i)

    for j in range(mask.shape[2]):
        plt.imshow(mask[:, :, j], cmap='jet', alpha=0.3)

# plt.show()

for image in train_set.image_ids:
    info = train_set.image_info[image]
    print(info)

