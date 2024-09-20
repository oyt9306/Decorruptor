import pixmix_utils as utils
import numpy as np
import torch

def augment_input(image, aug_severity=1):
    aug_list = utils.augmentations_all 
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


def pixmix(orig, mixing_pic, preprocess):
    k, beta = 4, 4
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(k + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, beta)
    mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)


# mixing_set = datasets.ImageFolder(
#     args.mixing_set, 
#     transform=transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224)
#     ])
# )
    

# train_data = ImageNetSubsetDataset(
#         args.data_standard,
#         transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip()
#         ])
# )
# train_dataset = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})


# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=args.batch_size, shuffle=True,
#     num_workers=args.workers, pin_memory=True, sampler=None, worker_init_fn=wif)
