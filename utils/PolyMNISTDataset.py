import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image

# from scipy import ndimage


class PolyMNIST(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, dir_data, num_views, transform=None, target_transform=None):
        """
        Args:
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
        """
        super().__init__()
        self.num_modalities = num_views
        self.dir_data = dir_data
        self.transform = transform
        self.target_transform = target_transform
        self.label_names = "digit"

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in range(self.num_modalities)}
        for dp in range(self.num_modalities):
            files = glob.glob(os.path.join(self.dir_data, "m" + str(dp), "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            print(num_files, len(files))
            assert len(files) == num_files
        self.num_files = num_files

    @staticmethod
    def _create_mmnist_dataset(
        savepath,
        backgroundimagepath,
        num_modalities,
        train,
        rotate_mnist=False,
        translate_mnist=False,
    ):
        """Created the Multimodal MNIST Dataset under 'savepath' given a directory of background images.

        Args:
            savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                exist.
            backgroundimagepath (str): path to a directory filled with background images. One background images is
                used per modality.
            num_modalities (int): number of modalities to create.
            train (bool): create the dataset based on MNIST training (True) or test data (False).
            rotate_mnist (bool): add a random rotation [-45, ..., 45] degree to each digit.
            translate_mnist (bool): downsample MNIST by a factor of 2 and place it at a random x/y-coordinate

        """

        # load MNIST data
        mnist = datasets.MNIST("/tmp", train=train, download=True, transform=None)

        # load background images
        background_filepaths = sorted(
            glob.glob(os.path.join(backgroundimagepath, "*.jpg"))
        )  # TODO: handle more filetypes
        print("\nbackground_filepaths:\n", background_filepaths, "\n")
        if num_modalities > len(background_filepaths):
            raise ValueError(
                "Number of background images must be larger or equal to number of modalities"
            )
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(num_modalities):
            unimodal_path = os.path.join(savepath, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(num_modalities):
                ixs_perm = ixs[
                    torch.randperm(len(ixs))
                ]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = PolyMNIST._add_background_image(
                        background_images[m],
                        mnist.data[ix],
                        rotate_mnist=rotate_mnist,
                        translate_mnist=translate_mnist,
                    )
                    # save as png
                    filepath = os.path.join(savepath, "m%d/%d.%d.png" % (m, i, digit))
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        print(
                            "Saved %d/%d images to %s"
                            % (cnt, len(mnist) * num_modalities, savepath)
                        )
        assert cnt == len(mnist) * num_modalities

    @staticmethod
    def _add_background_image(
        background_image_pil,
        mnist_image_tensor,
        change_colors=False,
        rotate_mnist=False,
        translate_mnist=False,
    ):
        # rotate mnist image
        if rotate_mnist is True:
            deg = np.random.randint(-45, 45)
            mnist_image_pil = Image.fromarray(
                mnist_image_tensor.squeeze(0).numpy(), mode="L"
            )
            mnist_image_pil_rotated = mnist_image_pil.rotate(deg)
            mnist_image_tensor_rotated = (
                transforms.ToTensor()(mnist_image_pil_rotated) * 255.0
            )
            mnist_image_tensor = mnist_image_tensor_rotated

        # translate mnist image: downsamle digit and place it at a random location
        if translate_mnist is True:
            mnist_image_tensor_downsampled = torch.nn.functional.interpolate(
                mnist_image_tensor.unsqueeze(0).float(),
                scale_factor=0.75,
                mode="bilinear",
            )
            mnist_image_tensor = mnist_image_tensor * 0  # black out everything
            x = np.random.randint(0, 21)
            y = np.random.randint(0, 21)
            mnist_image_tensor[:, x : x + 21, y : y + 21] = (
                mnist_image_tensor_downsampled
            )

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(
            torch.bool
        )  # NOTE: mnist is _not_ normalized to [0, 1]

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError(
                "Unexpected dimensionality of MNIST image:", img_binarized.shape
            )

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = [self.file_paths[dp][index] for dp in range(self.num_modalities)]
        labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]

        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]
        if self.target_transform:
            labels = [self.transform(label) for label in labels]

        images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        return (
            images_dict,
            labels[0],
        )  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-modalities", type=int, default=5)
    parser.add_argument("--savepath-train", type=str, required=True)
    parser.add_argument("--savepath-test", type=str, required=True)
    parser.add_argument("--backgroundimagepath", type=str, required=True)
    parser.add_argument("--rotate-mnist", default=False, action="store_true")
    parser.add_argument("--translate-mnist", default=False, action="store_true")
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    PolyMNIST._create_mmnist_dataset(
        args.savepath_train,
        args.backgroundimagepath,
        args.num_modalities,
        train=True,
        rotate_mnist=args.rotate_mnist,
        translate_mnist=args.translate_mnist,
    )
    PolyMNIST._create_mmnist_dataset(
        args.savepath_test,
        args.backgroundimagepath,
        args.num_modalities,
        train=False,
        rotate_mnist=args.rotate_mnist,
        translate_mnist=args.translate_mnist,
    )
    print("Done.")
