import argparse

from data.process_image import PatchImage


def create_patches():
    root_original = args.path_original
    destination = args.path_destination
    patch_size = args.patch_size
    overlap_size = args.overlap_size

    patcher = PatchImage(patch_size=patch_size,
                         overlap_size=overlap_size,
                         destination_root=destination)
    patcher.create_patches(root_original=root_original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create patches')

    parser.add_argument('-destination', '--path_destination',
                        metavar='<path>',
                        type=str,
                        help=f"destination folder path with contains the patches")
    parser.add_argument('-original', '--path_original',
                        metavar='<path>',
                        type=str,
                        help="the path witch contains the ruined images")
    parser.add_argument('-size', '--patch_size',
                        metavar='<number>',
                        type=int,
                        help="size of ruined patch",
                        default=384)
    parser.add_argument('-overlap', '--overlap_size',
                        metavar='<number>',
                        type=int,
                        help='overlap_size',
                        default=192)

    args = parser.parse_args()

    create_patches()
