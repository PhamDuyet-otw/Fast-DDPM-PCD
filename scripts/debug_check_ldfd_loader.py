import argparse
from datasets.LDFDCT import LDFDCT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    dataset = LDFDCT(args.manifest, args.img_size, split=args.split)
    print("Dataset length:", len(dataset))

    sample = dataset[0]
    ld = sample["LD"]
    fd = sample["FD"]

    print("case_name:", sample["case_name"])
    print("LD:", ld.shape, ld.dtype, float(ld.min()), float(ld.max()))
    print("FD:", fd.shape, fd.dtype, float(fd.min()), float(fd.max()))

    assert ld.shape == (1, args.img_size, args.img_size)
    assert fd.shape == (1, args.img_size, args.img_size)
    assert -1.001 <= float(ld.min()) <= 1.001
    assert -1.001 <= float(ld.max()) <= 1.001
    assert -1.001 <= float(fd.min()) <= 1.001
    assert -1.001 <= float(fd.max()) <= 1.001

    print("OK: loader works.")


if __name__ == "__main__":
    main()
