import io

import cv2
import datasets
import numpy as np
from PIL import Image


def main():
    ds = datasets.load_dataset(
        "jxu124/OpenX-Embodiment",
        "fractal20220817_data",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    print("Searching for the first sample...")

    for sample in ds.take(1):
        data = sample["data.pickle"]

        steps = data["steps"]
        video_name = "episode_video.mp4"
        fps = 10
        video_writer = None

        for i, step in enumerate(steps):
            img_dict = step["observation"]["image"]

            for key, value in img_dict.items():
                if isinstance(value, bytes):
                    img = Image.open(io.BytesIO(value))
                    img_np = np.array(img)

                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    if video_writer is None:
                        height, width, _ = img_bgr.shape
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            video_name, fourcc, fps, (width, height)
                        )

                    video_writer.write(img_bgr)

                    if i % 10 == 0:
                        print(f"Frame {i}/{len(steps)} processed...")

        if video_writer:
            video_writer.release()
            print(f"Finished! Video saved as {video_name}")


if __name__ == "__main__":
    main()
