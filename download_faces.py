import os
import argparse
from PIL import Image
import requests
import pickle
import multiprocessing
import threading
from functools import partial
from tqdm import tqdm
import logging


data_files = {
    "2021-07-03": dict(file_path="naver_webtoon_face_data_2021-07-03.pkl", url="https://onedrive.live.com/download?cid=467C8AA2DE5C1D02&resid=467C8AA2DE5C1D02%21160&authkey=AAw9aFNjEndsdeY"),
}


def download_data_file(file_path, url):
    data = requests.get(url, stream=True)
    with open(file_path, "wb") as file:  
        file.write(data.content)


def get_image_url(title_id, image_name):
    return f"https://image-comic.pstatic.net/webtoon/{title_id}/{image_name}"


def download_image(session, url):
    try:
        image_data = session.get(url, stream=True).raw
        image = Image.open(image_data).convert('RGB')
        return image
    except:
        return None


def download_faces_worker_fn(session, result_list, image_info):
    title_id, image_name, box_info = image_info
    image_url = get_image_url(title_id, image_name)
    image = download_image(session, image_url)
    for idx, box, score in box_info:
        face = image.crop(box) if image is not None else None
        result_list.append((title_id, image_name, idx, face))   


def download_faces(result_list, images_to_download, n_workers=4):
    session = requests.Session()
    session.headers.update({
      'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
    })
    worker_fn = partial(download_faces_worker_fn, session, result_list)
    p = multiprocessing.Pool(processes=n_workers)
    p.map(worker_fn, images_to_download) 
    p.close()
    p.join()


def get_logger(path):
    logger = logging.getLogger("Naver Webtoon Downloader")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(path, "log.txt"), mode="w"))
    logger.propagate = False
    return logger

def run(args):
    save_path = "./faces"
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(save_path)

    # load data info
    assert args.version in data_files, f"wrong data version: {args.version}"
    data_file = data_files[args.version]
    if not os.path.isfile(data_file["file_path"]):
        print(f"downloading file: {data_file['file_path']}")
        download_data_file(**data_file)

    with open(data_file["file_path"], "rb") as f:
        data = pickle.load(f)
    logger.info(f"version: {args.version}")
    print(f"version: {args.version}")
    print(f"titles: {len(data)}")

    # check data to download
    titles_to_download = args.titles
    if titles_to_download == ["all"]:
        titles_to_download = list(data)
    print(f"{len(titles_to_download)} titles will be downloaded:")
    for title_id in titles_to_download:
        assert title_id in data, f"wrong title id: {title_id}"
        print(f"  {title_id:8} ({data[title_id]['title']}, {data[title_id]['author']})")
        os.makedirs(os.path.join(save_path, title_id), exist_ok=True)

    images_to_download = [
        (title_id, image_name, box_info)
        for title_id in titles_to_download
        for image_name, box_info in data[title_id]["faces"].items()
    ]
    n_data = sum([len(image_info[2]) for image_info in images_to_download])

    # waifu2x
    if not args.save_raw_crops:
        import torch
        from waifu2x import Waifu2x
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waifu2x = Waifu2x().to(device)

    # background: download full images -> crop faces -> append to list
    manager = multiprocessing.Manager()
    cropped_faces = manager.list()
    download_thread = threading.Thread(
        target=download_faces,
        args=(cropped_faces, images_to_download),
        daemon=True
    )
    download_thread.start()

    # main thread: post-process cropped images and save
    # TODO: when using [cpu/no post-processing], process & save directly in worker_fn
    n_processesed = 0
    n_failed = 0
    with tqdm(total=n_data) as pbar:
        while n_processesed < n_data:
            if len(cropped_faces) < 1:
                continue
            else:
                title_id, image_name, idx, face = cropped_faces.pop()
                n_processesed += 1
                pbar.update(1)
                if face is None:
                    logger.info(f"failed: {title_id} {image_name} {idx}")
                    n_failed += 1
                    continue

                if not args.save_raw_crops:
                    face = waifu2x(face).resize((256, 256), Image.ANTIALIAS)
                face.save(os.path.join(save_path, title_id, idx + ".png"))
    
    result = f"{n_data - n_failed}/{n_data} face images saved, {n_failed} failed."
    logger.info(result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",        help="dataset version", type=str, default="2021-07-03")
    parser.add_argument("--titles",         help="title ids to download, use [all] to save all", nargs='+', default=["703846", "597447", "702608"])
    parser.add_argument("--save_raw_crops", help="to save cropped faces without post-processing", action="store_true")
    args = parser.parse_args()
    run(args)
