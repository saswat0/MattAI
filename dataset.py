import logging
import os
import re
import time
import ipdb
import numpy as np
import scipy.io
import scipy.sparse as sp
import skimage.io
import torch as th
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class MattingDataset(Dataset):
  """"""

  def __init__(self, root_dir, transform=None):
    super(MattingDataset, self).__init__()
    self.transform = transform

    self.root_dir = root_dir
    self.ifm_data_dir = os.path.join(root_dir, 'IFMData')
    # self.ifm_data_dir = os.path.join(root_dir, 'IFMData1_overkill')
    self.matte_dir = os.path.join(root_dir, 'alpha')
    self.images_dir = os.path.join(root_dir, 'images')
    self.trimap_dir = os.path.join(root_dir, 'trimap')
    self.vanilla_dir = os.path.join(root_dir, 'vanilla')

    files = os.listdir(self.images_dir)
    data_regex = re.compile(r".*.(png|jpg|jpeg)$")
    files = sorted([f for f in files if data_regex.match(f)])

    fid = open("missing_files.txt", 'w')

    start = time.time()

    self.files = []
    for f in files:
      if not os.path.exists(self.ifm_path(f)):
        fid.write(self.ifm_path(f))
        fid.write("\n")
        continue
      if not os.path.exists(self.matte_path(f)):
        fid.write(self.matte_path(f))
        fid.write("\n")
        continue
      if not os.path.exists(self.trimap_path(f)):
        fid.write(self.trimap_path(f))
        fid.write("\n")
        continue
      self.files.append(f)
    fid.close()

    duration = time.time() - start

    log.debug("Parsed dataset {} with {} samples in {:.2f}s".format(
      root_dir, len(self), duration))


  def image_path(self, f):
    return os.path.join(self.images_dir, f)

  def basename(self, f):
    fname = os.path.splitext(f)[0]
    basename = fname
    # basename = "_".join(fname.split("_")[:-1])
    return basename

  def ifm_path(self, f):
    return os.path.join(self.ifm_data_dir, os.path.splitext(f)[0]+".mat")

  def matte_path(self, f):
    return os.path.join(self.matte_dir, self.basename(f)+".png")

  def vanilla_path(self, f):
    return os.path.join(self.vanilla_dir, self.basename(f)+".png")

  def trimap_path(self, f):
    return os.path.join(self.trimap_dir, self.basename(f)+".png")

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    start = time.time()
    fname = self.files[idx]

    matte = skimage.io.imread(self.matte_path(fname)).astype(np.float32)[:, :, 0:1]/255.0
    vanilla = skimage.io.imread(self.vanilla_path(fname)).astype(np.float32)[:, :, np.newaxis]/255.0
    image = skimage.io.imread(self.image_path(fname)).astype(np.float32)/255.0
    trimap = skimage.io.imread(self.trimap_path(fname)).astype(np.float32)/255.0
    image = image.transpose([2, 0, 1])
    matte = matte.transpose([2, 0, 1])
    vanilla = vanilla.transpose([2, 0, 1])
    trimap = np.expand_dims(trimap, 0)

    data = scipy.io.loadmat(self.ifm_path(fname))["IFMdata"]

    CM_inInd    = data['CM_inInd'][0][0].astype(np.int64)  # NOTE(mgharbi): these are saved as floats
    CM_neighInd = data['CM_neighInd'][0][0].astype(np.int64)
    CM_flows    = data['CM_flows'][0][0]

    LOC_inInd    = data['LOC_inInd'][0][0].astype(np.int64)
    LOC_flows    = data['LOC_flows'][0][0]
    # LOC_flows1    = data['LOC_flows1'][0][0]
    # LOC_flows2    = data['LOC_flows2'][0][0]
    # LOC_flows3    = data['LOC_flows3'][0][0]
    # LOC_flows4    = data['LOC_flows4'][0][0]
    # LOC_flows5    = data['LOC_flows5'][0][0]
    # LOC_flows6    = data['LOC_flows6'][0][0]
    # LOC_flows7    = data['LOC_flows7'][0][0]
    # LOC_flows8    = data['LOC_flows8'][0][0]
    # LOC_flows9    = data['LOC_flows9'][0][0]

    IU_inInd    = data['IU_inInd'][0][0].astype(np.int64)
    IU_neighInd = data['IU_neighInd'][0][0].astype(np.int64)
    IU_flows    = data['IU_flows'][0][0]

    kToU = data['kToU'][0][0]
    kToUconf = np.ravel(data['kToUconf'][0][0])

    known = data['known'][0][0].ravel()

    h, w = kToU.shape
    N = h*w

    kToU = np.ravel(kToU)

    # Convert indices from matlab to numpy format
    CM_inInd     = self.convert_index(CM_inInd, h, w)
    CM_neighInd  = self.convert_index(CM_neighInd, h, w)
    LOC_inInd    = self.convert_index(LOC_inInd, h, w)
    IU_inInd     = self.convert_index(IU_inInd, h, w)
    IU_neighInd  = self.convert_index(IU_neighInd, h, w)

    Wcm = self.color_mixture(N, CM_inInd, CM_neighInd, CM_flows)
    sample = {
        "Wcm_row": np.squeeze(Wcm.row),
        "Wcm_col": np.squeeze(Wcm.col),
        "Wcm_data": np.squeeze(Wcm.data),
        "LOC_inInd": LOC_inInd,
        "LOC_flows": LOC_flows,
        # "LOC_flows1": LOC_flows1,
        # "LOC_flows2": LOC_flows2,
        # "LOC_flows3": LOC_flows3,
        # "LOC_flows4": LOC_flows4,
        # "LOC_flows5": LOC_flows5,
        # "LOC_flows6": LOC_flows6,
        # "LOC_flows7": LOC_flows7,
        # "LOC_flows8": LOC_flows8,
        # "LOC_flows9": LOC_flows9,
        "IU_inInd": IU_inInd,
        "IU_neighInd": IU_neighInd,
        "IU_flows": IU_flows,
        "kToUconf": kToUconf,
        "known": known,
        "kToU": kToU,
        "height": h,
        "width": w,
        "image": image,
        "matte": matte,
        "vanilla": vanilla,
        "trimap": trimap,
    }

    if self.transform is not None:
      sample = self.transform(sample)

    end = time.time()
    log.debug("load sample {:.2f}s/im".format((end-start)))
    return sample

  def convert_index(self, old, h, w):
    if not (old-1 >=0).all():
      print "invalid index", np.amin(old-1)
      import ipdb; ipdb.set_trace()
    try:
      idx = np.unravel_index(old-1, [w, h])
    except ValueError as e:
      print e
      ipdb.set_trace()
    new = np.ravel_multi_index((idx[1], idx[0]), (h, w)).astype(np.int32)
    return new

  def color_mixture(self, N, inInd, neighInd, flows):
    row_idx = np.tile(inInd, (1, flows.shape[1]))
    col_idx = neighInd
    Wcm = sp.coo_matrix(
        (np.ravel(flows), (np.ravel(row_idx), np.ravel(col_idx))), shape=(N, N))
    return Wcm


class ToTensor(object):
  """Convert sample ndarrays to tensors."""

  def __call__(self, sample):
    xformed = {}
    for k in sample.keys():
      if type(sample[k]) == np.ndarray:
        xformed[k] = th.from_numpy(sample[k])
      else:
        xformed[k] = sample[k]
    return xformed
