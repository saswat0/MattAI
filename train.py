#!/usr/bin/env python

import argparse
import logging
import os
import setproctitle
import time

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

import matting.dataset as dataset
import matting.modules as modules

import torchlib.viz as viz
from torchlib.utils import save
from torchlib.utils import make_variable
from torchlib.image import crop_like

log = logging.getLogger(__name__)


PROCESS_NAME = "automatting"


def main(args, params):
  data = dataset.MattingDataset(args.data_dir, transform=dataset.ToTensor())
  val_data = dataset.MattingDataset(args.data_dir, transform=dataset.ToTensor())

  if len(data) == 0:
    log.info("no input files found, aborting.")
    return

  dataloader = DataLoader(data, 
      batch_size=1,
      shuffle=True, num_workers=4)

  val_dataloader = DataLoader(val_data, 
      batch_size=1, shuffle=True, num_workers=0)

  log.info("Training with {} samples".format(len(data)))

  # Starting checkpoint file
  checkpoint = os.path.join(args.output, "checkpoint.ph")
  if args.checkpoint is not None:
    checkpoint = args.checkpoint

  chkpt = None
  if os.path.isfile(checkpoint):
    log.info("Resuming from checkpoint {}".format(checkpoint))
    chkpt = th.load(checkpoint)
    params = chkpt['params']  # override params

  log.info("Model parameters: {}".format(params))

  model = modules.get(params)

  # loss_fn = modules.CharbonnierLoss()
  loss_fn = modules.AlphaLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  global_step = 0

  if chkpt is not None:
    model.load_state_dict(chkpt['model_state'])
    optimizer.load_state_dict(chkpt['optimizer'])
    global_step = chkpt['step']

  # Destination checkpoint file
  checkpoint = os.path.join(args.output, "checkpoint.ph")

  name = os.path.basename(args.output)
  loss_viz = viz.ScalarVisualizer("loss", env=name)
  image_viz = viz.BatchVisualizer("images", env=name)
  matte_viz = viz.BatchVisualizer("mattes", env=name)
  weights_viz = viz.BatchVisualizer("weights", env=name)
  trimap_viz = viz.BatchVisualizer("trimap", env=name)

  log.info("Model: {}\n".format(model))

  model.cuda()
  loss_fn.cuda()

  log.info("Starting training from step {}".format(global_step))

  smooth_loss = 0
  smooth_loss_ifm = 0
  smooth_time = 0
  ema_alpha = 0.9
  last_checkpoint_time = time.time()
  try:
    epoch = 0
    while True:
      # Train for one epoch
      for step, batch in enumerate(dataloader):
        batch_start = time.time()
        frac_epoch =  epoch+1.0*step/len(dataloader)

        batch_v = make_variable(batch, cuda=True)

        optimizer.zero_grad()
        output = model(batch_v)
        target = crop_like(batch_v['matte'], output)
        ifm = crop_like(batch_v['vanilla'], output)
        loss = loss_fn(output, target)
        loss_ifm = loss_fn(ifm, target)

        loss.backward()
        # th.nn.utils.clip_grad_norm(model.parameters(), 1e-1)
        optimizer.step()
        global_step += 1

        batch_end = time.time()
        smooth_loss = (1.0-ema_alpha)*loss.data[0] + ema_alpha*smooth_loss
        smooth_loss_ifm = (1.0-ema_alpha)*loss_ifm.data[0] + ema_alpha*smooth_loss_ifm
        smooth_time = (1.0-ema_alpha)*(batch_end-batch_start) + ema_alpha*smooth_time

        if global_step % args.log_step == 0:
          log.info("Epoch {:.1f} | loss = {:.7f} | {:.1f} samples/s".format(
            frac_epoch, smooth_loss, target.shape[0]/smooth_time))

        if args.viz_step > 0 and global_step % args.viz_step == 0:
          model.train(False)
          for val_batch in val_dataloader:
            val_batchv = make_variable(val_batch, cuda=True)
            output = model(val_batchv)
            target = crop_like(val_batchv['matte'], output)
            vanilla = crop_like(val_batchv['vanilla'], output)
            val_loss = loss_fn(output, target)

            mini, maxi = target.min(), target.max()

            diff = (th.abs(output-target))
            vizdata = th.cat((target, output, vanilla, diff), 0)
            vizdata = (vizdata-mini)/(maxi-mini)
            imgs = np.power(np.clip(vizdata.cpu().data, 0, 1), 1.0/2.2)

            image_viz.update(val_batchv['image'].cpu().data, per_row=1)
            trimap_viz.update(val_batchv['trimap'].cpu().data, per_row=1)
            weights = model.predicted_weights.permute(1, 0, 2, 3)
            new_w = []
            means = []
            var = []
            for ii in range(weights.shape[0]):
              w = weights[ii:ii+1, ...]
              mu = w.mean()
              sigma = w.std()
              new_w.append(0.5*((w-mu)/(2*sigma)+1.0))
              means.append(mu.data.cpu()[0])
              var.append(sigma.data.cpu()[0])
            weights = th.cat(new_w, 0)
            weights = th.clamp(weights, 0, 1)
            weights_viz.update(weights.cpu().data,
                caption="CM {:.4f} ({:.4f})| LOC {:.4f} ({:.4f}) | IU {:.4f} ({:.4f}) | KU {:.4f} ({:.4f})".format(
                  means[0], var[0],
                  means[1], var[1],
                  means[2], var[2],
                  means[3], var[3]), per_row=4)
            matte_viz.update(
                imgs,
                caption="Epoch {:.1f} | loss = {:.6f} | target, output, vanilla, diff".format(
                  frac_epoch, val_loss.data[0]), per_row=4)
            log.info("  viz at step {}, loss = {:.6f}".format(global_step, val_loss.cpu().data[0]))
            break  # Only one batch for validation

          losses = [smooth_loss, smooth_loss_ifm]
          legend = ["ours", "ref_ifm"]
          loss_viz.update(frac_epoch, losses, legend=legend)

          model.train(True)

        if batch_end-last_checkpoint_time > args.checkpoint_interval:
          last_checkpoint_time = time.time()
          save(checkpoint, model, params, optimizer, global_step)


      epoch += 1
      if args.epochs > 0 and epoch >= args.epochs:
        log.info("Ending training at epoch {} of {}".format(epoch, args.epochs))
        break

  except KeyboardInterrupt:
    log.info("training interrupted at step {}".format(global_step))
    checkpoint = os.path.join(args.output, "on_stop.ph")
    save(checkpoint, model, params, optimizer, global_step)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--val_data_dir')
  parser.add_argument('--checkpoint')
  parser.add_argument('--epochs', type=int, default=-1)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0)
  parser.add_argument('--debug', dest="debug", action="store_true")
  parser.add_argument('--params', nargs="*", default=["model=MattingCNN"])

  parser.add_argument('--log_step', type=int, default=25)
  parser.add_argument('--checkpoint_interval', type=int, default=1200, help='in seconds')
  parser.add_argument('--viz_step', type=int, default=5000)
  parser.set_defaults(debug=False)
  args = parser.parse_args()

  params = {}
  if args.params is not None:
    for p in args.params:
      k, v = p.split("=")
      if v.isdigit():
        v = int(v)
      params[k] = v

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle('{}_{}'.format(PROCESS_NAME, os.path.basename(args.output)))


  main(args, params)
