#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import skimage.io

import torch as th


def color_mixture_laplacian(N, inInd, neighInd, flows, weights):
  """ """

  row_idx = np.tile(inInd, (1, flows.shape[1]))
  col_idx = neighInd
  Wcm = sp.coo_matrix((np.ravel(flows), (np.ravel(row_idx), np.ravel(col_idx))), shape=(N, N))

  Wcm = sp.spdiags(np.ravel(weights), 0, N, N).dot(Wcm)
  Lcm = sp.spdiags(np.ravel(np.sum(Wcm, axis=1)), 0, N, N) - Wcm
  Lcm = (Lcm.T).dot(Lcm)

  return Lcm


def matting_laplacian(N, inInd, flowRows, flowCols, flows, weights):
  """ """

  weights = np.ravel(weights)[inInd]
  nweights = weights.size
  flow_sz = flows.shape[0]
  flows *= np.tile(np.reshape(weights, (1, 1, nweights)), [flow_sz, flow_sz, 1])
  Wmat = sp.coo_matrix((np.ravel(flows), (np.ravel(flowRows), np.ravel(flowCols))), shape=(N, N))
  Wmat = (Wmat + Wmat.T)*0.5
  Lmat = sp.spdiags(np.ravel(np.sum(Wmat, 1)), 0, N, N) - Wmat
  return Lmat


def similarity_laplacian(N, inInd, neighInd, flows, weights):
  """ """

  weights = np.ravel(weights)[inInd]
  nweights = weights.size
  flow_sz = flows.shape[1]
  flows *= np.tile(weights,[1, flow_sz])
  inInd = np.tile(inInd, [1, neighInd.shape[1]])
  Wcs = sp.coo_matrix((np.ravel(flows), (np.ravel(inInd), np.ravel(neighInd))), shape=(N, N))
  Wcs = (Wcs + Wcs.T)*0.5
  Lcs = sp.spdiags(np.ravel(np.sum(Wcs, 1)), 0, N, N) - Wcs
  return Lcs


def convert_index(old, h, w):
  idx = np.unravel_index(old, [w, h])
  new = np.ravel_multi_index((idx[1], idx[0]), (h, w))
  return new


def main(args):
  data = scipy.io.loadmat(args.ifm_data)['IFMdata']

  CM_inInd    = data['CM_inInd'][0][0]
  CM_neighInd = data['CM_neighInd'][0][0]
  CM_flows    = data['CM_flows'][0][0]

  LOC_inInd    = data['LOC_inInd'][0][0]
  LOC_flowRows = data['LOC_flowRows'][0][0]
  LOC_flowCols = data['LOC_flowCols'][0][0]
  LOC_flows    = data['LOC_flows'][0][0]

  IU_inInd    = data['IU_inInd'][0][0]
  IU_neighInd = data['IU_neighInd'][0][0]
  IU_flows    = data['IU_flows'][0][0]

  kToU = data['kToU'][0][0]
  kToUconf = data['kToUconf'][0][0]

  known = data['known'][0][0]

  h, w = kToU.shape
  N = h*w

  # Convert indices from matlab to numpy format
  CM_inInd     = convert_index(CM_inInd, h, w)
  CM_neighInd  = convert_index(CM_neighInd, h, w)
  LOC_inInd    = convert_index(LOC_inInd, h, w)
  LOC_flowRows = convert_index(LOC_flowRows, h, w)
  LOC_flowCols = convert_index(LOC_flowCols, h, w)
  IU_inInd     = convert_index(IU_inInd, h, w)
  IU_neighInd  = convert_index(IU_neighInd, h, w)

  CM_weights  = np.ones((N,))
  LOC_weights = np.ones((N,))
  IU_weights  = np.ones((N,))
  KU_weights  = np.ones((N,))

  cm_mult  = 1;
  loc_mult = 1;
  iu_mult  = 0.01;
  ku_mult  = 0.05;
  lmbda    = 100;

  print("Assembling linear system")

  A = cm_mult*color_mixture_laplacian(N, CM_inInd, CM_neighInd, CM_flows, CM_weights) + \
      loc_mult*matting_laplacian(N, LOC_inInd, LOC_flowRows, LOC_flowCols, LOC_flows, LOC_weights) + \
      iu_mult*similarity_laplacian(N, IU_inInd, IU_neighInd, IU_flows, IU_weights) + \
      ku_mult*sp.spdiags(np.ravel(KU_weights), 0, N, N).dot(sp.spdiags(np.ravel(kToUconf), 0, N, N)) + \
      lmbda*sp.spdiags(np.ravel(known).astype(np.float64), 0, N, N)

  b = (ku_mult*sp.spdiags(np.ravel(KU_weights), 0, N, N).dot(sp.spdiags(np.ravel(kToUconf), 0, N, N)) + \
      lmbda*sp.spdiags(np.ravel(known).astype(np.float64), 0, N, N)).dot(np.ravel(kToU))

  print("Solving")
  alpha, info = cg(A, b, tol=1e-6, maxiter=2000)
  print(np.amin(alpha), " ", np.amax(alpha))

  alpha = np.clip(alpha, 0, 1)
  alpha = np.reshape(alpha, (h, w))

  print(info)
  skimage.io.imsave("alpha.png", alpha)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("ifm_data")
  args = parser.parse_args()
  main(args)
