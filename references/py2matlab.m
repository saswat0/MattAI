function idx = py2matlab(idx0, h, w)
  col = mod(idx0, w);
  row = idivide(idx0, w);
  idx = 1 + row + h*col;
end
