function test()
  N = 2^12;

  itrs = 12;
  gauss = randn(N, itrs);

  t = zeros(2, 1);
  
  tic;
  r1 = acf2(gauss);
  %r1 = zeros(2*N - 1, itrs);
  t(1) = toc;
  
  tic;
  r2 = acf_est(gauss);
  r2 = [flipud(r2(2:end, :)); r2];
  t(2) = toc;

  disp(['Elapsed time ' num2str(t(1)) ' ' num2str(t(2)) 's']);
  disp(['Speedup ' num2str(t(1) / t(2))]);
  
  err2 = sum(sum(abs(r1 - r2)));
  err1 = 1e-15*N*itrs;
  
  disp(['Error ' num2str(err2) ' <= ' num2str(err1)]);
  
  if err2 > err1
      error('Results differ :(');
  end
end

function r = acf2(x)
  N = size(x, 1);
  C = size(x, 2);
  r = zeros(2*N - 1, C);
  for c = 1:C
      for k = 1:N
          s = 0;
          for n = 0:N - k
              s = s + x(n + k, c)*x(n + 1, c);
          end
          s = s/N;
          r(N - 1 + k, c) = s;
          r(N + 1 - k, c) = s;
      end
  end
end
