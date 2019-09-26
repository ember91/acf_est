function test()
    tester(single(randn(2^4, 1)));
    tester(double(randn(2^4, 1)));
    tester(single(randn(1, 2^4)));
    tester(double(randn(1, 2^4)));
    tester(single(randn(2^4 - 1, 1)));
    tester(double(randn(2^4 - 1, 1)));
    tester(single(randn(2^4 - 2, 1)));
    tester(double(randn(2^4 - 2, 1)));
    tester(single(randn(2^4 - 3, 1)));
    tester(double(randn(2^4 - 3, 1)));
    tester(single(randn(2^4 - 7, 1)));
    tester(double(randn(2^4 - 7, 1)));
    tester(single(randn(2^4 - 7, 1)));
    tester(double(randn(2^4 - 7, 1)));
    tester(single(randn(2^4, 4)));
    tester(double(randn(2^4, 4)));
    tester(single(randn(2^4 - 1, 7)));
    tester(double(randn(2^4 - 1, 7)));
    tester(single(randn(2^15, 4)));
    tester(double(randn(2^15, 4)));
end

function tester(gauss)
  sz = size(gauss);
  M = sz(1);
  N = sz(2);

  disp('----------------')
  if isa(gauss, 'double')
      type = 'double';
  else
      type = 'single';
  end
  disp(['Test ' type ' ' num2str(M) 'x' num2str(N)]);

  t = zeros(2, 1);
  
  tic;
  r1 = bartlett_simple(gauss);
  t(1) = toc;
  
  tic;
  r2 = acf_est(gauss);
  if M == 1
    r2 = [fliplr(r2(1, 2:end)) r2];
  else
    r2 = [flipud(r2(2:end, :)); r2];
  end
  t(2) = toc;

  disp(['Elapsed time ' num2str(t(1)) ' ' num2str(t(2)) 's']);
  disp(['Speedup ' num2str(t(1) / t(2))]);
  
  err2 = sum(sum(abs(r1 - r2)));
  if isa(gauss, 'double')
      err1 = 1e-15*M*N;
  else
      err1 = 1e-7*M*N;
  end
  
  disp(['Error ' num2str(err2) ' <= ' num2str(err1)]);
  
  if err2 > err1
      error('Results differ :(');
  end
end

function r = bartlett_simple(x)
  N = size(x, 1);
  C = size(x, 2);
  if N ~= 1
      r = zeros(2*N - 1, C, 'like', x);
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
  else
      N = C;
      r = zeros(1, 2*N-1, 'like', x);
      for k = 1:N
          s = 0;
          for n = 0:N - k
              s = s + x(n + k)*x(n + 1);
          end
          s = s/N;
          r(N - 1 + k) = s;
          r(N + 1 - k) = s;
      end
  end
end
