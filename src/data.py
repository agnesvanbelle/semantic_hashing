import numpy as np

#np.random.seed(102)

def mvn_rand(mu_delta, nr_samples, dims=4):
  mu = np.random.random_sample(dims) + mu_delta
  A = np.random.random_sample(dims * dims).reshape(-1, dims)
  return np.random.multivariate_normal(mu, A.T @ A, size=nr_samples)

def make_sample(nr_samples):
  xsample = np.vstack((mvn_rand(3, nr_samples), mvn_rand(0, nr_samples), 
                     mvn_rand(-3, nr_samples)))
  ysample = np.vstack((mvn_rand(3, nr_samples), mvn_rand(0, nr_samples), 
                     mvn_rand(-3, nr_samples)))
  col = np.repeat(['red' , 'green', 'blue'], nr_samples)
  return (xsample, ysample, col)

if __name__== "__main__":
  print(" hi" )
  
  print(make_sample(1))