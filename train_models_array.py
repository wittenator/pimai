from train import train
import itertools

for warmup_method,reparametrization in  itertools.product(['cycle', 'tanh', 'none'], ['km', 'gamma']):
	train(warmup_method=warmup_method,reparametrization=reparametrization,max_epoch=500)
