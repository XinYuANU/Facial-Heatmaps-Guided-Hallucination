------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------
require 'hdf5'
require 'nngraph'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'
require 'threads'
require 'PerceptionLoss'
require 'preprocess'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
URnet = require 'adversarial_xin_landmarks_v1'

dl = require 'dataload'
lm = require 'dataloader_LM'

model = require 'Model'
----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs128_UR_LM_v2_skip") subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 8)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  --scale            (default 128)          scale of images to train on
  --lambda           (default 0.01)       trade off D and Euclidean distance  
  --eta     	     (default 0.01)       trade off G and perception loss   
  --margin           (default 0)        trade off D and G
  --nStack           (default 4)          number of stack of hourglass
  --nOutChannels     (default 4)		  Component 5 and landmark 68
  --num_stn          (default 1)		  num of STNs  
]]

if opt.margin <= 0 then 
	opt.save = opt.save .. '_noTradeOff'
end

opt.save = opt.save .. string.format('_Stack_%d', opt.nStack)

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)
-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
--[[
-- Dataset Loader 
--]]

ntrain = 34048
nval = 3780

-- load image_dataset
datapath = 'dataset/'
loadsize = {3, opt.scale, opt.scale}
nthread  = opt.threads

-- train_dataset
train_imagefolder_HR = 'face_HR_train'
train_filename_HR = 'filename_HR_list_train.txt'
print('loading training_HR')
train_HR = dl.ImageClassPairs(datapath, train_filename_HR, train_imagefolder_HR, loadsize)

test_imagefolder_HR = 'face_HR_test'
test_filename_HR = 'filename_HR_list_test.txt'
print('loading testing_HR')
test_HR = dl.ImageClassPairs(datapath, test_filename_HR, test_imagefolder_HR, loadsize)

local lowHd5 = hdf5.open('dataset/YTC_LR_train.hdf5', 'r')
local data_LR_train = lowHd5:read('YTC'):all()
data_LR_train:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR_train[{{1, ntrain}}]

lowHd5 = hdf5.open('dataset/YTC_LR_test.hdf5', 'r')
local data_LR_test = lowHd5:read('YTC'):all()
data_LR_test:mul(2):add(-1)
lowHd5:close()
valData_LR = data_LR_test[{{1, nval}}]

landmark_filename = 'filename_landmark.txt'
landmark_path = 'dataset/face_train_LM_Comp_v1'
landmark_loader = lm(datapath, landmark_path, landmark_filename)

-- fix seed
torch.manualSeed(1)

if opt.gpu then
	cutorch.setDevice(opt.gpu + 1)
	print('<gpu> using device ' .. opt.gpu)
	torch.setdefaulttensortype('torch.CudaTensor')
else
	torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

model_D = model.createNetD()
model_G = model.createNetG_skip(opt)

print('Copy model to gpu')
model_D:cuda()
model_G:cuda()  -- convert model to CUDA

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion():cuda()
criterion_G = nn.ParallelCriterion()
for i = 1, opt.nStack+1 do
	criterion_G:add(nn.MSECriterion())
end

vgg_model = createVggmodel()
PerceptionLoss = nn.PerceptionLoss(vgg_model, 1):cuda()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Training parameters
sgdState_D = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	trade_off = false,
	optimize=true,
	numUpdates = 0,
	beta1 = 0.5
}
if opt.margin > 0 then
	sgdState_D.trade_off = true
end

sgdState_G = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates=0,
	beta1 = 0.5
}


idx_test = torch.LongTensor(100)
for ii = 1, 20 do
	idx_test[ii] = ii
end
for ii = 21, 50 do
	idx_test[ii] = 1704 + ii - 20
end
for ii = 51, 100 do
	idx_test[ii] = 1704 + 728 + ii - 50
end

idx_test_comp = torch.LongTensor(50)
for ii = 1, 10 do
	idx_test_comp[ii] = ii
end
for ii = 11, 20 do
	idx_test_comp[ii] = 1704 + ii 
end
for ii = 21, 50 do
	idx_test_comp[ii] = 1704 + 728 + ii 
end

-- Get examples to plot
function getSamples(dataset, N, idx)
	local N = N or 10
	local dataset_HR = dataset
	local inputs   = torch.Tensor(N,3,16,16)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_HR[i]),16,16)
		inputs[i] = dataset[idx[i]]
	end
	
	local samples = model_G:forward(inputs:cuda())
	samples = nn.HardTanh():forward(samples[#samples])
	local to_plot = {}
	for i = 1,N do 
		to_plot[#to_plot+1] = samples[i]:float()
	end
	return to_plot
end

function getSamples_compare(dataset_LR_L1, dataset_HR_fromFile, idx)
	local N = idx:size(1)
	
	dataset_HR = dataset_HR_fromFile:index(idx)
	dataset_HR:mul(2):add(-1)
	
	local inputs = torch.Tensor(N, 3, 16, 16)
	for i = 1,N do 
		inputs[i] = dataset_LR_L1[idx[i]]
	end
	
	local samples = model_G:forward(inputs:cuda())
	samples = nn.HardTanh():forward(samples[#samples])
	
	torch.setdefaulttensortype('torch.FloatTensor')
	dataset_HR = dataset_HR:index(2, torch.LongTensor{3,2,1})
	local to_plot = {}
	for i = 1,N do
		to_plot[#to_plot+1] = samples[i]:float()
		to_plot[#to_plot+1] = dataset_HR[i]:float()
	end
	torch.setdefaulttensortype('torch.CudaTensor')
	return to_plot
end

ntrain_part1 = 17024
ntrain_part2 = ntrain - ntrain_part1
epoch = 1
while true do 
	collectgarbage()
	--local to_plot = getSamples(valData_LR,100, idx_test)
	local to_plot = getSamples_compare(valData_LR, test_HR, idx_test_comp)
	torch.setdefaulttensortype('torch.FloatTensor')
	
	--trainLogger:style{['MSE accuarcy1'] = '-'}
	--trainLogger:plot()
	
	local formatted = image.toDisplayTensor({input = to_plot, nrow = 10})
	formatted:float()
	formatted = formatted:index(1,torch.LongTensor{3,2,1})
	
	image.save(opt.save .. '/UR_example_' .. (epoch-1) .. '.png', formatted)
	
	IDX_part1 = torch.randperm(ntrain_part1)
	IDX_part2 = IDX_part1 + ntrain_part1

	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	URnet.train(trainData_LR,train_HR, landmark_loader)
	
	sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
	sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	opt.lambda = math.max(opt.lambda*0.995, 0.005)   -- or 0.995
end
